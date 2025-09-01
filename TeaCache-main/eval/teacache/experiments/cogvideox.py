from utils import generate_func, read_prompt_list
from videosys import CogVideoXConfig, VideoSysEngine
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from typing import Any, Dict, Optional, Tuple,  Union
from videosys.core.comm import all_to_all_with_pad, gather_sequence, get_pad, set_pad, split_sequence
from videosys.models.transformers.cogvideox_transformer_3d import Transformer2DModelOutput
from videosys.utils.utils import batch_func
from functools import partial
from diffusers.utils import is_torch_version

def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        all_timesteps=None
    ):
        if self.parallel_manager.cp_size > 1:
            (
                hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_cond,
                image_rotary_emb,
            ) = batch_func(
                partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0),
                hidden_states,
                encoder_hidden_states,
                timestep,
                timestep_cond,
                image_rotary_emb,
            )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        org_timestep = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)

        # 3. Position embedding
        text_seq_length = encoder_hidden_states.shape[1]
        if not self.config.use_rotary_positional_embeddings:
            seq_length = height * width * num_frames // (self.config.patch_size**2)

            pos_embeds = self.pos_embedding[:, : text_seq_length + seq_length]
            hidden_states = hidden_states + pos_embeds
            hidden_states = self.embedding_dropout(hidden_states)

        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if self.enable_teacache:
            if org_timestep[0]  == all_timesteps[0] or org_timestep[0]  == all_timesteps[-1]:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                if not self.config.use_rotary_positional_embeddings:
                    # CogVideoX-2B
                    coefficients = [-3.10658903e+01,  2.54732368e+01, -5.92380459e+00,  1.75769064e+00, -3.61568434e-03]   
                else:
                    # CogVideoX-5B
                    coefficients = [-1.53880483e+03,  8.43202495e+02, -1.34363087e+02,  7.97131516e+00, -5.23162339e-02]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((emb-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = emb            
        
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
                encoder_hidden_states += self.previous_residual_encoder
            else:
                if self.parallel_manager.sp_size > 1:
                    set_pad("pad", hidden_states.shape[1], self.parallel_manager.sp_group)
                    hidden_states = split_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()
                # 4. Transformer blocks
                for i, block in enumerate(self.transformer_blocks):
                    if self.training and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            emb,
                            image_rotary_emb,
                            **ckpt_kwargs,
                        )
                    else:
                        hidden_states, encoder_hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=emb,
                            image_rotary_emb=image_rotary_emb,
                            timestep=timesteps if False else None,
                        )
                self.previous_residual = hidden_states - ori_hidden_states
                self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
        else:
            if self.parallel_manager.sp_size > 1:
                set_pad("pad", hidden_states.shape[1], self.parallel_manager.sp_group)
                hidden_states = split_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))

            # 4. Transformer blocks
            for i, block in enumerate(self.transformer_blocks):
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        timestep=timesteps if False else None,
                    )

        if self.parallel_manager.sp_size > 1:
            if self.enable_teacache:
                    if should_calc:
                        hidden_states = gather_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))
                        self.previous_residual = gather_sequence(self.previous_residual, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))
            else:
                hidden_states = gather_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 5. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 6. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, channels, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if self.parallel_manager.cp_size > 1:
            output = gather_sequence(output, self.parallel_manager.cp_group, dim=0)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def eval_teacache_slow(prompt_list):
    config = CogVideoXConfig()
    engine = VideoSysEngine(config)
    engine.driver_worker.transformer.__class__.enable_teacache = True
    engine.driver_worker.transformer.__class__.rel_l1_thresh = 0.1
    engine.driver_worker.transformer.__class__.accumulated_rel_l1_distance = 0
    engine.driver_worker.transformer.__class__.previous_modulated_input = None
    engine.driver_worker.transformer.__class__.previous_residual = None
    engine.driver_worker.transformer.__class__.previous_residual_encoder = None
    engine.driver_worker.transformer.__class__.forward = teacache_forward
    generate_func(engine, prompt_list, "./samples/cogvideox_teacache_slow", loop=5)

def eval_teacache_fast(prompt_list):
    config = CogVideoXConfig()
    engine = VideoSysEngine(config)
    engine.driver_worker.transformer.__class__.enable_teacache = True
    engine.driver_worker.transformer.__class__.rel_l1_thresh = 0.2
    engine.driver_worker.transformer.__class__.accumulated_rel_l1_distance = 0
    engine.driver_worker.transformer.__class__.previous_modulated_input = None
    engine.driver_worker.transformer.__class__.previous_residual = None
    engine.driver_worker.transformer.__class__.previous_residual_encoder = None
    engine.driver_worker.transformer.__class__.forward = teacache_forward
    generate_func(engine, prompt_list, "./samples/cogvideox_teacache_fast", loop=5)


def eval_base(prompt_list):
    config = CogVideoXConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/cogvideox_base", loop=5)


if __name__ == "__main__":
    prompt_list = read_prompt_list("vbench/VBench_full_info.json")
    eval_base(prompt_list) 
    eval_teacache_slow(prompt_list) 
    eval_teacache_fast(prompt_list) 
    