import torch
from diffusers import LTXPipeline
from diffusers.models.transformers import LTXVideoTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils import export_to_video
from typing import Any, Dict, Optional, Tuple
import numpy as np


def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        if self.enable_teacache:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            inp = self.transformer_blocks[0].norm1(inp)
            num_ada_params = self.transformer_blocks[0].scale_shift_table.shape[0]
            ada_values = self.transformer_blocks[0].scale_shift_table[None, None] + temb_.reshape(batch_size, temb_.size(1), num_ada_params, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
            modulated_inp = inp * (1 + scale_msa) + shift_msa
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [2.14700694e+01, -1.28016453e+01,  2.31279151e+00,  7.92487521e-01, 9.69274326e-03]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp  
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0         
        
        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for block in self.transformer_blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            encoder_attention_mask,
                            **ckpt_kwargs,
                        )
                    else:
                        hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            encoder_attention_mask=encoder_attention_mask,
                        )

                scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
                shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

                hidden_states = self.norm_out(hidden_states)
                hidden_states = hidden_states * (1 + scale) + shift
                self.previous_residual = hidden_states - ori_hidden_states
        else:
            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                    )

            scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift


        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

LTXVideoTransformer3DModel.forward = teacache_forward
prompt = "A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility."
negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
seed = 42
num_inference_steps = 50
pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)

# TeaCache
pipe.transformer.__class__.enable_teacache = True
pipe.transformer.__class__.cnt = 0
pipe.transformer.__class__.num_steps = num_inference_steps
pipe.transformer.__class__.rel_l1_thresh = 0.05 # 0.03 for 1.6x speedup, 0.05 for 2.1x speedup
pipe.transformer.__class__.accumulated_rel_l1_distance = 0
pipe.transformer.__class__.previous_modulated_input = None
pipe.transformer.__class__.previous_residual = None

pipe.to("cuda")
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=512,
    num_frames=161,
    decode_timestep=0.03,
    decode_noise_scale=0.025,
    num_inference_steps=num_inference_steps,
    generator=torch.Generator("cuda").manual_seed(seed)
).frames[0]
export_to_video(video, "teacache_ltx_{}.mp4".format(prompt[:50]), fps=24)