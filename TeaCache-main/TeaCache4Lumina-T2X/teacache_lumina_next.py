import torch
from typing import Any, Dict, Optional, Tuple, Union
from diffusers import LuminaText2ImgPipeline
from diffusers.models import LuminaNextDiT2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
import numpy as np
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict=True,
    ) -> torch.Tensor:
        """
        Forward pass of LuminaNextDiT.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of shape (N, C, H, W).
            timestep (torch.Tensor): Tensor of diffusion timesteps of shape (N,).
            encoder_hidden_states (torch.Tensor): Tensor of caption features of shape (N, D).
            encoder_mask (torch.Tensor): Tensor of caption masks of shape (N, L).
        """
        hidden_states, mask, img_size, image_rotary_emb = self.patch_embedder(hidden_states, image_rotary_emb)
        image_rotary_emb = image_rotary_emb.to(hidden_states.device)

        temb = self.time_caption_embed(timestep, encoder_hidden_states, encoder_mask)

        encoder_mask = encoder_mask.bool()
        if self.enable_teacache:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            modulated_inp, gate_msa, scale_mlp, gate_mlp = self.layers[0].norm1(inp, temb_)
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else: 
                coefficients = [393.76566581, -603.50993606,  209.10239044,  -23.00726601,    0.86377344]
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
                for layer in self.layers:
                    hidden_states = layer(
                        hidden_states,
                        mask,
                        image_rotary_emb,
                        encoder_hidden_states,
                        encoder_mask,
                        temb=temb,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                self.previous_residual = hidden_states - ori_hidden_states

        else:
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    mask,
                    image_rotary_emb,
                    encoder_hidden_states,
                    encoder_mask,
                    temb=temb,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        hidden_states = self.norm_out(hidden_states, temb)

        # unpatchify
        height_tokens = width_tokens = self.patch_size
        height, width = img_size[0]
        batch_size = hidden_states.size(0)
        sequence_length = (height // height_tokens) * (width // width_tokens)
        hidden_states = hidden_states[:, :sequence_length].view(
            batch_size, height // height_tokens, width // width_tokens, height_tokens, width_tokens, self.out_channels
        )
        output = hidden_states.permute(0, 5, 1, 3, 2, 4).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


LuminaNextDiT2DModel.forward = teacache_forward
num_inference_steps = 30
seed = 1024
prompt = "Upper body of a young woman in a Victorian-era outfit with brass goggles and leather straps. "
pipeline = LuminaText2ImgPipeline.from_pretrained("Alpha-VLLM/Lumina-Next-SFT-diffusers", torch_dtype=torch.bfloat16).to("cuda")

# TeaCache
pipeline.transformer.__class__.enable_teacache = True
pipeline.transformer.__class__.cnt = 0
pipeline.transformer.__class__.num_steps = num_inference_steps
pipeline.transformer.__class__.rel_l1_thresh = 0.3 # 0.2 for 1.5x speedup, 0.3 for 1.9x speedup, 0.4 for 2.4x speedup, 0.5 for 2.8x speedup
pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
pipeline.transformer.__class__.previous_modulated_input = None
pipeline.transformer.__class__.previous_residual = None

image = pipeline(
    prompt=prompt, 
    num_inference_steps=num_inference_steps, 
    generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]
image.save("teacache_lumina_{}.png".format(prompt))