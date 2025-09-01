# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from einops import rearrange
import torch
import numpy as np
from typing import Optional, Tuple, List, Optional
from cosmos1.models.diffusion.inference.inference_utils import add_common_arguments, check_input_frames, validate_args
from cosmos1.models.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldGenerationPipeline
from cosmos1.utils import log, misc
from cosmos1.utils.io import read_prompts_from_file, save_video
from cosmos1.models.diffusion.conditioner import DataType
from cosmos1.models.diffusion.module.blocks import adaln_norm_state

torch.enable_grad(False)

def teacache_forward(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    crossattn_mask: Optional[torch.Tensor] = None,
    fps: Optional[torch.Tensor] = None,
    image_size: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    scalar_feature: Optional[torch.Tensor] = None,
    data_type: Optional[DataType] = DataType.VIDEO,
    latent_condition: Optional[torch.Tensor] = None,
    latent_condition_sigma: Optional[torch.Tensor] = None,
    condition_video_augment_sigma: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Args:
        x: (B, C, T, H, W) tensor of spatial-temp inputs
        timesteps: (B, ) tensor of timesteps
        crossattn_emb: (B, N, D) tensor of cross-attention embeddings
        crossattn_mask: (B, N) tensor of cross-attention masks
        condition_video_augment_sigma: (B,) used in lvg(long video generation), we add noise with this sigma to
            augment condition input, the lvg model will condition on the condition_video_augment_sigma value;
            we need forward_before_blocks pass to the forward_before_blocks function.
    """

    inputs = self.forward_before_blocks(
        x=x,
        timesteps=timesteps,
        crossattn_emb=crossattn_emb,
        crossattn_mask=crossattn_mask,
        fps=fps,
        image_size=image_size,
        padding_mask=padding_mask,
        scalar_feature=scalar_feature,
        data_type=data_type,
        latent_condition=latent_condition,
        latent_condition_sigma=latent_condition_sigma,
        condition_video_augment_sigma=condition_video_augment_sigma,
        **kwargs,
    )
    x, affline_emb_B_D, crossattn_emb, crossattn_mask, rope_emb_L_1_1_D, adaln_lora_B_3D, original_shape = (
        inputs["x"],
        inputs["affline_emb_B_D"],
        inputs["crossattn_emb"],
        inputs["crossattn_mask"],
        inputs["rope_emb_L_1_1_D"],
        inputs["adaln_lora_B_3D"],
        inputs["original_shape"],
    )
    extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = inputs["extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D"]
    if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
        assert (
            x.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
        ), f"{x.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape} {original_shape}"
    
    if self.enable_teacache:
        inp = x.clone()
        if self.blocks["block0"].blocks[0].use_adaln_lora:
            shift_B_D, scale_B_D, gate_B_D = (self.blocks["block0"].blocks[0].adaLN_modulation(affline_emb_B_D) + adaln_lora_B_3D).chunk(
                self.blocks["block0"].blocks[0].n_adaln_chunks, dim=1
            )
        else:
            shift_B_D, scale_B_D, gate_B_D = self.blocks["block0"].blocks[0].adaLN_modulation(affline_emb_B_D).chunk(self.blocks["block0"].blocks[0].n_adaln_chunks, dim=1)

        shift_1_1_1_B_D, scale_1_1_1_B_D, _ = (
            shift_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            scale_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            gate_B_D.unsqueeze(0).unsqueeze(0).unsqueeze(0),
        )

        modulated_inp = adaln_norm_state(self.blocks["block0"].blocks[0].norm_state, inp, scale_1_1_1_B_D, shift_1_1_1_B_D)

        if self.cnt%2 == 0:
            self.is_even = True # even->condition odd->uncondition
            if self.cnt == 0 or self.cnt == self.num_steps:
                should_calc_even = True
                self.accumulated_rel_l1_distance_even = 0  
            else: 
                coefficients = [2.71156237e+02, -9.19775607e+01, 2.24437250e+00, 2.08355751e+00, 1.41776330e-01]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_even += rescale_func(((modulated_inp-self.previous_modulated_input_even).abs().mean() / self.previous_modulated_input_even.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_even < self.rel_l1_thresh:
                    should_calc_even = False
                else:
                    should_calc_even = True
                    self.accumulated_rel_l1_distance_even = 0
            self.previous_modulated_input_even = modulated_inp.clone()
            self.cnt += 1
            if self.cnt == self.num_steps+2:
                self.cnt = 0

        else:
            self.is_even = False
            if self.cnt == 1 or self.cnt == self.num_steps+1:
                should_calc_odd = True
                self.accumulated_rel_l1_distance_odd = 0  
            else: 
                coefficients = [2.71156237e+02, -9.19775607e+01, 2.24437250e+00, 2.08355751e+00, 1.41776330e-01]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_odd += rescale_func(((modulated_inp-self.previous_modulated_input_odd).abs().mean() / self.previous_modulated_input_odd.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance_odd < self.rel_l1_thresh:
                    should_calc_odd = False
                else:
                    should_calc_odd = True
                    self.accumulated_rel_l1_distance_odd = 0
            self.previous_modulated_input_odd = modulated_inp.clone()
            self.cnt += 1
            if self.cnt == self.num_steps+2:
                self.cnt = 0

                
    if self.enable_teacache:

        if self.is_even:
            if not should_calc_even:
                x += self.previous_residual_even
            else:
                ori_x = x.clone() # (t, h, w, b, d) = (16, 44, 80, 1, 4096)
                for _, block in self.blocks.items():
                    assert (
                        self.blocks["block0"].x_format == block.x_format
                    ), f"First block has x_format {self.blocks[0].x_format}, got {block.x_format}"

                    x = block(
                        x,
                        affline_emb_B_D,
                        crossattn_emb,
                        crossattn_mask,
                        rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                        adaln_lora_B_3D=adaln_lora_B_3D,
                        extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                    ) # x(t, h, w, b, d) = (16, 44, 80, 1, 4096)
                self.previous_residual_even = x - ori_x
            x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D") # (b, t, h, w, d) = (1, 16, 40, 80, 4096)
            x_B_D_T_H_W = self.decoder_head(
                x_B_T_H_W_D=x_B_T_H_W_D,
                emb_B_D=affline_emb_B_D,
                crossattn_emb=None,
                origin_shape=original_shape,
                crossattn_mask=None,
                adaln_lora_B_3D=adaln_lora_B_3D,
            ) # (b, d, t, h, w) = (1, 16, 16, 88, 160)
            return x_B_D_T_H_W
            
        else: # odd
            if not should_calc_odd:
                x += self.previous_residual_odd
            else:
                ori_x = x.clone()
                for _, block in self.blocks.items():
                    assert (
                        self.blocks["block0"].x_format == block.x_format
                    ), f"First block has x_format {self.blocks[0].x_format}, got {block.x_format}"

                    x = block(
                        x,
                        affline_emb_B_D,
                        crossattn_emb,
                        crossattn_mask,
                        rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                        adaln_lora_B_3D=adaln_lora_B_3D,
                        extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
                    )
                self.previous_residual_odd = x - ori_x
            x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")
            x_B_D_T_H_W = self.decoder_head(
                x_B_T_H_W_D=x_B_T_H_W_D,
                emb_B_D=affline_emb_B_D,
                crossattn_emb=None,
                origin_shape=original_shape,
                crossattn_mask=None,
                adaln_lora_B_3D=adaln_lora_B_3D,
            )
            return x_B_D_T_H_W
  
    else:
        for _, block in self.blocks.items():
            assert (
                self.blocks["block0"].x_format == block.x_format
            ), f"First block has x_format {self.blocks[0].x_format}, got {block.x_format}"

            x = block(
                x,
                affline_emb_B_D,
                crossattn_emb,
                crossattn_mask,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_3D=adaln_lora_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
        x_B_T_H_W_D = rearrange(x, "T H W B D -> B T H W D")

        x_B_D_T_H_W = self.decoder_head(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_D=affline_emb_B_D,
            crossattn_emb=None,
            origin_shape=original_shape,
            crossattn_mask=None,
            adaln_lora_B_3D=adaln_lora_B_3D,
        )

        return x_B_D_T_H_W

def teacache_v2v_forward(
    self,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    crossattn_mask: Optional[torch.Tensor] = None,
    fps: Optional[torch.Tensor] = None,
    image_size: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    scalar_feature: Optional[torch.Tensor] = None,
    data_type: Optional[DataType] = DataType.VIDEO,
    video_cond_bool: Optional[torch.Tensor] = None,
    condition_video_indicator: Optional[torch.Tensor] = None,
    condition_video_input_mask: Optional[torch.Tensor] = None,
    condition_video_augment_sigma: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Forward pass of the video-conditioned DIT model.

    Args:
        x: Input tensor of shape (B, C, T, H, W)
        timesteps: Timestep tensor of shape (B,)
        crossattn_emb: Cross attention embeddings of shape (B, N, D)
        crossattn_mask: Optional cross attention mask of shape (B, N)
        fps: Optional frames per second tensor
        image_size: Optional image size tensor
        padding_mask: Optional padding mask tensor
        scalar_feature: Optional scalar features tensor
        data_type: Type of data being processed (default: DataType.VIDEO)
        video_cond_bool: Optional video conditioning boolean tensor
        condition_video_indicator: Optional video condition indicator tensor
        condition_video_input_mask: Required mask tensor for video data type
        condition_video_augment_sigma: Optional sigma values for conditional input augmentation
        **kwargs: Additional keyword arguments

    Returns:
        torch.Tensor: Output tensor
    """
    B, C, T, H, W = x.shape

    if data_type == DataType.VIDEO:
        assert condition_video_input_mask is not None, "condition_video_input_mask is required for video data type"

        input_list = [x, condition_video_input_mask]
        x = torch.cat(
            input_list,
            dim=1,
        )

    return teacache_forward(
        self=self,
        x=x,
        timesteps=timesteps,
        crossattn_emb=crossattn_emb,
        crossattn_mask=crossattn_mask,
        fps=fps,
        image_size=image_size,
        padding_mask=padding_mask,
        scalar_feature=scalar_feature,
        data_type=data_type,
        condition_video_augment_sigma=condition_video_augment_sigma,
        **kwargs,
    )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add video2world specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Cosmos-1.0-Diffusion-7B-Video2World",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Cosmos-1.0-Diffusion-7B-Video2World",
            "Cosmos-1.0-Diffusion-14B-Video2World",
        ],
    )
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--input_image_or_video_path",
        type=str,
        help="Input video/image path for generating a single video",
    )
    parser.add_argument(
        "--num_input_frames",
        type=int,
        default=1,
        help="Number of input frames for video2world prediction",
        choices=[1, 9],
    )

    parser.add_argument(
        "--rel_l1_thresh",
        type=float,
        default=0.4,
        help="Higher speedup will cause to worse quality -- 0.1 for 1.3x speedup -- 0.2 for 1.8x speedup -- 0.3 for 2.1x speedup",
    )

    return parser.parse_args()


def demo(cfg):
    """Run video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    misc.set_random_seed(cfg.seed)
    inference_type = "video2world"
    validate_args(cfg, inference_type)

    # Initialize video2world generation model pipeline
    pipeline = DiffusionVideo2WorldGenerationPipeline(
        inference_type=inference_type,
        checkpoint_dir=cfg.checkpoint_dir,
        checkpoint_name=cfg.diffusion_transformer_dir,
        prompt_upsampler_dir=cfg.prompt_upsampler_dir,
        enable_prompt_upsampler=not cfg.disable_prompt_upsampler,
        offload_network=cfg.offload_diffusion_transformer,
        offload_tokenizer=cfg.offload_tokenizer,
        offload_text_encoder_model=cfg.offload_text_encoder_model,
        offload_prompt_upsampler=cfg.offload_prompt_upsampler,
        offload_guardrail_models=cfg.offload_guardrail_models,
        guidance=cfg.guidance,
        num_steps=cfg.num_steps,
        height=cfg.height,
        width=cfg.width,
        fps=cfg.fps,
        num_video_frames=cfg.num_video_frames,
        seed=cfg.seed,
        num_input_frames=cfg.num_input_frames,
        enable_text_guardrail=False,
        enable_video_guardrail=False,
    )

    # TeaCache
    pipeline.model.net.__class__.forward = teacache_v2v_forward
    pipeline.model.net.__class__.enable_teacache = True
    pipeline.model.net.__class__.cnt = 0
    pipeline.model.net.__class__.num_steps = cfg.num_steps * 2
    pipeline.model.net.__class__.rel_l1_thresh = cfg.rel_l1_thresh
    pipeline.model.net.__class__.accumulated_rel_l1_distance_even = 0
    pipeline.model.net.__class__.accumulated_rel_l1_distance_odd = 0
    pipeline.model.net.__class__.previous_modulated_input_even = None
    pipeline.model.net.__class__.previous_modulated_input_odd = None
    pipeline.model.net.__class__.previous_residual_even = None
    pipeline.model.net.__class__.previous_residual_odd = None

    # Handle multiple prompts if prompt file is provided
    if cfg.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(cfg.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": cfg.prompt, "visual_input": cfg.input_image_or_video_path}]

    os.makedirs(cfg.video_save_folder, exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        if current_prompt is None and cfg.disable_prompt_upsampler:
            log.critical("Prompt is missing, skipping world generation.")
            continue
        current_image_or_video_path = input_dict.get("visual_input", None)
        if current_image_or_video_path is None:
            log.critical("Visual input is missing, skipping world generation.")
            continue

        # Check input frames
        if not check_input_frames(current_image_or_video_path, cfg.num_input_frames):
            continue

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_or_video_path=current_image_or_video_path,
            negative_prompt=cfg.negative_prompt,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        video, prompt = generated_output

        if cfg.batch_input_path:
            video_save_path = os.path.join(cfg.video_save_folder, f"{i}.mp4")
            prompt_save_path = os.path.join(cfg.video_save_folder, f"{i}.txt")
        else:
            video_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.mp4")
            prompt_save_path = os.path.join(cfg.video_save_folder, f"{cfg.video_save_name}.txt")

        # Save video
        save_video(
            video=video,
            fps=cfg.fps,
            H=cfg.height,
            W=cfg.width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )

        # Save prompt to text file alongside video
        with open(prompt_save_path, "wb") as f:
            f.write(prompt.encode("utf-8"))

        log.info(f"Saved video to {video_save_path}")
        log.info(f"Saved prompt to {prompt_save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    demo(args)
