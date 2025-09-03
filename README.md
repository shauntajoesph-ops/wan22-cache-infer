# Wan2.2 Inference (Cache‑Optimized)

This repository provides an inference‑only implementation of Wan2.2 pipelines with an integrated cache system that accelerates generation without changing quality. It targets practical, efficient deployment of:

- i2v‑A14B (Image‑to‑Video)
- ti2v‑5B (Text‑Image‑to‑Video)
- s2v‑14B (Speech‑to‑Video)

Text‑to‑Video (T2V) is no longer supported in this codebase. For T2V, please use the official Diffusers integrations.


## Why This Repo

- Identical generation quality: Caches are indicator‑only and never alter final math when a compute path is required. If a gate is unsafe, compute is forced.
- Lower latency and VRAM: Conditional skipping reduces transformer work; optional CPU offload keeps peak memory down on single‑GPU.
- Unified Cache Manager: One mechanism governs TeaCache, FBCache, and CFG cache with a single set of flags and consistent telemetry.
- Inference‑only surfaces: Training is disabled by default to keep the code small, safe, and focused.


## Installation

- OS: Ubuntu 20.04/22.04 (x86_64) or compatible Linux.
- Python: 3.10.x.
- NVIDIA Driver: 535+ (CUDA 12.x runtime); verify with `nvidia-smi`.
- GPU: Ampere (SM80, e.g., A100/3090) or newer recommended for Flash‑Attention. Hopper (H100) supported. Older GPUs can run without flash_attn.
- System tools: `ffmpeg` for video muxing (`sudo apt-get install -y ffmpeg`).

Recommended pinned dependencies (constraints for reproducibility):

```txt
torch==2.4.0
torchvision==0.19.0
diffusers==0.31.0
transformers==4.49.0
tokenizers==0.20.3
accelerate==1.1.1
opencv-python==4.9.0.80
imageio==2.34.1
imageio-ffmpeg==0.4.9
easydict==1.10
ftfy==6.2.0
numpy==1.26.4
# Optional but recommended on Ampere/Hopper
flash-attn>=2.5.0
```

Install (use a clean virtualenv/conda):

```bash
pip install -r requirements.txt  # or: pip install .
# dev tools (optional)
pip install .[dev]
```

GPU VRAM guidance (single GPU):
- I2V‑A14B: ≥ 80 GB recommended. Use multi‑GPU (FSDP+SP) or `--offload_model True` on smaller cards.
- TI2V‑5B: ≥ 24 GB for text‑only; add `--offload_model True --t5_cpu` on 24 GB.
- S2V‑14B: ≥ 80 GB recommended; `--infer_frames` reduces memory.

Models: download weights from Hugging Face or ModelScope and point `--ckpt_dir` at the local folder, e.g.:

- Wan2.2‑I2V‑A14B  →  `./Wan2.2-I2V-A14B`
- Wan2.2‑TI2V‑5B   →  `./Wan2.2-TI2V-5B`
- Wan2.2‑S2V‑14B   →  `./Wan2.2-S2V-14B`


## Model Download

Sources:
- Hugging Face: `huggingface-cli download Wan-AI/<MODEL_NAME> --local-dir <TARGET_DIR>`
- ModelScope: `modelscope download Wan-AI/<MODEL_NAME> --local_dir <TARGET_DIR>`

Integrity (optional but recommended):
```bash
find <TARGET_DIR> -type f -maxdepth 1 -print0 | xargs -0 sha256sum > checksums.txt
# Compare against published checksums or keep for provenance
```

Expected folder layouts:

- I2V‑A14B (`./Wan2.2-I2V-A14B`)
```
Wan2.2-I2V-A14B/
├─ models_t5_umt5-xxl-enc-bf16.pth
├─ Wan2.1_VAE.pth
├─ low_noise_model/      # diffusers-style weights (config + *.safetensors)
└─ high_noise_model/     # diffusers-style weights
```

- TI2V‑5B (`./Wan2.2-TI2V-5B`)
```
Wan2.2-TI2V-5B/
├─ models_t5_umt5-xxl-enc-bf16.pth
├─ Wan2.2_VAE.pth
└─ model files at root   # diffusers-style weights (model_index.json, *.safetensors)
```

- S2V‑14B (`./Wan2.2-S2V-14B`)
```
Wan2.2-S2V-14B/
├─ models_t5_umt5-xxl-enc-bf16.pth
├─ Wan2.1_VAE.pth
├─ wav2vec2-large-xlsr-53-english/   # HF wav2vec2 directory
└─ model files at root               # diffusers-style weights
```


## Quickstart

Below are minimal commands. For more options (FSDP, Ulysses, offload, cache flags), see the sections after this.

- Single‑GPU I2V
```bash
python generate.py \
  --task i2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image examples/i2v_input.JPG \
  --prompt "A cinematic close-up of a playful cat at the beach."
```

- Multi‑GPU I2V (FSDP + Ulysses)
```bash
torchrun --nproc_per_node=8 generate.py \
  --task i2v-A14B \
  --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image examples/i2v_input.JPG \
  --dit_fsdp --t5_fsdp --ulysses_size 8
```

- Single‑GPU TI2V (text‑only)
```bash
python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --prompt "Two anthropomorphic cats boxing on a spotlighted stage."
```

- Single‑GPU TI2V (with image)
```bash
python generate.py \
  --task ti2v-5B \
  --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --image examples/i2v_input.JPG \
  --prompt "A serene seaside scene; maintain subject identity."
```

- Single‑GPU S2V
```bash
python generate.py \
  --task s2v-14B \
  --size 1024*704 \
  --ckpt_dir ./Wan2.2-S2V-14B \
  --image examples/i2v_input.JPG \
  --audio examples/talk.wav \
  --prompt "A person calmly narrating a scene."
```


## Cache Manager (TeaCache · FBCache · CFG)

This repo includes a unified cache manager that coordinates TeaCache, FBCache, and CFG cache. It provides a shared lifecycle, a single set of flags, and clear telemetry. Caches are indicator‑only gates; when a gate is unsafe, the code falls back to compute.

- Priority: FBCache → TeaCache → Compute (first eligible skip wins)
- Lifecycle: warmup → main → last‑steps, applied to the cond branch; uncond shares the same timestep index
- Distributed: scalar metrics are all‑reduced (mean) across ranks in Ulysses/SP mode
- Offload: cached residuals move with the model when offloading to/from CPU
- Telemetry: printed on rank 0 at the end of a run, per expert (I2V/TI2V) or per model (S2V)

Expected effects (typical):
- 1.3×–2.0× speedups depending on threshold and content
- Identical generation quality; skips reuse cached residuals conservatively and force compute on shape/dtype mismatches or invalid metrics

### TeaCache flags
Enable conditional transformer skipping based on modulated hidden dynamics.

- `--teacache`: enable (default off)
- `--teacache_thresh`: skip aggressiveness (default 0.08; lower is more conservative)
- `--teacache_policy`: rescale policy (`linear` default; unknown values fallback to `linear`)
- `--teacache_warmup`: force compute first K executed cond steps (default 1)
- `--teacache_last_steps`: force compute last K executed cond steps (default 1)
- `--teacache_alternating`: alternate skip eligibility to stabilize long reuse sequences (default off)

Example:
```bash
python generate.py --task i2v-A14B --size 832*480 --ckpt_dir ./Wan2.2-I2V-A14B \
  --image examples/i2v_input.JPG \
  --teacache --teacache_thresh 0.08
```

### FBCache flags
Enable first‑block cache using early hidden/residual metrics (indicator‑only).

- `--fbcache`: enable (default off)
- `--fb_thresh`: rescaled metric threshold for skip (default 0.08)
- `--fb_metric`: metric type (`hidden_rel_l1` default; `residual_rel_l1`, `hidden_rel_l2` optional)
- `--fb_downsample`: stride for metric computation over tokens (default 1)
- `--fb_ema`: EMA factor [0,1) for smoothing the metric (default 0 → off)
- `--fb_warmup`: force compute first K executed cond steps (default 1)
- `--fb_last_steps`: force compute last K executed cond steps (default 1)
- `--fb_cfg_sep_diff`: compute CFG and non‑CFG diffs separately (default true)

Example:
```bash
python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B \
  --fbcache --fb_thresh 0.08
```

Notes:
- Prefer enabling one of TeaCache or FBCache at a time. If both are set, FBCache takes precedence.
- On shape/dtype mismatch or invalid metrics, compute is forced and accumulators reset (failsafe).

### CFG cache
- Cond‑authoritative by default: reuse cond branch metrics/action for uncond within a step (aligns with separate‑CFG pipelines in Wan2.2).
- Controlled inside the Cache Manager; no extra CLI flag required. When `--fb_cfg_sep_diff=true`, FBCache can evaluate CFG and non‑CFG diffs separately but still respects cond action unless a hard failsafe triggers.

### Telemetry
- End‑of‑run summary lines like:
  - `CacheManager[low] summary: ...`
  - `CacheManager[high] summary: ...`
  - or a single `CacheManager[...] summary` for S2V
- Includes per‑branch totals/skips/averages and failsafe counts.


## Performance Tips
- Single GPU: `--offload_model True` can reduce peak VRAM (slower but fits bigger sizes).
- Multi‑GPU: prefer `--dit_fsdp --t5_fsdp --ulysses_size <num_ranks>`; offload is typically unnecessary.
- Data types: `--convert_model_dtype` casts DiT params to bf16/fp16 when not using FSDP.
- Set `WAN_INFERENCE_ONLY=1` (default) to keep autograd disabled and avoid accidental training calls.


## Notes on Scope
- This is an inference‑only library. Training APIs are disabled by default; set `WAN_INFERENCE_ONLY=0` to bypass at your own risk.
- T2V (Text‑to‑Video) was removed from this codebase. Use the official Diffusers pipelines for T2V.


## MoE Architecture (I2V‑A14B)

I2V‑A14B uses a Mixture‑of‑Experts backbone with two experts: a high‑noise expert for early (noisy) steps and a low‑noise expert for late (clean) steps. Switching is deterministic and based on the global denoising step index.

- Experts: `high_noise_model` and `low_noise_model` are loaded and swapped per step.
- Boundary: `boundary_step = boundary * num_train_timesteps` (from config). For step `t`:
  - if `t >= boundary_step` → use high‑noise expert
  - else → use low‑noise expert
- Offload: when `--offload_model True`, the inactive expert is moved to CPU to reduce VRAM; cached residuals are moved alongside to keep device state consistent.

This scheme preserves quality while allowing memory‑efficient runs in single‑GPU and multi‑GPU settings.


## Benchmarks (Reproducible)

We include bash scripts to measure end‑to‑end wallclock time and verify cache telemetry.

- TeaCache smoke tests: `bash tests/test_teacache.sh <models_root> [gpus]`
- Quick smoke (I2V/TI2V/S2V): `bash tests/test_cache_smoke.sh <models_root>`
- TeaCache benchmark: `bash benchmarks/bench_teacache.sh <models_root>`

Methodology:
- Fixed seeds (`--base_seed 42`) and fixed prompts/images to ensure repeatability.
- Measure wallclock with `/usr/bin/time` and extract VRAM via `nvidia-smi --query-gpu=memory.used` during peak sections.
- Run two variants per task: baseline (no cache) vs cache enabled (`--teacache --teacache_thresh 0.08` or `--fbcache --fb_thresh 0.08`).

What to expect (typical):

| Task     | Resolution    | Cache     | Speedup vs. baseline | Peak VRAM impact |
|----------|---------------|-----------|-----------------------|------------------|
| I2V‑A14B | 832×480       | TeaCache  | 1.3×–2.0×            | Neutral/slightly lower |
| TI2V‑5B  | 1280×704      | TeaCache  | 1.3×–2.0×            | Neutral/slightly lower |
| S2V‑14B  | 1024×704      | TeaCache  | 1.3×–2.0×            | Neutral/slightly lower |

Notes:
- Actual numbers depend on content, hardware, and thresholds.
- FBCache provides similar ranges and may work better on some prompts; enable either TeaCache or FBCache (FBCache wins if both set).

## License
Apache 2.0. See `LICENSE.txt`.


## Acknowledgements
Thanks to contributors and upstream projects including Diffusers, Qwen, and the broader Wan community.
