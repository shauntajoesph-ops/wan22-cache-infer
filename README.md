# Wan2.2 Inference · Cache Acceleration

This repo is based on Wan2.2 (https://github.com/Wan-Video/Wan2.2) and adds inference caches for acceleration (TeaCache, FBCache, CFG). Supported tasks: Image‑to‑Video, Text‑Image‑to‑Video, Speech‑to‑Video(No Text-to-Video).


Same Wan2.2 pipelines with cache‑assisted acceleration. At conservative defaults, quality parity is typically maintained, but you should validate with your workloads. A unified Cache Manager governs TeaCache, FBCache, and CFG reuse, working on single GPU (optional offload) and scaling to multi‑GPU (FSDP + Ulysses SP).


Note: pure Text‑to‑Video (T2V) is not included in this fork.


WARNING
- Cache features can slightly degrade generation quality at aggressive thresholds; validate quality on your data before production.
- Caches add residual buffers that may increase VRAM/RAM usage (especially with separate CFG branches and per‑expert models).



## Cache Manager (TeaCache · FBCache · CFG)

Design (first‑mode‑wins: FBCache → TeaCache → Compute):

```
Inputs (Prompt / Image / Audio)
          │
    Encoders (T5 / Vision / Audio)
          │
          v
        DiT Step (t)
          │
   [Block 0]
      │   └─> mod_inp / x_after_b0 ──┐  (signals)
      v                               │
   [Blocks 1..N]                      │
      │                               │
      v                               │
   hidden/residuals  <──────────────┐ │  (cached full‑stack residuals)
                                     │ │
   +----------- Cache Manager -------------------------------+
   |  FBCache (within‑step, first‑block metric)              |
   |   • compute rel from mod_inp or residual (b0)          |
   |   • SP all‑reduce(mean) if SP>1                        |
   |   • if rescaled_acc < fb_thresh → SKIP rest of stack   |
   |                                                        |
   |  TeaCache (across‑steps)                               |
   |   • rel vs prev‑step signature                         |
   |   • SP all‑reduce(mean) if SP>1                        |
   |   • if rescaled_acc < tc_thresh → SKIP at step t       |
   |                                                        |
   |  CFG reuse (within‑step)                               |
   |   • cond decision/metric authoritative for uncond      |
   |                                                        |
   |  Offload moves · Fail‑safes · Telemetry                |
   +--------------------------+-----------------------------+
                              │
                    decision: SKIP/COMPUTE
                              │
    if SKIP: apply cached residuals (resume_from_block=1 for FB)
                              │
                              v
                        VAE Decode → Video
```


- Distributed (SP/Ulysses): fp32 all‑reduce (mean) of scalar metrics across ranks; if reduction fails, use local metric and record a failsafe.
- Offload: cached residuals move with the model on CPU↔GPU transitions; shape/dtype mismatches force compute and increment failsafe count.
- Telemetry: per‑branch totals/skips/avg rel/avg rescaled, global failsafe count, and pair stats; printed once on rank 0 at end of run.
- Fail‑safes (compute is forced): invalid metrics (NaN/Inf), reduce errors, shape/dtype mismatch, missing/OOM residual moves, lifecycle guards, or pair‑consistency issues in separate‑CFG mode.


## Cache Principles & Implementations

### TeaCache
- Principle: Across‑step gating based on stability of hidden dynamics. If the step‑to‑step signature change is small, reuse the previous step’s residuals.
- What is cached: Full‑stack residuals for the DiT step (cond branch; uncond follows cond action).
- Decision signal: Signature of normalized, time‑modulated input to the first self‑attention block; relative change vs previous step; rescaled via `--teacache_policy` (default: linear/identity).
- Scope: Across steps (within the DiT forward for the cond branch); uncond reuses cond action within the same step.
- Expected speedup: ~1.4–2.2× (content/threshold dependent).
- Memory/quality trade‑offs: Adds per‑branch residual buffers; aggressive thresholds can cause detail staleness or motion jitter. Use `--teacache_warmup/--teacache_last_steps` and optionally `--teacache_alternating` to stabilize long reuse runs.
- Compatibility: SP/Ulysses uses fp32 mean for scalar metrics; FSDP supported (attach after wrapping); Offload supported (residuals move with model); MoE/expert models supported with per‑expert telemetry.
- Enable flags & defaults: `--teacache` (off), `--teacache_thresh 0.08`, `--teacache_policy linear`, `--teacache_warmup 1`, `--teacache_last_steps 1`, `--teacache_alternating` (off).

- What: Conditional transformer skipping; when hidden/residual dynamics are stable, reuse cached residuals instead of recomputing all layers.
- Where: Across sampling steps within the DiT block(s) for the cond branch; uncond follows cond decisions.
- Speedup: ~1.4–2.2× in typical tests (threshold‑dependent).


### FBCache
- Principle: Indicator‑only gate using an early (block‑0) metric to decide whether to skip the rest of the stack for the current step and reuse cached residuals.
- What is cached: Full‑stack residuals for the current step (resume from block‑0 when skipping).
- Decision signal: `--fb_metric` on block‑0 inputs/outputs (`hidden_rel_l1`, `hidden_rel_l2`, or `residual_rel_l1`); optional `--fb_downsample` stride and `--fb_ema` smoothing; rescaled and accumulated vs threshold.
- Scope: Within‑step (first‑block gate) for each timestep.
- Expected speedup: ~1.2–1.6× (metric/threshold dependent).
- Memory/quality trade‑offs: Adds residual buffers; at aggressive thresholds, can under‑react to rapid changes and cause subtle artifacts. EMA and warmup/last‑steps mitigate thrashing; defaults are conservative.
- Compatibility: SP/Ulysses scalar mean; FSDP supported; Offload supported; MoE/expert models supported per expert. `--fb_cfg_sep_diff` allows uncond metric to be computed separately while cond action remains authoritative unless a hard failsafe requires compute.
- Enable flags & defaults: `--fbcache` (off), `--fb_thresh 0.08`, `--fb_metric hidden_rel_l1|hidden_rel_l2|residual_rel_l1` (default hidden_rel_l1), `--fb_downsample 1`, `--fb_ema 0.0`, `--fb_warmup 1`, `--fb_last_steps 1`, `--fb_cfg_sep_diff true`.
- What: First‑block indicator gating; compute a lightweight metric on the first block, then reuse cached residuals for the rest of the step if changes are small.

- Where: First transformer block each step (indicator‑only); affects the whole step.
- Speedup: ~1.2–1.6× in typical tests (metric/threshold dependent).
 - Extra: `--fb_cfg_sep_diff` lets uncond compute its own metric; cond action remains authoritative unless a hard failsafe requires compute.


### CFG Cache
- Principle: Cond‑authoritative pair policy. Within a timestep, reuse the cond branch’s metric and decision for uncond to keep CFG pairs consistent and avoid duplicate gating work.
- What is cached: Last cond rescaled metric(s) and action for the current step; uncond may compute its own metric when `--fb_cfg_sep_diff=true` but still follows cond action unless a hard failsafe applies.
- Decision signal: Cond branch rescaled metric and action; optional separate metric for uncond if `--fb_cfg_sep_diff=true`.
- Scope: Within‑step (cond → uncond) only.
- Expected speedup: ~1.1–1.7× depending on CFG strength and content.
- Memory/quality trade‑offs: Minimal; prioritizes pair consistency. Uncond can be forced to compute if residuals are missing or guards trigger.
- Compatibility: Works with SP/FSDP/offload; decisions remain consistent across ranks via cond‑authoritative action.
- Enable flags & defaults: Enabled by design (no separate CLI flag). Pair behavior can be adjusted via `--fb_cfg_sep_diff` (default true).

- What: Reuse cond branch decision/metrics for the uncond branch within a timestep (cond‑authoritative) to keep pairs consistent.
- Where: Within‑step (cond → uncond) only; separate‑CFG pipelines use cond as the authority.
- Speedup: ~1.1–1.7× depending on CFG strength and content.


## Quick Start

Install:
```bash
pip install .            # add: pip install .[dev]  (optional)
```

Single‑GPU examples:
```bash
# I2V (TeaCache)
python generate.py --task i2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image examples/i2v_input.JPG \
  --teacache --teacache_thresh 0.08

# TI2V (FBCache)
python generate.py --task ti2v-5B --size 1280*704 \
  --ckpt_dir ./Wan2.2-TI2V-5B \
  --prompt "Two anthropomorphic cats boxing on a spotlighted stage." \
  --fbcache --fb_thresh 0.10

# S2V (TeaCache)
python generate.py --task s2v-14B --size 1024*704 \
  --ckpt_dir ./Wan2.2-S2V-14B \
  --image examples/i2v_input.JPG \
  --audio examples/talk.wav \
  --teacache --teacache_thresh 0.08
```

Multi‑GPU (FSDP + Ulysses/SP):
```bash
torchrun --nproc_per_node=8 generate.py \
  --task i2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image examples/i2v_input.JPG \
  --dit_fsdp --t5_fsdp --ulysses_size 8 \
  --teacache --teacache_thresh 0.08
```

Examples & Flags:
- TeaCache: `--teacache --teacache_thresh 0.08 [--teacache_policy linear --teacache_warmup 1 --teacache_last_steps 1 --teacache_alternating]`
- FBCache: `--fbcache --fb_thresh 0.10 [--fb_metric hidden_rel_l1 --fb_downsample 1 --fb_ema 0 --fb_warmup 1 --fb_last_steps 1 --fb_cfg_sep_diff true]`
- CFG reuse: on by default in separate‑CFG pipelines; no flag required.
- Combining: you can pass both, but FBCache takes precedence by design.

Combined example (FBCache + TeaCache; FBCache wins):
```bash
python generate.py --task i2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image examples/i2v_input.JPG \
  --fbcache --fb_thresh 0.10 \
  --teacache --teacache_thresh 0.08
```

Notes:
- Single GPU: you may add `--offload_model True` to reduce peak VRAM.
- Multi‑GPU: offload is auto‑disabled; keep `--ulysses_size` equal to world size.

## Model Download https://github.com/Wan-Video/Wan2.2#model-download
- Download official Wan2.2 weights from the upstream release and keep them locally.
- Create per‑task folders you will pass to `--ckpt_dir`, e.g. `./Wan2.2-I2V-A14B`, `./Wan2.2-TI2V-5B`, `./Wan2.2-S2V-14B`.
- Each folder contains the VAE `.pth`, the T5 encoder weights, and DiT weights as released upstream (diffusers‑style files).
- (Optional) Verify integrity: `find <dir> -maxdepth 1 -type f -print0 | xargs -0 shasum -a 256 > checksums.txt`.

## Compiled Run Script (torch.compile + CUDA Graphs)

- Path: `scripts/run_compiled.sh`

Prerequisites (minimum recommended):
- OS: Linux x86_64; Python 3.10+; PyTorch ≥ 2.4.0.
- NVIDIA: Driver ≥ 535 (CUDA 12.x), recent GPU (Ampere/Hopper recommended).
- Workload: Prefer static shapes and fixed batch for best CUDA Graphs results.

Usage:
```bash
# Basic
scripts/run_compiled.sh -- <your_script.py> [script args...]

# Common options (condensed)
#   --python PATH       # override interpreter (default: python)
#   --mode MODE         # reduce-overhead | max-autotune | default
#   --no-cudagraphs     # do not request CUDA Graphs
#   --connections N
#   --alloc-conf STR
#   --logs +dynamo,+inductor

# Example: compile+graphs for I2V
scripts/run_compiled.sh --mode reduce-overhead -- \
  generate.py --task i2v-A14B --size 1280*720 \
  --ckpt_dir ./Wan2.2-I2V-A14B \
  --image examples/i2v_input.JPG
```

What it sets:
- `TORCH_COMPILE=1`, `TORCH_COMPILE_MODE=<mode>`
- `TORCHINDUCTOR_USE_CUDAGRAPHS=1` (unless `--no-cudagraphs`)
- `CUDA_DEVICE_MAX_CONNECTIONS`, `PYTORCH_CUDA_ALLOC_CONF`
- Optional: `TORCH_LOGS` for Dynamo/Inductor diagnostics
