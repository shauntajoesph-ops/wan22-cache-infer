# First‑Block Cache (FBCache)

Status: Design spec (no code)

Scope: Wan2.2 I2V/TI2V (Phase 1), compatible with S2V in a later phase

Goals: Deliver a precise, low‑overhead First‑Block Cache that composes cleanly with TeaCache and CFG Cache, with strong lifecycle, distributed, and offload behavior.


## 1) What FBCache Is

- Purpose: Accelerate DiT inference by skipping most transformer blocks on steps where the state change is small.
- Core idea: Use an early, cheap signal from the first transformer block to decide whether to reuse a cached approximation of the entire stack’s work.
- Reuse target: Typically the previous step’s “stack residual” (output − input of the full block stack). Safer variants can reuse first‑block output as well.
- When to skip: If a similarity metric (relative L1/L2) between current and previous early signals stays below a threshold, add the cached residual to the current input and bypass the stack.


## 2) Prior Art & Variants

- cache‑dit FBCache/DBCache:
  - FBCache corresponds to Fn=1, Bn=0 in DBCache framing.
  - Variants support hidden‑state vs residual diffs, CFG separation/reuse, distributed all‑reduce of the scalar decision, and optional Bn tail calibration.
- ParaAttention/Wavespeed patterns:
  - Early‑signal gating (often first‑block residual diff); residual reuse; optional small tail for calibration; strong synergy with sequence/context parallel.
- Common variants:
  - Indicator‑only (default): Early signal only; reused approximation is prior step’s stack residual.
  - First‑block reuse: Reuse first‑block output as well (safer, slightly lower speedup).
  - Δ‑style: Maintain and apply a predicted delta payload (more complex; guarded by tighter caps).


## 3) Architecture Overview

- Manager: Share the existing TeaCache manager interface (modeled as a “cache mode”) or provide `FBCacheState` mirroring TeaCache fields.
- Branch states: Independent per‑branch (cond/uncond) states store previous signature, cached residual, accumulator, shape/dtype guards, and counters.
- Signal extractor: Computes an early, low‑cost scalar from the first block; configurable metric and downsample.
- Policy engine: Rescales raw metric (linear/polynomial profiles), manages accumulation, warmup/last‑steps guards, and caps for continuous/total skips.
- Dist sync: Reduces a single scalar across SP group; decision is rank‑consistent; no tensor shuffling across shards.
- Offload: Stores residual on CPU when models offload; casts/moves to device/dtype on apply; OOM fallback switches to CPU storage.
- Telemetry: Per‑branch counters, residual‑diff statistics, histograms; end‑of‑run summaries.


## 4) Gating Signal & Decision Metric

- Default signal: Early hidden‑state signature based on first‑block normalized + modulated input (cheap, stable).
  - Example computation: `mod_in = norm1(x) * (1 + scale) + bias` (same tensors already present in block‑0).
  - Scalar: mean(|mod_in|) or relative L1 between current and previous signatures.
- Alternatives:
  - Residual‑based signal: Relative L1 between first‑block residuals at current vs previous steps.
  - L2 variants: Less robust to outliers; optional and disabled by default.
- Downsampling: Optionally compute the metric on a strided subset (e.g., every 2nd/4th token) to cut overhead.
- EMA smoothing (optional): Apply a short EMA to the raw metric to reduce jitter in long videos (default off).
- Decision:
  - Accumulate rescaled metric values while below a threshold; when accumulation exceeds threshold or guards trigger, compute the stack and reset.
  - Conservative by default: threshold tuned for minimal quality loss with warmup/last‑steps compute.


## 5) Thresholds & Presets

- Typical bands (model/content dependent):
  - Conservative: 0.06–0.08 → modest skip rate; ≈1.3× speedup.
  - Balanced: 0.08–0.12 → higher skip rate; ≈1.5–1.7× speedup.
  - Aggressive: 0.12–0.20 → high skip rate; up to ≈2.0×, more risk of artifacts.
- Presets:
  - `fb:video_safe`: thresh=0.06, warmup=2, last=2, downsample=2, ema=0.1, max_continuous=2.
  - `fb:balanced`: thresh=0.08, warmup=1, last=1, downsample=2.
  - `fb:turbo`: thresh=0.12, warmup=1, last=1, downsample=4, max_continuous=3.


## 6) Lifecycle & CFG Interaction

- Initialization:
  - Attach per‑model manager on run start with `num_steps`, flags, and dist groups.
  - Reset states: accumulator, signature, residual, guards; clear counters; set `run_id`.
- Per step (separate‑CFG pipelines; Wan2.2):
  - Set branch to `cond`, increment executed step counter, compute decision; then set branch to `uncond` and compute decision.
  - Option `cfg_sep_diff`: if false, reuse `cond` diff to decide `uncond` within the same step for coherence.
- Warmup & last‑steps:
  - Force full compute for first K executed steps; force compute last K steps.
- Caps & alternating:
  - Limit both total cached steps and max continuous cached steps; optional alternating eligibility (every other executed step) to reduce drift.
- Resets:
  - On expert swap, shape/resolution change, scheduler reset, OOM fallback, or anomalies (NaN/Inf metric, reduction failures).


## 7) Distributed Behavior

- Scope: Sequence‑Parallel (SP) ranks only (default). Avoid reducing across param shards (e.g., FSDP/TP groups).
- Operation: All‑reduce the fp32 scalar metric (SUM) and divide by SP world size (mean) for a unified decision.
- Reproducibility: Fixed reduction dtype and groups; deterministic policy based solely on reduced scalar and static flags.
- Fail‑safes: If reduce fails or groups are misconfigured, force compute, reset state, increment a failsafe counter, and continue.


## 8) Offload & Memory Rules

- Storage policy:
  - GPU: fastest; uses device memory; default when headroom exists.
  - CPU: pinned CPU storage; prefetch to device on apply; moderate overhead.
  - Auto (future): switch to CPU when VRAM watermark exceeded; stay for remainder of run.
- Moves & casts: On apply, move cached residual to current device and cast to compute dtype; guard shape/dtype equality.
- OOM fallback: On allocation failure, free cache, switch to CPU policy, log once, and continue.


## 9) API & Flags

- CLI (proposed):
  - `--fbcache`: enable First‑Block Cache (disabled by default).
  - `--fb_thresh`: base threshold (default 0.08).
  - `--fb_warmup`, `--fb_last_steps`: warmup/last step guards (default 1/1).
  - `--fb_metric`: `hidden_rel_l1|residual_rel_l1|hidden_rel_l2` (default `hidden_rel_l1`).
  - `--fb_downsample`: stride for metric (1/2/4; default 1).
  - `--fb_ema`: EMA factor [0,1) (default 0 → off).
  - `--fb_cfg_sep_diff`: compute CFG diff separately (default true).
  - `--fb_dist_scope`: `sp|dp|none` (default `sp`).
  - Advanced: `--fb_first_block_reuse` (reuse first‑block output), `--fb_delta_payload` (Δ‑style; guarded), `--fb_max_continuous`, `--fb_max_cached`.
- Programmatic:
  - `model.fbcache.enabled: bool`
  - `model.fbcache.attach(run_id, steps, dist_groups, device, dtype)`
  - `model.fbcache.branch = {'cond','uncond'}` (separate‑CFG)
  - `model.fbcache.stats()` for telemetry.


## 10) Telemetry & QA

- Counters (per branch): `total`, `skipped`, `avg_rel`, `avg_rescaled`, `max_rel`, `failsafe_count`.
- Histograms: Relative diff histograms for early/mid/late thirds of the run to visualize drift.
- Summaries: End‑of‑run line on rank‑0 with task, res, steps, GPUs, SP size, flags, skip rate, and counters.
- Sinks: Console (default); CSV/TensorBoard optional for ablation runs.
- Dry‑run mode: Compute metrics/decisions without skipping to calibrate thresholds and presets.


## 11) Composition with TeaCache & CFG Cache

- Orthogonality:
  - TeaCache: early modulated input scalar + stack residual reuse (already integrated).
  - FBCache: first‑block oriented signal (hidden/residual) + stack residual reuse.
  - CFG Cache: within‑step (cond/uncond) decision; FBCache is across steps.
- Manager unification:
  - Prefer a single cache manager with `mode in {'tc','fb','db'}` to avoid duplicated lifecycle and telemetry code.
  - Shared lifecycle: attach/reset guards, warmup/last, offload, distributed reductions.
- Operational policy:
  - Only one mode active by default (TeaCache or FBCache). Expose both for ablations but document that mixing gates can conflict unless carefully ordered.


## 12) Risks & Mitigations

- Quality drift under aggressive thresholds:
  - Mitigate via conservative defaults, warmup/last‑steps compute, continuous skip caps, and optional alternating eligibility.
- Metric sensitivity in long videos:
  - Downsample the metric, allow EMA smoothing, and prefer hidden‑state relative L1 as default.
- Distributed divergence:
  - Fix reduction dtype/scope; treat any anomalies as forced compute with a single log; never mix SP and param shard groups.
- Compile instability:
  - Localize dynamic control flow; minimize graph breaks around the decision boundary; allow an eager flag to debug.


## 13) Integration Plan (No Code Yet)

- Flags: Add FBCache CLI flags alongside TeaCache; thread through `generate.py` to pipelines.
- Attachment: Reuse TeaCache attach/reset sites; set manager `mode='fb'` or attach an `FBCacheState` side‑by‑side with identical lifecycle.
- Pipelines: Keep separate‑CFG sequencing (cond first, increments executed step; uncond second) and reuse diff policy as a switch.
- SP/FSDP: Continue using SP process group for scalar reductions; preserve shard‑local residuals; no cross‑shard tensors.
- Offload: Use the same residual offload helpers already implemented for TeaCache.
- Telemetry: Reuse end‑of‑run stats printing; add FBCache keys and preset names for clarity.


## 14) Roadmap

- Profiles: Polynomial rescale profiles calibrated per model/task; preset bundles for image/video.
- Calibrator tail (Bn>0): Optional 1–4 tail blocks on skip to reduce drift at aggressive settings.
- Δ‑style payload: Guarded by strict caps; only if calibration data shows benefit.
- Control‑aware guard: Extra threshold for strong conditioning (e.g., depth/pose) to avoid artifacts.
- S2V extension: Temporal token saliency and voice‑condition changes as additional guards.


Appendix: Defaults

- `fb_metric=hidden_rel_l1`, `fb_thresh=0.08`, `fb_downsample=1`, `fb_ema=0`, `fb_warmup=1`, `fb_last_steps=1`, `fb_cfg_sep_diff=true`, `fb_dist_scope=sp`, `max_continuous=disabled`, `max_cached=disabled`.

