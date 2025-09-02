# CFG‑Aware TeaCache for Wan2.2

Status: Design spec (no code)

Scope: I2V/TI2V (Wan2.2); S2V planned for a later phase

Goals: CFG‑aware transformer skipping with strong quality controls; multi‑GPU safe; low integration friction; extensible and testable


## 1) Background & External Best Practices

- Cache‑DiT highlights
  - Dual‑block cache (Fn/Bn): compute a small prefix to measure similarity; reuse middle; compute a small tail to calibrate and curb drift.
  - CFG‑aware paths: separate vs fused CFG; reuse non‑CFG signal for CFG at the same step; track steps independently.
  - Robustness: warmup/last‑steps compute guards, diff downsampling, selective Bn, distributed all‑reduce of diffs, compile‑safe graph breaks, detailed telemetry.
  - Forecasting: Taylor‑style prediction as an alternative calibrator when steps are sparsely computed.
- Mature CFG patterns (Mochi, FLUX, Cosmos, HiDream, ControlNet)
  - Fused CFG: cond/uncond batched together; derive a single conservative decision (e.g., max of per‑branch signals).
  - Separate CFG: independent branch states; often reuse the non‑CFG diff for the CFG pass within the same timestep to avoid branch divergence.
  - Control residual awareness: treat strong control injections as a guardrail; avoid skipping when control deltas are high.
- Additional industry practices
  - EMA smoothing: stabilize noisy signals with a short exponential moving average.
  - Noise‑phase adaptivity: loosen thresholds at high noise, tighten at low noise.
  - Quality budgets: target skip rates under PSNR/LPIPS/FID budgets (ablation driven).
  - Scene‑change guards (video): detect large latent deltas; temporarily suspend skipping.
  - Deterministic distributed ops: fixed reduction groups and dtypes for reproducible gating decisions.
  - Dry‑run profiling: collect signals/decisions without skipping to calibrate thresholds and policies.


## 2) Architecture Overview

Components

- CFGCacheManager: orchestrates global run state; owns per‑branch sub‑states; applies decisions.
- BranchState (cond/uncond): per‑branch signature, accumulator, cached correction (residual or hidden‑state), guards (shape/dtype/run id), counters.
- SignalExtractor: computes a low‑cost scalar from early transformer computation; supports downsampling and control‑aware augmentation.
- PolicyEngine: pluggable strategies for rescaling, branch/pair decisions, EMA smoothing, and noise‑phase adaptivity.
- Calibrator: optional Bn tail compute and/or Taylor‑style forecasting to reduce drift when reusing.
- DistSync: abstracts reductions of scalar decisions across sequence‑parallel/data‑parallel groups.
- OffloadPolicy: manages where caches live (GPU/CPU/auto) with prefetch/evict hooks.
- TelemetrySink: collects counters, histograms, and summaries; supports console/CSV/TensorBoard.

Dataflow (per step)

1) Warmup/last‑steps guard → 2) extract early signal (Fn) → 3) compute relative change vs previous step → 4) rescale via policy (EMA/noise adapt) → 5) distributed reduction (as configured) → 6) decide skip vs compute → 7a) on skip: apply cached correction + optional Bn subset; 7b) on compute: refresh caches and accumulators → 8) update telemetry.

Wan2.2 specifics

- Wan2.2 uses separate CFG (non‑CFG and CFG forwards are distinct); design defaults to `cfg_mode=separate` with optional `shared_diff` reuse.
- I2V has experts; each expert maintains an independent cache; first call after expert swap forces compute and resets branch states.


## 3) APIs & Flags (with Defaults)

CLI flags (extend TeaCache)

- `--teacache`: enable (default: false).
- `--teacache_thresh`: base threshold (default: 0.08).
- `--teacache_policy`: `linear` | `poly:<name>` | `pair_max` | `pair_mean` (default: linear).
- `--teacache_warmup`: K warmup executed steps (default: 1).
- `--teacache_last_steps`: last K steps forced compute (default: 1).
- `--teacache_downsample`: diff stride 1/2/4 (default: 1).
- `--teacache_bn`: tail blocks to compute even on skip (default: 0).
- `--teacache_bn_ids`: optional subset within the tail window [N‑Bn,…,N‑1] (default: empty → compute all tail blocks if `bn>0`).
- `--teacache_cfg_mode`: `separate` | `shared_diff` | `fused` (default: separate).
- `--teacache_cfg_first`: compute CFG first (default: false → non‑CFG first).
- `--teacache_cfg_sep_diff`: compute diffs independently for CFG (default: true).
- `--teacache_alternating`: alternate skip eligibility (default: false).
- `--teacache_ctrl_guard`: control‑aware guard threshold (default: 0 → disabled).
- `--teacache_dist_scope`: `sp` | `dp` | `node` | `none` (default: sp).
- `--teacache_offload`: `gpu` | `cpu` | `auto` (default: auto).
- `--teacache_limits`: `max_cached=<N>,max_continuous=<M>` (default: disabled).
- `--teacache_debug`: bool/int for verbosity or per‑step interval (default: false).
- Optional advanced:
  - `--teacache_ema`: EMA factor in [0,1) to smooth the signal (default: 0 → off).
  - `--teacache_noise_adapt`: noise‑aware scaling: `none` | `linear` | `table:<name>` (default: none).
  - `--teacache_scene_guard`: video latent delta threshold (default: 0 → off).

Programmatic (attached to WanModel)

- `model.teacache.enabled: bool`
- `model.teacache.cfg_mode: enum`
- `model.teacache.branch: {'cond','uncond'}` (separate CFG only)
- `model.teacache.attach(run_id, steps, dist_groups, device, dtype, offload_policy)`
- `model.teacache.stats(): dict` for telemetry dashboards and run summaries


## 4) Gating Signal & Decision Policies

Signal sources

- Early‑block modulated/normalized hidden‑state scalar (default): stable, low‑cost, and aligns with existing TeaCache.
- Residual‑based signal (Fn residual): DBCache‑style; allows switching between residual and hidden‑state based gating per model.
- Control‑aware augmentation (optional): include a scalar derived from control feature deltas when ControlNet‑like branches are active.

Similarity metric

- Relative L1: `mean(|Δ|) / mean(|prev|)` with fp32 accumulation on CPU; robust to scale shifts.
- Token saliency weighting (optional): focus on tokens with `|Δ|/|prev| > τ`; fallback to full mean if none exceed `τ`.
- EMA smoothing (optional): `rel_ema = ema*prev + (1-ema)*rel` before thresholding.

Rescale policies

- `linear`: identity mapping (default).
- `poly:<name>`: polynomial mapping calibrated offline for a task/model profile.
- `pair_max` / `pair_mean`: for fused CFG; conservative vs smoother combination of branch scalars.
- Noise‑aware: multiply effective threshold by a schedule‑dependent factor (looser at high noise, stricter at low noise).

Decision modes

- Branch‑first (separate): decide per branch; if `cfg_sep_diff=false`, reuse the non‑CFG diff for CFG at the same step.
- Pair‑level (fused): one decision governs both branches; if either branch’s signal is NaN/Inf, force compute both and reset caches.
- Alternating: only every other eligible step may skip to limit long‑run drift.


## 5) CFG Modes & Timestepping

Separate CFG

- Step order: non‑CFG then CFG (default), or flipped with `--teacache_cfg_first`.
- Executed step: increment on non‑CFG only; CFG shares the logical step index; ensures warmup/last‑steps apply once per denoising step.
- Diff reuse: if `--teacache_cfg_sep_diff=false`, reuse the non‑CFG diff for CFG at the same timestep for coherence.

Fused CFG

- Batch order `[uncond, cond]`; extract two scalars; combine with pair policy.
- Cache keys carry branch index when storing hidden‑state caches; residual caches may be shared if correction is symmetric.

Control‑aware guard

- Compute a control delta (relative L1 on control features). If above `--teacache_ctrl_guard`, force compute both branches and reset alternating state to preserve guidance fidelity.


## 6) Calibrators: Bn & Forecasting

Tail calibrator (Bn)

- Compute last `bn` blocks even on skipped steps; if `bn_ids` set, treat indices relative to the tail window [N‑Bn,…,N‑1].
- Use stricter thresholds for skipped tail blocks when applying `bn_ids` selection.

Forecasting (optional)

- Taylor‑style predictor: use small order (≤2) derivatives to approximate hidden states between computed steps; guarded by `max_continuous` skips.
- Disable or fallback to Bn when the scheduler changes or scene change is detected.


## 7) Distributed Behavior

Reduction scope

- SP (sequence‑parallel): all‑reduce the scalar gating signal across the SP group (recommended default for Ulysses‑style sharding).
- DP (data‑parallel): optional all‑reduce across replicas (usually not needed if inputs are identical).
- Node/global: avoid across parameter shards; never reduce across tensor/FSDP shards.

Consistency & reproducibility

- Fixed process groups and dtypes; identical flags across ranks; pure function of reduced scalars.
- On reduction failure or mismatch, force compute, reset branch cache, and increment `failsafe_count`.
- Disallow `--teacache_dist_scope` changes mid‑run; lock scope at attach time.


## 8) Offload & Memory Rules

Storage policy

- `gpu`: keep caches on device; fastest but VRAM heavy.
- `cpu`: store caches in pinned CPU; prefetch on use; moderate overhead.
- `auto`: if VRAM watermark exceeds a threshold (e.g., 85–90% of reserved), switch to CPU for remainder of run.

Guards & eviction

- Enforce shape/dtype/device match; on mismatch, force compute and reset the affected branch cache.
- Respect `max_cached` and `max_continuous` caps; reset continuity counters when exceeded.
- OOM trap: on allocation failure, free cache, switch to `cpu` offload policy, log once, and continue.


## 9) Debug, Telemetry, and Tooling

Counters

- Per branch: `total`, `skipped`, `avg_rel`, `avg_rescaled`, `max_rel`, `failsafe_count`.
- Global: `skip_rate`, `bn_usage`, `scene_guard_hits`, `oom_switches`.

Histograms

- Relative diff histograms per branch for step ranges (e.g., first/median/last thirds) to characterize drift patterns.

Logging & sinks

- End‑of‑run summary (rank 0): task, resolution, steps, GPUs, SP size, flags, speedup, counters.
- Optional per‑step logs at interval K or on decision flips; include `(step, branch, rel, rescaled, decision, bn_used, offload_state)`.
- Sinks: console (default), CSV, TensorBoard; off by default except console.

Dry‑run mode

- Compute signals and prospective decisions without actually skipping; used to calibrate thresholds/policies and build presets.


## 10) Performance & Quality Targets

Expected speedups

- With `thresh=0.06–0.08`, `bn=0–4`, `downsample=1–2`: 1.3–2.0× depending on resolution and content dynamics.

Quality safeguards

- Warmup and last‑steps compute; pair‑level conservative decision for fused CFG; control‑aware guard to avoid quality loss under strong conditioning.

Quality budgets (optional)

- “Budgeted mode”: target skip rate while bounding PSNR/LPIPS deltas; requires metrics hooks in test harness.


## 11) Presets (User‑Friendly)

- `safe`: `thresh=0.06`, `bn=4`, `downsample=1`, `cfg_sep_diff=true`, `alternating=false`.
- `balanced`: `thresh=0.08`, `bn=2`, `downsample=2`, `cfg_sep_diff=true`, `alternating=true`.
- `turbo`: `thresh=0.12`, `bn=0`, `downsample=4`, `cfg_sep_diff=false`, `alternating=true`, small `scene_guard`.
- `hybrid`: `thresh=0.10`, `bn=0`, `forecasting=order2`, `max_continuous=3`, noise‑aware scaling enabled.


## 12) Integration Steps (Wan2.2/TeaCache)

- Flags: add CFG/Bn/downsample/offload/dist flags in `generate.py`; propagate into I2V/TI2V pipelines.
- Attach per‑model: create and attach CFGCacheManager per WanModel instance (I2V experts independent); set `cfg_mode` and dist groups.
- Separate CFG (Wan2.2): set branch before each forward; increment executed step on non‑CFG only; reuse diff for CFG if configured.
- Fused CFG (if present): ensure pair‑level policy and branch‑tagged caches.
- SP/FSDP: use SP process group for scalar reductions; keep cache application local to shards.
- Offload: reuse existing offload infra; add prefetch/evict hooks in the cache manager.
- Telemetry: print end‑of‑run summaries; expose `stats()` for test runners and benchmarks.


## 13) Testing & Calibration Playbook

Unit tests

- Signal extraction stability across dtypes/devices; EMA and rescale correctness.
- Offload move/cast helpers; guard behavior (shape/dtype/device).
- Policy edge cases: alternating, shared_diff, bn_ids bounds, ctrl_guard toggling.

Integration (single GPU)

- Fixed‑seed I2V/TI2V runs at 480p/720p with multiple thresholds; verify no runtime errors and reasonable skip rates.
- Visual spot‑checks; compute PSNR/LPIPS/FID deltas for conservative presets.

Multi‑GPU

- SP and SP+FSDP on 2/8 GPUs; verify consistent decisions across ranks; test offload transitions (auto→cpu).

Stress & failure paths

- Expert swaps; scheduler changes; dynamic shape changes (force reset); OOM path; dist reduce failures; NaN/Inf signal injection.

Calibration

- Dry‑run on representative prompts to collect rel/rescaled histograms; select thresholds/presets per task; validate with PSNR/LPIPS/FID.


## 14) Fallback & Compatibility Handling

- Compile: limit graph breaks to decision boundaries; provide a flag to run in eager (no compile) if desired.
- Mixed precision: force fp32 for signal math; keep caches in model precision; cast on apply.
- Dynamic shapes: on transformer input shape change, clear caches, restart warmup, and log the event.
- Failure taxonomy: categorize fail‑safes (`nan_inf`, `reduce_err`, `shape_mismatch`, `oom`, `policy_guard`); each increments dedicated counters and triggers force compute.


## 15) Extensibility Plan

Modular registries

- SignalExtractorRegistry: register extractors by name (e.g., `modulated_norm`, `fn_residual`, `control_aug`).
- PolicyRegistry: register rescale and decision policies (e.g., `linear`, `poly:wan_i2v_v1`, `pair_max`, `noise_adapt:linear`).
- CalibratorRegistry: register Bn strategies and forecasting variants (e.g., `bn_subset`, `taylor:order2`).
- DistSyncRegistry: register reduction strategies (e.g., `sp_mean`, `dp_mean`, `noop`).
- OffloadRegistry: register offload policies (e.g., `gpu`, `cpu`, `auto_vram90`).
- TelemetryRegistry: register sinks (e.g., `console`, `csv`, `tensorboard`).

Plugin protocols

- Simple call interfaces with pure‑function semantics and clear contracts; inputs are tensors/scalars plus context; outputs are scalars/decisions/corrections.
- Configurable dotted‑path loading via CLI/config for signals, policies, and calibrators; no hard dependency on diffusers.

Configuration

- YAML/JSON support for presets and profiles; CLI string flags map to registry names; environment‑based overrides supported.

Backwards compatibility

- Default implementations mirror current TeaCache; new behaviors are additive and disabled by default.

Documentation

- Developer guide for adding a new signal/policy/calibrator with examples and a test checklist.


## 16) Roadmap & Future Work

- Polynomial profiles: calibrated rescale curves per model/task (Wan I2V/TI2V) from ablation data.
- Wider signatures: 8–32D pooled vectors with cosine similarity (evaluate cost/benefit vs scalar).
- Adaptive thresholding: controller that targets skip‑rate or quality budget based on online drift estimates.
- Video‑specific guards: latent scene‑change detector; motion magnitude as a temporal guard.
- S2V extension: temporal token saliency and voice‑condition feature change as guards.
- Quantization awareness: validate under int8/nf4; keep signal math in fp32; adjust thresholds accordingly.
- Cross‑pipeline consistency: if multiple transformers run in a pipeline, use a shared manager with per‑stage contexts for coherent decisions.


Appendix: Default Summary

- Defaults: `cfg_mode=separate`, `cfg_sep_diff=true`, `cfg_first=false`, `thresh=0.08`, `policy=linear`, `warmup=1`, `last_steps=1`, `downsample=1`, `bn=0`, `bn_ids=[]`, `alternating=false`, `ctrl_guard=0`, `dist_scope=sp`, `offload=auto`, `limits=disabled`, `ema=0`, `noise_adapt=none`, `scene_guard=0`.

