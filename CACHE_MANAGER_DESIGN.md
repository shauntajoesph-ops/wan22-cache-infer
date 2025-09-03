# Unified Cache Manager Design (TeaCache, FBCache, CFG)

Status: Design spec (no code)

Owners: video-inference

Scope: Wan2.2 I2V/TI2V/S2V (inference-only)


## Goals

- Replace prior “orchestrator” concept with a clearer, maintainable Cache Manager.
- Provide one place to govern TeaCache, FBCache, and CFG-aware decisions with consistent lifecycle, distributed sync, offload rules, and telemetry.
- Keep configuration minimal and defaults conservative; prefer safety over aggressiveness.
- Make the design easy to implement and verify on first pass, without ambiguity.


## Non-Goals

- No training-time behaviors or optimizer state.
- No new CLI surface beyond minimal, task-relevant flags.
- No speculative features (forecasting, calibrator Bn windows) beyond what’s defined here.


## Terminology

- Branch: CFG branch label, one of `cond` or `uncond`.
- Mode: Caching strategy identifier, one of `fb` (FBCache) or `tc` (TeaCache).
- Step: A denoising iteration in diffusion sampling.
- SP: Sequence-parallel sharding (Ulysses-style) group for scalar reduction.


## Architecture Overview

- CacheManager (core): Manages lifecycle, decisions, residual caches, distributed sync, offload moves, telemetry, and fail-safes.
- Submodules (logical responsibilities; implement as cohesive helpers/classes under the same package):
  - CMConfig: Validated, immutable configuration for a run (global knobs and priorities).
  - CMState: Runtime mutable state across branches (counters, accumulators, cached residuals, guards).
  - CMSignal: Signal extraction utilities for both modes (TeaCache/FBCache) and optional EMA smoothing.
  - CMPriority: Decision evaluation order of modes (first skip-eligible mode wins).
  - CMDistSync: Distributed scalar reduction helpers (SP mean, guarded by fail-safes).
  - CMOffload: Helpers to move cached residuals CPU↔GPU and to handle model swaps/offloads safely.
  - CMTelemetry: Collection and summarization of per-branch counters, averages, and global fail-safes.
  - CMFailSafes: Enumerated reasons and policies for falling back to compute.


## Minimal Configuration

- TeaCache
  - enable_tc: bool (default false)
  - tc_thresh: float (default 0.08)
  - tc_policy: str (default "linear"; unknown → identity)

- FBCache
  - enable_fb: bool (default false)
  - fb_thresh: float (default 0.08)
  - fb_metric: str (default "hidden_rel_l1"; options: "hidden_rel_l1", "hidden_rel_l2", "residual_rel_l1")
  - fb_downsample: int (default 1)
  - fb_ema: float in [0,1) (default 0 → off)
  - fb_cfg_sep_diff: bool (default true; if false, reuse cond diff for uncond within the same step)

- CFG Cache (pair-level policy)
  - cfg_sep_diff: bool (default false; reuse cond diff and action for uncond within the same step to maintain pair consistency)
  - cfg_mode: str (default "separate"); for Wan2.2 only separate is supported. If a fused-CFG forward exists, a future "fused" mode may be enabled with pair aggregation. Not exposed as CLI for now.
  - Notes:
    - With `cfg_sep_diff=false` (default), the cond branch’s rescaled metric and action are authoritative for the uncond branch in the same step. This prevents divergence without changing pipeline structure.
    - If `cfg_sep_diff=true`, the uncond branch computes its own metric; the manager still enforces cond-authoritative action unless a hard failsafe requires uncond compute.

- Lifecycle (global)
  - warmup: int (default 1)
  - last_steps: int (default 1)
  - num_steps: int (set at attach, required)

- Priority
  - evaluation_order: tuple of modes (default ("fb", "tc"))

- Distributed
  - sp_world_size: int (default 1; >1 enables SP scalar reduction)


## Data Model

- BranchState (per branch: cond/uncond)
  - residual: Optional[tensor] — cached full-stack residual applied on skip
  - shape: Optional[tuple] — guard for residual shape
  - dtype: Optional[dtype] — guard for residual dtype
  - tc_prev_sig: Optional[float] — TeaCache signature from prior step
  - tc_accum: float — TeaCache accumulator for rescaled rel
  - fb_prev_sig: Optional[float] — FBCache signature from prior step
  - fb_accum: float — FBCache accumulator for rescaled rel
  - total: int — gating decisions count
  - skipped: int — skip decisions taken
  - sum_rel: float — sum of raw rel values
  - sum_rescaled: float — sum of rescaled values
  - count_rel: int — number of rel samples tracked

- GlobalState
  - branch: Branch — active branch
  - cnt: int — executed step counter (increments on cond only)
  - failsafe_count: int — forced compute count due to anomalies/guards
  - last_cond_rel: dict[mode→Optional[float]] — last cond rel per mode (for optional reuse)
  - last_cond_rescaled: dict[mode→Optional[float]] — last cond rescaled per mode

- Decision
  - action: "skip" | "compute"
  - mode: Optional[mode] — deciding mode, if any
  - resume_from_block: int — 0 for full stack, 1 if block0 already computed (residual metric)
  - reason: str — e.g., "acc<fb_thresh", "forced", "no-mode", "tc>=thresh"
  - rel: float — raw relative metric after reduction
  - rel_rescaled: float — rescaled value added to accumulator


## Public API

- attach(num_steps: int, sp_world_size: int = 1) -> None
  - Binds the manager to a run, sets lifecycle bounds and SP size, and calls reset().

- reset() -> None
  - Clears both BranchState objects, zeroes counters and accumulators, resets last_cond_* caches, sets cnt=0.

- begin_step(branch: Literal['cond','uncond']) -> None
  - Sets active branch. Increments `cnt` only when branch is 'cond'.

- decide(x: Tensor, mod_inp: Optional[Tensor], x_after_block0: Optional[Tensor] = None) -> Decision
  - Computes scalar metric(s) for enabled modes in priority order.
  - Applies lifecycle guards (warmup/last-steps on cond), performs SP reduction of scalar metrics, rescaling/EMA as configured.
  - CFG cache semantics:
    - Separate-CFG (Wan2.2): on 'cond', compute normally and record pair-level action and rescaled metric for reuse; on 'uncond', if `cfg_sep_diff=false`, reuse cond rescaled metric and action. If `cfg_sep_diff=true`, compute a separate metric but the action still follows cond unless a failsafe forces compute.
    - Fused-CFG (future): when both branches are available in one call, compute per-branch metrics and aggregate with a conservative pair policy (pair_max) before deciding; both branches take the same action.
  - Returns the first SKIP-eligible decision (respecting priority); otherwise returns COMPUTE decision for the last evaluated mode (or no-mode if none enabled).

- apply(decision: Decision, x: Tensor) -> tuple[Tensor, int]
  - On SKIP, validates cached residual against shape/dtype, casts to current device/dtype, and returns `x+residual` and `decision.resume_from_block`.
  - On guard failure (shape/dtype mismatch/missing residual), converts to COMPUTE path and bumps failsafe_count.
  - On COMPUTE, returns input `x` unchanged with the same resume index.
  - CFG invariants:
    - Separate-CFG: uncond branch follows the cond decision. If cond skipped but uncond cannot apply residual due to a guard, uncond computes and the event is recorded as a pair-consistency failsafe. Cond’s prior skip is not retroactively changed.

- update(decision: Decision, x_before: Tensor, x_after: Tensor) -> None
  - Caches full-stack residual as `(x_after - x_before)` (detached, current dtype), updates shape/dtype guards.
  - Resets the accumulator of the deciding mode (or both if mode is None).
  - Pair bookkeeping: record last cond rel/rescaled metric per mode for within-step reuse; record pair action for the current step.

- move_cached_residuals_to(device: torch.device) -> None
  - Moves cached residual tensors for both branches to the specified device (CPU↔GPU), used when models offload/reload.

- summary() -> dict
  - Returns a dictionary with per-branch totals, skipped counts, skip_rate, avg_rel, avg_rescaled, and global failsafe_count and config echoes.


## Lifecycle

- init: attach() sets `num_steps`, `sp_world_size`, then reset().
- warmup: for the first `warmup` executed steps (cond branch only), force COMPUTE decisions.
- main: normal gating using accumulated rescaled rel; first mode whose accumulator remains below threshold yields SKIP.
- last-steps: for the last `last_steps` executed steps (cond branch only), force COMPUTE decisions.
- reset: at the end of run or after model swap/offload if required, clears all state.

Branch ordering within a step is always cond then uncond for separate-CFG pipelines; `cnt` increments only on cond. The cond action is authoritative for the uncond branch within the same step.


## Signal Extraction (CMSignal)

- TeaCache
  - Input: `mod_inp` (normalized + time-modulated input to the first self-attention block).
  - Signature: mean absolute value (scalar float on CPU).
  - Rel: `abs(cur - prev) / (abs(prev) + eps)`.
  - Rescale: policy-based; default identity (linear). Unknown identifiers fall back to identity.

- FBCache
  - hidden_rel_l1: signature from `mod_inp` with optional `downsample` stride, rel via L1.
  - hidden_rel_l2: signature from `mod_inp` with optional `downsample` stride, rel via squared difference normalized by |prev|+eps.
  - residual_rel_l1: requires `x_after_block0`; compute residual `r1 = x_after_block0 - x` and derive signature with optional `downsample`.
  - EMA: if `fb_ema > 0`, smooth the raw rel before rescaling using `rel = alpha * prev_rel + (1 - alpha) * rel`.

All signals become Python floats (CPU) for branch state storage; SP reduction re-materializes on device for collective ops and returns a float.


## Distributed Sync (CMDistSync)

- Scope: Sequence-parallel only (SP group); `sp_world_size > 1` enables reductions.
- Operation: fp32 `all_reduce(SUM)` then divide by world size to obtain mean.
- Guards: if `torch.distributed` is not initialized or the op fails, return the local value and increment `failsafe_count`.
- Determinism: rely on fixed group membership and dtype for reproducible decisions per step.


## Offload Rules (CMOffload)

- Storage policy: manager stores cached residuals in the dtype of the model’s compute; device is whatever the model is on at update time.
- Hooks: pipelines must call `move_cached_residuals_to(device)` when moving models CPU↔GPU.
- Guards: on apply(), residuals are validated for matching `shape` and cast to current device/dtype; mismatches force compute and record a failsafe.
- OOM fallback: if residual device move fails due to OOM, clear the residual for that branch, increment `failsafe_count`, and force compute.


## Priority and Decision Policy (CMPriority)

- Evaluation order: `evaluation_order` tuple; default `('fb', 'tc')`.
- First-mode-wins: iterate enabled modes in order; the first that yields SKIP (accumulator + current rescaled rel < threshold) decides the step.
- If no mode is enabled or eligible, return a COMPUTE decision (`mode=None`, reason `no-mode`).
- CFG cache:
  - Separate-CFG: cond’s decision (skip/compute) is authoritative for both branches; uncond uses `last_cond_rescaled[mode]` when `cfg_sep_diff=false` (default). When `cfg_sep_diff=true`, uncond computes its own metric but still follows cond’s action unless a failsafe requires compute.
  - Fused-CFG (future): aggregate per-branch metrics using `pair_max` (default) or `pair_mean` to produce a conservative decision applied to both branches.


## Telemetry (CMTelemetry)

- Per branch (cond/uncond):
  - total, skipped, skip_rate = 100*skipped/total
  - avg_rel = sum_rel / count_rel
  - avg_rescaled = sum_rescaled / count_rel
- Global:
  - failsafe_count
  - echoed config: num_steps, warmup, last_steps, tc/fb enabled, priority, sp_world_size
  - pair stats: `pair_total`, `pair_skipped`, `pair_forced_compute`, `pair_divergence_failsafes`
- Reporting: manager exposes `summary()`; pipelines log once at end of run on rank 0.


## Fail-Safes (CMFailSafes)

- InvalidMetric: NaN/Inf from signal or rescale → force compute; reset relevant accumulator; increment `failsafe_count`.
- ReduceError: distributed reduction failure → use local metric; increment `failsafe_count`.
- ShapeMismatch: cached residual shape differs at apply() → force compute; increment `failsafe_count`.
- DTypeMismatch: dtype mismatch cannot be cast appropriately → force compute; increment `failsafe_count`.
- MissingResidual: skip requested but residual missing → force compute; increment `failsafe_count`.
- OOMOnMove: device move/cast OOM → drop residual for branch; force compute; increment `failsafe_count`.
- LifecycleGuard: warmup/last-steps enforcement → compute with reason `forced` (does not increment `failsafe_count`).
- PairConsistency: in separate-CFG mode, if the uncond branch cannot follow cond’s authoritative action due to a guard (e.g., missing residual), force compute uncond and increment `failsafe_count` under `pair_divergence_failsafes`.


## Integration Blueprint

- Pipelines (I2V/TI2V/S2V)
  - At timesteps known: construct `CacheManager` with `CMConfig` and call `attach(num_steps, sp_world_size)` once per model (per expert in I2V).
  - On each step: `begin_step('cond')` → call model forward (which will internally query manager) → `begin_step('uncond')` → call model forward again. The manager enforces cond-authoritative action for uncond.
  - Offload: when moving models CPU↔GPU, call `move_cached_residuals_to(target_device)` to keep cache device in sync.
  - End of run: log `summary()` on rank 0.

- Model forward (non-SP/SP)
  - Compute block-0 normalized + modulated input (`mod_inp`) once per forward.
  - If FBCache metric is `residual_rel_l1`, compute `x_after_block0` once as part of the signal path.
  - Call `decide(x, mod_inp, x_after_block0)` to get a Decision. In separate-CFG, `decide` on uncond reuses cond rescaled metric (default) and cond action to ensure pair consistency.
  - Call `apply(decision, x)` which returns `(x_or_xplusres, resume_from_block)`.
  - If `decision.action == 'compute'`:
    - If `resume_from_block == 0`, run all blocks; else reuse `x_after_block0` and continue from block 1.
    - After compute: `update(decision, x_before, x_after)` to refresh residuals and reset accumulators for the deciding mode.


## API Signatures (Ready to Implement)

- class CacheManager:
  - __init__(config: CMConfig)
  - attach(self, num_steps: int, sp_world_size: int = 1) -> None
  - reset(self) -> None
  - begin_step(self, branch: Literal['cond','uncond']) -> None
  - decide(self, x: Tensor, mod_inp: Optional[Tensor], x_after_block0: Optional[Tensor] = None) -> Decision
  - apply(self, decision: Decision, x: Tensor) -> tuple[Tensor, int]
  - update(self, decision: Decision, x_before: Tensor, x_after: Tensor) -> None
  - move_cached_residuals_to(self, device: torch.device) -> None
  - summary(self) -> dict

- class CMConfig:
  - Fields (immutable after construction): num_steps, warmup, last_steps, enable_tc, tc_thresh, tc_policy, enable_fb, fb_thresh, fb_metric, fb_downsample, fb_ema, fb_cfg_sep_diff, evaluation_order, sp_world_size

- class Decision:
  - Fields: action, mode, resume_from_block, reason, rel, rel_rescaled


## Migration Plan (from “Orchestrator” to Cache Manager)

- File/package
  - New: `wan/utils/cache_manager.py` (implementation). Keep `cache_orchestrator.py` temporarily exporting a compatibility alias `CacheOrchestrator = CacheManager`.
  - Update imports in models and pipelines to reference `CacheManager` directly.

- Class/Identifier mapping
  - CacheOrchestrator → CacheManager
  - OrchestratorConfig → CMConfig
  - Decision (unchanged)

- Pipelines
  - Replace constructions to use `CacheManager(CMConfig(...))` and `attach(...)`.
  - Replace any direct references to `enable_teacache` / `enable_fbcache` gating in forwards with manager presence checks only.
  - Ensure per-step lifecycle uses `begin_step('cond')` and `begin_step('uncond')` consistently across I2V/TI2V/S2V.

- Models
  - Replace direct TeaCache/FBCache logic in forwards with the manager flow described above. Reuse existing `mod_inp` and optional `x_after_block0` to avoid duplicate compute.

- Offload
  - Replace per-feature residual move helpers with `CacheManager.move_cached_residuals_to(device)`.

- Telemetry
  - Replace ad-hoc logs with `CacheManager.summary()` output at end of run.

- Deletion window
  - Stage 1: Keep orchestrator alias for two releases; issue deprecation warnings.
  - Stage 2: Remove alias and update tests/docs accordingly.


## Verification Strategy

- Unit Tests
  - Signal extraction correctness and stability across dtypes/devices.
  - Accumulator/threshold behavior and lifecycle guards.
  - SP reduction parity and failure fallbacks.
  - Offload move/cast helpers and guard behavior (shape/dtype/device).
  - Telemetry counters and summaries.

- Integration Tests
  - Smoke tests on I2V/TI2V/S2V single-GPU and SP multi-GPU.
  - Threshold sweeps at conservative defaults (expect 1.3–2.0× speedups on some content with minimal quality deltas).
  - CFG branch handling: separate vs reuse for FBCache.

- Failure Injection
  - NaN/Inf signals, reduce errors (mocked), shape/dtype mismatches, and OOM moves should all force compute and increment `failsafe_count` without crashes.


## Appendix — Minimal CLI Cheatsheet (Mapping)

- TeaCache: `--teacache`, `--teacache_thresh`, `--teacache_policy`, `--teacache_warmup`, `--teacache_last_steps`
- FBCache: `--fbcache`, `--fb_thresh`, `--fb_metric`, `--fb_downsample`, `--fb_ema`, `--fb_warmup`, `--fb_last_steps`, `--fb_cfg_sep_diff`
- SP size wired from `--ulysses_size` (pipelines pass into `attach()`)
