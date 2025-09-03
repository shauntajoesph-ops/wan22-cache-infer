# Unified Caching Layer Refactor (TeaCache, FBCache, CFG)

Status: Implemented (orchestrator core); integration staged

Owners: video-inference

Scope: Wan2.2 I2V/TI2V (Phase 1); S2V compatible, to follow

Goals

- Replace duplicated gating logic with a single Cache Orchestrator.
- Keep configuration simple (few knobs) and defaults conservative.
- Enforce a unified lifecycle, telemetry, offload, and distributed behavior for all caches.
- Improve safety (fail‑safes) and maintainability; enable evolutions like DBCache/Bn tail later.

---

## 1) Motivation & Issues Pre‑Refactor

- Duplicated logic: Similar skip/compute control flows existed in non‑SP and SP forwards for TeaCache and FBCache, increasing bug risk.
- Mixed concerns: Model forward paths interleaved metric computation, accumulation, SP synchronization, and residual application.
- Plumbing hazards:
  - FBCache branch/counter updates missing in I2V loop → permanent warmup; no skips.
  - TI2V FBCache attachment included a stray, undefined reference → potential crash.
  - TeaCache inner reset could be called unconditionally → crash risk in single‑GPU.
- Config sprawl: Many parameters surfaced through pipelines; precedence between caches decided inside forward.

---

## 2) Architecture Overview

- **CacheOrchestrator** (wan/utils/cache_orchestrator.py)
  - Single point of control for TeaCache and FBCache with CFG policies.
  - Centralizes lifecycle, decision making, SP reduction, residual application, and telemetry.
  - Minimal external configuration; fail‑safes on anomalies.

- **Core data structures**
  - BranchState: per‑branch (cond/uncond) signatures/accumulators per mode, cached residual, guards, telemetry.
  - GlobalState: thresholds, CFG enable, mode priority, num_steps, warmup/last‑steps, SP size, executed step (cond only), CFG reuse storage, and fail‑safe counters.
  - Decision: normalized SKIP/COMPUTE result with mode, metrics, resume_from_block, and reason.

- **Separation of concerns**
  - Model forwards provide early signal inputs (e.g., block‑0 modulated input) and resume hooks.
  - Orchestrator computes scalar metric, performs SP reduce, accumulates, and returns a Decision.
  - Residual application and cache refresh are performed through orchestrator helpers.

---

## 3) API Surface

- Pipeline‑level
  - `attach(model, num_steps, sp_world_size=1)`: bind orchestrator to a model for a run and reset.
  - `reset()`: clear both branch states; start fresh.
  - `begin_step(branch)`: branch ∈ {cond, uncond}; increments executed step on cond only.
  - Offload hooks (by pipeline): move cached residuals CPU↔GPU explicitly when models move.

- Step‑level (used in forward)
  - `decide(x, mod_inp) -> Decision`: compute scalar metric(s), apply global guards, run priority modes, and return SKIP/COMPUTE.
  - `apply(decision, x) -> (x, resume_from_block)`: on SKIP, apply cached residual with guards; on COMPUTE, return resume index.
  - `update(decision, x_before, x_after)`: refresh cached residual after compute; reset accumulators for the deciding mode.
  - `summary()`: return structured telemetry for logging.

- Notes
  - CFG reuse: if enabled, uncond step reuses cond diff for within‑step consistency.
  - Priority: default `('fb','tc')`; first SKIP‑eligible mode wins; else COMPUTE.

---

## 4) Configuration Rules (Minimal)

- Exposed knobs
  - TeaCache: `teacache_thresh` (default 0.08; conservative)
  - FBCache: `fbcache_thresh` (default 0.10; moderate)
  - CFG Cache: `cfgcache_enable` (boolean toggle; default false)

- Internal defaults (not CLI)
  - Warmup/Last‑steps: 1/1; applied globally
  - Downsample: 1 (metric computation stride)
  - EMA smoothing: off (alpha kept for future)
  - Dist scope: SP only (scalar all‑reduce fp32)
  - Continuous skip caps: disabled (future option)

- Precedence
  - Default evaluation order `('fb','tc')` at attach time; conflicts resolved early; no runtime ambiguity.

---

## 5) Lifecycle Management

- Initialize: `attach(model, num_steps, sp_world_size)` → `reset()`
- Warmup: force compute for first K executed steps (cond only)
- Main: per step: cond then uncond; compute scalar(s), reduce, accumulate → SKIP/COMPUTE
- Last‑steps: force compute for final K executed steps
- Reset/Invalidation conditions: model swap, offload/reload, shape change, reduction failure, NaN/Inf metrics

---

## 6) Distributed & Offload Behavior

- SP reduction
  - fp32 scalar all‑reduce; mean over SP world size; exceptions increment `failsafe_count` and fall back to local scalar.
- Offload
  - Pipelines must move cached residuals explicitly on offload/reload to keep device/dtype/shape correct; orchestrator guards prevent misuse.

---

## 7) Telemetry & Safety

- Per‑branch telemetry
  - total, skipped, avg_rel, avg_rescaled, failsafe_count, reasons histogram
- Safety
  - On NaN/Inf metric, shape/dtype mismatch, or reduce errors → force compute; reset appropriate state and record reason.
- Summaries
  - Orchestrator `summary()` returns a consistent dict; pipelines log at end of run.

---

## 8) Integration Blueprint

- Model forward (non‑SP/SP)
  - Compute block‑0 normalized+modulated input `mod_inp` once.
  - Call orchestrator: `begin_step(branch)` → `decide(x, mod_inp)` → `apply()`.
  - If COMPUTE: run blocks; then `update()` with `(x_before, x_after)`.
  - If SKIP: continue to head.

- Pipelines (I2V/TI2V)
  - At timesteps known: `attach()` once per model; set priority and thresholds.
  - Per step: set `branch='cond'`, increment executed step; run forward; then set `branch='uncond'` and run again.
  - Offload: move residuals CPU↔GPU alongside model; no stale state.

---

## 9) Changes & Fixes Applied

- Fixed
  - FBCache branch/counter updates in I2V per‑step loop; FBCache now leaves warmup and can skip.
  - TeaCache inner reset guarded by `if inner is not None:`; avoids single‑GPU crashes.
  - Removed stray undefined reference in TI2V FBCache attachment; consistent attachment/reset.

- Added
  - `CacheOrchestrator` with unified decisions and telemetry.
  - Attach‑time assert for non‑None model; early failure if miswired.
  - Guarded patterns for inner resets; consistent branch/counter updates across pipelines.

---

## 10) Test Strategy (Addendum)

- Determinism (seed parity), cross‑device tolerance (A100/H100), SP vs single‑GPU parity.
- Numerical stability: NaN/Inf injection, precision sweeps, epsilon sensitivity.
- Threshold sweeps: quality/perf Pareto; continuous skip caps (future).
- CFG interactions: branch isolation, reuse policies.
- Long‑video: window/reset boundaries; mid‑run shape changes.
- Offload & memory: CPU↔GPU stress, VRAM bounds.
- Distributed: rank‑consistent gating; fault injection.
- Error paths: shape/dtype mismatch, missing state → safe compute.
- UX/telemetry: flags honored; defaults safe; summaries consistent.

---

## 11) Future Work

- DBCache/Bn tail calibrator under orchestrator policies to reduce drift at aggressive thresholds.
- Forecasting (Taylor‑style) option for sparse compute regimes.
- Control‑aware guards (e.g., strong conditioning deltas) and noise‑adaptive scaling profiles.
- Compile tuning: explicit graph boundaries around gating for maximum fusion elsewhere.

---

## 12) Appendix — Minimal Config Cheatsheet

- TeaCache: `--teacache_thresh 0.08`
- FBCache: `--fbcache_thresh 0.10`
- CFG: `--cfgcache_enable`
- Defaults: warmup=1, last_steps=1, SP reduction fp32, downsample=1, EMA disabled

