# TeaCache-Style Conditional Transformer Skipping for Wan2.2 (Phase 1)

Status: Design only (pre-implementation)

Owners: video-inference

Scope: I2V (A14B) and TI2V (5B) pipelines only. S2V is Phase 2.

Goals

- Add TeaCache-style conditional transformer skipping to Wan’s DiT backbone to reduce latency with minimal quality impact.
- Keep changes minimal, reversible, and behind explicit flags (disabled by default).
- Support the repo’s single-GPU and multi-GPU (FSDP and sequence-parallel/Ulysses) patterns, both UniPC and DPM++ samplers, and model offload.
- Maintain inference-only posture — do not add training code.

Non-Goals (Phase 1)

- No S2V support (Phase 2 follow-up).
- No model-specific polynomial calibration (default linear rescale; hook provided).

---

## 1. Architecture Overview

We wrap the DiT block stack inside `WanModel` (used by I2V/TI2V) with a TeaCache-style gate:

1) At each denoising step, compute a low-cost “first-block modulated input” summary (scalar) derived from block 0’s normalization and timestep-modulation. This mirrors TeaCache’s core signal.

2) Compare to the previous step’s summary to compute a relative L1 change, rescale it by a policy (default linear / identity), and accumulate. If the accumulator stays below a threshold, skip the block stack and reuse a cached residual; otherwise, compute the blocks and refresh the residual and accumulator.

3) Maintain separate caches for CFG branches (cond/uncond). Pipelines set the active branch before each model forward. The step counter increments once per step on the cond branch only (so warmup/last-step logic aligns with actual steps).

4) In sequence-parallel (Ulysses), compute a local scalar and AllReduce (mean) to ensure a consistent decision across ranks. Residuals are sequence-sharded and added locally.

5) When models are offloaded, store cached residuals on CPU and move them to GPU on use.

Design is encapsulated in a small state object (`TeaCacheState`) and a util module. No changes are made to the internal math of blocks or attention.

---

## 2. API and Flags

CLI (in `generate.py`):

- `--teacache` (bool, default: False) — enable TeaCache gating.
- `--teacache_thresh` (float, default: 0.08) — accumulator threshold; larger → more skipping.
- `--teacache_policy` (str, default: `linear`) — rescale policy name; start with `linear` (identity). Hook for future `poly:<profile>`.
- `--teacache_warmup` (int, default: 1) — force compute for first K steps.
- `--teacache_last_steps` (int, default: 1) — force compute for last K steps.

Programmatic (on `WanModel`):

- `model.enable_teacache: bool`
- `model.teacache: TeaCacheState | None`
  - Pipelines attach an initialized state object per model after sampling timesteps are known.
  - Pipelines update `teacache.branch` to `cond` or `uncond` before each model call.

---

## 3. Data Structures

`TeaCacheBranchState`

- `prev_mod_sig: Optional[float]` — previous signature (CPU fp32 scalar).
- `prev_residual: Optional[Tensor]` — cached residual (device/dtype managed); shape matches block stack input.
- `accum: float` — accumulated rescaled change.
- `shape: Optional[Tuple[int,...]]` — tensor shape guard.
- `dtype: Optional[torch.dtype]` — dtype guard.

`TeaCacheState`

- `enabled: bool`
- `num_steps: int` — total denoising steps for this run.
- `cnt: int` — step index (incremented on cond branch only).
- `warmup: int` — warmup compute steps.
- `last_steps: int` — final compute steps.
- `thresh: float` — accumulator threshold.
- `policy: str` — rescale policy id.
- `branch: Literal['cond','uncond']` — active branch.
- `cond: TeaCacheBranchState`
- `uncond: TeaCacheBranchState`
- `run_id: int` — bump per run to prevent stale reuse.
- `sp_world_size: int` — set when using sequence-parallel, for sanity.

Helper functions (in `wan/utils/teacache.py`):

- `summarize_mod(t: Tensor) -> float` — returns mean(abs(t)).cpu().item().
- `rescale(rel: float, policy: str) -> float` — identity for `linear`. Hook for polynomials later.
- `reset(state: TeaCacheState)` — zero accumulators, clear residuals/signatures.
- `move_residual_to(residual, device, dtype) -> Tensor` — device/dtype management.

---

## 4. Algorithm Details

Let `x` be the hidden states before the transformer block stack; `e0` is time modulation output (`time_projection(...)`), shaped [B, L, 6, C]. For block 0 with layer norm `norm1` and learned modulation `modulation`:

1) Compute block-0 modulated input (same math used inside `WanAttentionBlock`):

```
norm_x = block0.norm1(x).float()                # [B, L, C]
e_chunks = (block0.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
mod_inp = norm_x * (1 + e_chunks[1].squeeze(2)) + e_chunks[0].squeeze(2)
```

2) Signature and relative change:

```
cur_sig = mean(abs(mod_inp)).cpu().item()
rel = abs(cur_sig - prev_sig) / (abs(prev_sig) + 1e-8)
rescaled = rescale(rel, policy)   # linear ⇒ rescaled = rel
accum += rescaled
```

3) Decision:

- Force compute if warmup (`cnt < warmup`) or last steps (`cnt >= num_steps - last_steps`) or `prev_sig` is None.
- Else skip if `accum < thresh`; otherwise compute and set `accum = 0`.

4) Skip path: ensure residual is on correct device/dtype, then `x = x + prev_residual`.

5) Compute path: save `x_before`, run `for block in self.blocks: x = block(...)`, then `prev_residual = (x - x_before).detach()` cast to model compute dtype.

6) Update `prev_mod_sig = cur_sig` (CPU scalar). CFG keeps separate branch states.

Sequence-parallel: compute local `rel` (or `cur_sig`) per rank and AllReduce mean for a consistent decision. Residuals are sharded; addition is local.

---

## 5. Device & Distribution Handling

Devices & offload

- Keep `prev_residual` on CPU when model is on CPU (offloaded). Before reuse, move residual to model’s device and cast to the model’s compute dtype.
- `prev_mod_sig` always stored on CPU.
- Shape/dtype guards prevent mismatched reuse.

FSDP

- TeaCache state is attached to the wrapped module; code accesses via `getattr(module, 'module', module)`.
- No changes to FSDP internals. No grad; no-sync contexts remain untouched.

Sequence-parallel (Ulysses)

- Add a single scalar AllReduce per step (mean of local `rel` or `cur_sig` deltas) for consistent gating.
- Residual is per-rank shard; addition remains local and consistent.

---

## 6. Fail-Safe Rules

At runtime, force compute and reset the affected branch state when any of the following happen:

- `prev_mod_sig` is None (first use), or `rel/rescaled` is NaN/Inf.
- Residual shape/dtype mismatch with the current `x`.
- In SP, anomaly detected (e.g., inconsistent world size or communication failure).
- After an exception or device move failure (log once and continue computing).

Each fail-safe increments a counter; final telemetry reports counts.

---

## 7. File-Level Change List (Pseudo-Snippets)

New file: `wan/utils/teacache.py`

```
class TeaCacheBranchState:
    prev_mod_sig: Optional[float]
    prev_residual: Optional[Tensor]
    accum: float
    shape: Optional[Tuple[int,...]]
    dtype: Optional[torch.dtype]

class TeaCacheState:
    enabled: bool
    num_steps: int
    cnt: int
    warmup: int
    last_steps: int
    thresh: float
    policy: str
    branch: Literal['cond','uncond']
    cond: TeaCacheBranchState
    uncond: TeaCacheBranchState
    run_id: int
    sp_world_size: int

def summarize_mod(t: Tensor) -> float: ...
def rescale(rel: float, policy: str) -> float: ...
def reset(state: TeaCacheState) -> None: ...
def move_residual_to(t: Tensor, device, dtype) -> Tensor: ...
```

Modify: `wan/modules/model.py` (WanModel)

```
# class WanModel:
self.enable_teacache: bool = False
self.teacache: Optional[TeaCacheState] = None

def forward(...):
    # compute x, e0 as today
    # build mod_inp using block0.norm1 and e0
    cur_sig = summarize_mod(mod_inp)
    if self.enable_teacache and self.teacache and self.teacache.enabled:
        st = self.teacache.cond if self.teacache.branch=='cond' else self.teacache.uncond
        force_compute = (prev_sig is None) or (cnt<warmup) or (cnt>=num_steps-last_steps)
        if not force_compute:
            rel = compute_rel(cur_sig, st.prev_mod_sig)
            rel = allreduce_mean(rel) if use_sp else rel
            st.accum += rescale(rel, policy)
            skip = st.accum < thresh
        if skip:
            res = ensure_on_device(st.prev_residual)
            x = x + res
        else:
            x_before = x
            for block in self.blocks:
                x = block(...)
            st.prev_residual = (x - x_before).detach().to(param_dtype)
            st.accum = 0.0
        st.prev_mod_sig = cur_sig
    else:
        # existing path: run blocks
```

Modify: `wan/distributed/sequence_parallel.py`

```
def sp_dit_forward(model, ...):
    # compute mod_inp similarly (block0.norm1 + e0 math)
    cur_sig = summarize_mod(mod_inp)
    # allreduce mean rel across ranks
    rel = compute_rel(cur_sig, prev_sig)
    rel = dist.all_reduce(rel, op=SUM); rel /= world_size
    # then same skip/compute logic; residual stays sharded
```

Modify: `wan/image2video.py`, `wan/textimage2video.py`

```
# argparse additions:
parser.add_argument('--teacache', action='store_true', default=False)
parser.add_argument('--teacache_thresh', type=float, default=0.08)
parser.add_argument('--teacache_policy', type=str, default='linear')
parser.add_argument('--teacache_warmup', type=int, default=1)
parser.add_argument('--teacache_last_steps', type=int, default=1)

# after sampling_steps known and models created:
state = TeaCacheState(enabled=args.teacache, num_steps=sampling_steps,
                      thresh=args.teacache_thresh, policy=args.teacache_policy,
                      warmup=args.teacache_warmup, last_steps=args.teacache_last_steps)
wan_model.enable_teacache = args.teacache
wan_model.teacache = state

# in step loop:
model.teacache.branch = 'cond'; noise_pred_cond = model(...)
model.teacache.branch = 'uncond'; noise_pred_uncond = model(...)
```

Modify: `generate.py` — add CLI flags; no behavioral change otherwise.

Documentation: README — add a short “Acceleration: TeaCache” section (flags + caveats).

---

## 8. SOP — Step-by-Step Implementation Plan

1) Utility module

- Add `wan/utils/teacache.py` with data classes, helpers, and minimal policies (`linear`).
- Include device/dtype helpers and `reset()`.

2) CLI flags

- Extend `generate.py` with TeaCache flags; thread through to I2V and TI2V pipelines.

3) Pipeline wiring

- In I2V/TI2V, after `sampling_steps` known and models instantiated, attach `TeaCacheState` to each model (two for I2V experts; one for TI2V).
- Set `branch` and increment `cnt` on `cond` once per step.

4) Non-SP model forward

- In `WanModel.forward`, compute block-0 `mod_inp`, call TeaCache gate, and perform skip/compute as needed.
- Keep existing code path if disabled or state missing.

5) SP forward path

- Update `sp_dit_forward` to mirror the gating logic, with a scalar AllReduce for `rel`.
- Ensure residual remains local shard and shape/dtype guards pass.

6) Offload compatibility

- Ensure offload paths (where models move to CPU) leave cached residual on CPU and move to GPU on use.

7) Telemetry

- Add counters and end-of-run logging in pipelines: skipped/total per branch, avg rel/rescaled, fail-safe counts.

8) Docs

- Update README with a short usage section and caveats.

9) QA and benchmarks

- Run functional tests, then perf tests; tune default threshold if needed.

---

## 9. Test Plan

Unit

- `summarize_mod` returns stable scalar across dtypes/devices.
- `rescale` respects `linear` and boundary values.
- Relative change and accumulator logic with synthetic sequences (ensure skip vs compute toggles correctly).
- Device/dtype guards: move residual CPU↔GPU and cast dtype.

Integration (single GPU)

- I2V & TI2V runs on 480p/720p with fixed seed:
  - TeaCache off vs on (thresh 0.06 and 0.08): verify no errors and reasonable skip rates.
  - Visual spot-check; compute PSNR/LPIPS deltas (expect minor changes at 0.06–0.08).

Multi-GPU

- SP + FSDP: 2 and 8 GPUs:
  - Verify consistent decisions (no mismatch logs); skip rates similar to single GPU.
  - Offload enabled: ensure no device mismatch errors; GPU memory steady.

Boundary cases

- Expert boundary (I2V): caches independent per expert; first call after swap forces compute.
- Scheduler variants: UniPC and DPM++ show similar behavior.
- Induce NaN or shape change: fail-safes trigger and run continues.

Performance benchmarks

- Report format (per run):
  - Task, resolution, steps, GPUs, SP size, FSDP on/off, offload on/off, threshold, policy.
  - Total time (baseline vs TeaCache), speedup X.
  - Skip counts: cond/uncond; skip rate %.
  - Avg rel, avg rescaled, fail-safe count.

---

## 10. Telemetry Spec and CLI Flags

Flags: see Section 2.

Counters (per model instance):

- `skipped_cond`, `total_cond`, `skipped_uncond`, `total_uncond`.
- `sum_rel`, `sum_rescaled`, `count_rel` (for averages).
- `failsafe_count`.

Logging:

- At INFO on completion (rank 0): single-line summary with the above fields.
- Optional debug (hidden flag): per-step `(step, branch, rel, rescaled, decision)` logs (rate-limited).

---

## 11. Rollback & Runtime Guards

Runtime guards (force compute + reset branch state):

- Missing `prev_mod_sig`.
- NaN/Inf in `rel`/`rescaled`.
- Shape/dtype mismatch of `prev_residual` vs current `x`.
- SP reduction failure or inconsistent world size.

Rollback:

- Disable flags (`--teacache` off) restores prior behavior without code changes.
- Code changes are isolated to one util file and guarded sections in a few methods; easy to revert.

---

## 12. Extension Points (Future)

- Model-specific polynomial rescale policies (`--teacache_policy=poly:wan_i2v_v1`).
- Wider signatures (tiny pooled vectors) if needed for sensitivity.
- Phase 2: S2V integration using segmented modulation in `WanS2VAttentionBlock`.

---

## 13. Risks & Mitigations

- Quality degradation with aggressive thresholds → conservative defaults (0.08) and warmup/last-steps compute; operator tuning.
- Device/offload mismatches → explicit move/cast helpers; shape/dtype guards; fail-safe resets.
- Distributed divergence → single scalar AllReduce; if anomaly, force compute and warn once.

---

## 14. Acceptance Criteria

- Functional parity when disabled; no change to outputs.
- With TeaCache enabled at conservative threshold, 1.3–1.5× speedup at 480p/720p with minor perceptual differences.
- Works in SP + FSDP + offload without runtime errors; telemetry reports sensible skip rates.

