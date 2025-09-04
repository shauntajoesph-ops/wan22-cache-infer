# Repository Guidelines

## Project Structure & Module Organization
- Source: `wan/` — core models under `wan/modules/` (attention, tokenizers, VAE, S2V), configs in `wan/configs/`, utilities in `wan/utils/`, distributed helpers in `wan/distributed/`.
- Entry points: `generate.py` (main CLI), plus task helpers (`wan/image2video.py`, `wan/speech2video.py`, `wan/textimage2video.py`).
- Assets & examples: images in `assets/`, sample inputs in `examples/`.
- Tooling: `pyproject.toml`, `requirements.txt`, `Makefile`, and tests in `tests/` (see `tests/test.sh`).

## Build, Test, and Development Commands
- Install: `pip install .` and `pip install .[dev]` (or `poetry install`).
- Run (single GPU): `python generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --prompt "..."`.
- Run (multi‑GPU): `torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8`.
- TeaCache (I2V/TI2V/S2V): add `--teacache --teacache_thresh 0.08` to enable conditional transformer skipping for speedups (~1.3–2.0x depending on threshold). Examples:
  - Single GPU TI2V: `python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --teacache --teacache_thresh 0.08`
  - Multi-GPU I2V: `torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --teacache --teacache_thresh 0.08`
  - Single GPU S2V: `python generate.py --task s2v-14B --ckpt_dir ./Wan2.2-S2V-14B --image examples/i2v_input.JPG --audio examples/talk.wav --teacache --teacache_thresh 0.08`
- FBCache (I2V/TI2V/S2V): add `--fbcache --fb_thresh 0.08` to enable First‑Block Cache (indicator‑only) for speedups in a similar range. Examples:
  - Single GPU TI2V: `python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --fbcache --fb_thresh 0.08`
  - Multi-GPU I2V: `torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --fbcache --fb_thresh 0.08`
  - Single GPU S2V: `python generate.py --task s2v-14B --ckpt_dir ./Wan2.2-S2V-14B --image examples/i2v_input.JPG --audio examples/talk.wav --fbcache --fb_thresh 0.08`
  - TeaCache flags:
    - `--teacache`: enable (disabled by default)
    - `--teacache_thresh`: skip aggressiveness (default 0.08)
    - `--teacache_policy`: rescale policy (`linear` default; unknown values fall back to `linear`)
    - `--teacache_warmup`: force compute first K steps (default 1)
    - `--teacache_last_steps`: force compute last K steps (default 1)
    - `--teacache_alternating`: alternate skip eligibility (default off). Helps stabilize long sequences of reuses.
    - See “Design (Merged)” below for architecture and details.
  - FBCache flags:
    - `--fbcache`: enable (disabled by default)
    - `--fb_thresh`: threshold for the (rescaled) gating metric (default 0.08)
    - `--fb_metric`: metric type (`hidden_rel_l1` default; `residual_rel_l1`, `hidden_rel_l2` optional)
    - `--fb_downsample`: stride for metric computation over tokens (default 1)
    - `--fb_ema`: EMA factor [0,1) for smoothing the metric (default 0 → off)
    - `--fb_warmup`: force compute first K steps (default 1)
    - `--fb_last_steps`: force compute last K steps (default 1)
    - `--fb_cfg_sep_diff`: compute CFG and non‑CFG diffs separately (default false)
    - See “Design (Merged)” below for full design, thresholds, and presets.
  - Notes and caveats:
    - State resets per run; cached residuals are cleared before each generation.
    - In SP/Ulysses mode, ranks synchronize a scalar decision each step; ensure ranks are aligned.
    - On shape/dtype mismatch or invalid metrics, compute is forced (failsafe) and accumulator resets.
    - Offload mode moves cached residuals to CPU when models are offloaded to release VRAM (I2V/TI2V/S2V).
    - In separate‑CFG pipelines (Wan2.2), the cond branch increments the executed step; warmup/last‑steps apply to cond; uncond shares the same timestep index.
    - TeaCache vs FBCache: prefer enabling one at a time. If both are set, FBCache takes precedence in the current implementation to avoid conflicting gates.

## Unified Cache Manager (TeaCache · FBCache · CFG)
- Overview: A single Cache Manager governs TeaCache, FBCache, and CFG cache with a unified lifecycle, priority, distributed sync, offload moves, and telemetry.
- Minimal flags (CLI → config):
  - TeaCache: `--teacache`, `--teacache_thresh`, `--teacache_policy`, `--teacache_warmup`, `--teacache_last_steps`.
  - FBCache: `--fbcache`, `--fb_thresh`, `--fb_metric` (`hidden_rel_l1` | `residual_rel_l1` | `hidden_rel_l2`), `--fb_downsample`, `--fb_ema`, `--fb_warmup`, `--fb_last_steps`, `--fb_cfg_sep_diff`.
  - CFG cache: cond‑authoritative by default (reuse cond metrics/action for uncond within a step); `cfg_sep_diff=false` default.
- Priority: FBCache → TeaCache → Compute. First eligible skip wins.
- Lifecycle: init → warmup → main → last‑steps → reset. Warmup/last‑steps apply to cond branch; uncond shares the same timestep index.
- Distributed: In SP (Ulysses) mode, scalar gating metrics are fp32 all‑reduced (mean) across ranks; failures force local compute (failsafe).
- Offload: Residuals move with the model; pipelines call manager move hooks on CPU↔GPU transitions to avoid stale device state.
- Telemetry: End‑of‑run summary printed on rank 0 (per expert for I2V); includes per‑branch totals/skips/averages, pair stats, and failsafe count.
- Migration: Legacy Tea/FBCache attach code may still exist for compatibility, but gating is managed by the Cache Manager in model forwards.

## Performance Hooks (Compile · CUDA Graphs)
- torch.compile: `--compile reduce-overhead|max-autotune` (default `none`). Applies to non‑FSDP, non‑SP runs.
- CUDA Graphs: `--cuda_graphs` to capture steady forward passes (single‑GPU, no FSDP/SP, and `--offload_model False`). Falls back if capture fails.
- Quick bench: `bash benchmarks/bench_compile.sh <task> <ckpt_dir> [--size ...] [--compile ...] [--cuda_graphs]`.

## Compatibility Switches
- Legacy cache attach: `--legacy_cache_compat` (default off). Prefer unified Cache Manager; legacy states are for transitional purposes only.

## Design (Merged)
This section is the single source of truth for TeaCache, FBCache, and CFG cache. It consolidates all relevant content from the legacy documents (Cache Manager, TeaCache, FBCache, CFG cache, and refactor notes).

Goals and Non‑Goals
- Goals:
  - Reduce inference latency via principled cache re‑use without retraining.
  - Keep output quality close to Wan2.2 baseline at conservative settings.
  - Provide a single, auditable control plane for all caches with clear invariants.
  - Support single‑GPU, FSDP, and SP/Ulysses multi‑GPU with safe fallbacks.
- Non‑Goals:
  - Training‑time caching or gradient‑aware reuse.
  - Aggressive model‑specific heuristics beyond the documented metrics/policies.

High‑Level Architecture
```
Prompt/Image/Audio → Encoders → DiT(t)
                      │
                      ├─ Block 0: modulated input (mod_inp), optional x_after_block0
                      │
   +------------------▼------------------+
   |        Unified Cache Manager        |
   |  (priority: FB → Tea → Compute)     |
   +------------------+------------------+
                      │ decision (skip/compute, resume_from)
             ┌────────┴─────────┐
             │                  │
          SKIP               COMPUTE
   x := x + residual    run DiT blocks (resume_from)
   (shape/dtype guard)  update residual := x_after − x_before
```

Lifecycle and Branching
- Lifecycle phases (per run): init → warmup → main → last‑steps → reset.
- Branches: separate‑CFG (cond/uncond). The manager only increments the executed step counter `cnt` on cond; uncond shares the same timestep index.
- Warmup/last‑steps apply to cond; when active, compute is forced regardless of metrics.

Gating Signals and Math
- Base scalar: mean absolute value of a tensor (hidden or residual), potentially downsampled along token dimension.
- Relative change per step: `rel = |cur − prev| / (|prev| + eps)` (or squared L2 variant when selected). `eps = 1e−8`.
- Rescale policy: currently `linear` (identity). Unknown identifiers fall back to identity for safety.
- Accumulator: `acc ← acc + rescaled_rel` (only when a valid previous signature exists and not in forced‑compute windows).
- Decision: skip if `acc < thresh` (TeaCache uses `--teacache_thresh`; FBCache uses `--fb_thresh`). On compute, the deciding mode’s accumulator resets to 0.

TeaCache (Across‑Step)
- What: Step‑to‑step stability gate. If modulated input hardly changes, reuse the previous step’s full‑stack residual and skip compute.
- Where: Across sampling steps for the DiT; uncond branch follows cond action by default.
- Signal: mod_inp = `norm1(x).float() * (1 + e1) + e0` at Block 0; take mean abs.
- FSDP/SP: scalar metric is all‑reduced (mean) across ranks when the process group is initialized; otherwise no reduction.
- State per branch: `prev_mod_sig`, `prev_residual`, `accum`, `shape`, `dtype`, counters (total/skipped/sum_rel/sum_rescaled/count_rel).
- Flags: `--teacache` (off), `--teacache_thresh` (0.08), `--teacache_policy` (linear), `--teacache_warmup` (1), `--teacache_last_steps` (1), `--teacache_alternating` (optional stabilizer).
- Typical speedups: ~1.4–2.2× depending on content and threshold; higher thresholds skip more but risk detail/motion staleness.

FBCache (Within‑Step)
- What: Early‑step indicator. If Block‑0 suggests little change, skip remaining blocks for this step and reuse the cached full‑stack residual.
- Metrics:
  - `hidden_rel_l1` (default): mean abs of mod_inp.
  - `hidden_rel_l2`: squared diff variant.
  - `residual_rel_l1`: compute `x_after_block0 − x_before` once, then measure mean abs (enables `resume_from_block=1`).
- Downsampling and EMA:
  - `--fb_downsample`: stride in token dimension to reduce cost (1/2/4...).
  - `--fb_ema`: EMA factor [0,1) to smooth raw metric before rescale/accumulate.
- Scope and resume: On compute with residual metric, resume from Block‑1; otherwise from Block‑0. Same per‑branch state/guards as TeaCache.
- Flags: `--fbcache`, `--fb_thresh` (0.08), `--fb_metric`, `--fb_downsample` (1), `--fb_ema` (0), `--fb_warmup` (1), `--fb_last_steps` (1), `--fb_cfg_sep_diff` (false).
- Typical speedups: ~1.2–1.6×; cheaper signal, lower risk to quality at moderate thresholds.

CFG Cache Semantics
- Default policy (`cfg_sep_diff=false`): cond‑authoritative re‑use — uncond branch follows cond’s action and rescaled metrics; no separate decision.
- Guarded reuse: when cond decides to skip, uncond only honors skip if a matching residual is available (shape/dtype/device). Otherwise, the manager synthesizes a compute decision for uncond and increments pair divergence.
- Separate diffs (`--fb_cfg_sep_diff=true`): compute CFG and non‑CFG diffs separately but still follow cond action; useful for analysis.
- Divergence failsafe: If a planned skip cannot be applied at runtime (e.g., residual dropped between `decide()` and `apply()`), `apply()` returns a negative signal and callers must compute; failsafes increment (and pair divergence for uncond).

Distributed (SP/Ulysses) and Invariants
- Reduction: For SP, all ranks compute identical scalar metrics and perform `all_reduce(mean)`. The actual group world size is queried; if uninitialized or any error occurs, compute continues locally and a failsafe is recorded.
- No conditional collectives: all ranks must call `begin_step('cond')` then `begin_step('uncond')` every step in lock‑step.
- Rank agreement: With identical inputs/flags, decisions are identical across ranks.

Offload, Device, and Dtype Handling
- Move hooks: On CPU↔GPU transitions, call manager `move_cached_residuals_to(device)` so cached residuals follow the model’s device.
- Guards on apply: Skip only if residual exists and `(shape == x.shape)` and `(dtype == x.dtype)`; else fall back to compute and increment failsafes.
- Interactions: In single‑GPU offload mode, moving residuals to CPU reduces VRAM pressure; in multi‑GPU, offload is disabled.

Telemetry and Observability
- Per‑branch: totals, skipped, skip‑rate (%), avg `rel`, avg rescaled.
- Pair‑level (CFG): planned skips, forced compute, divergence failsafes.
- Global: failsafe count; config snapshot (num_steps, warmup, last_steps, tc/fb enabled, cfg_sep_diff, priority, sp_world_size).
- Logging: One structured summary per expert (I2V low/high) or per model at end of run (rank 0).

Quality and Safety
- Trade‑offs: Higher thresholds increase speed but may cause detail staleness, temporal jitter, or motion artifacts. Use warmup/last‑steps to stabilize.
- Recommended starting points: TeaCache 0.08; FBCache 0.08–0.10; EMA off; downsample 1.
- Long sequences: Consider `--teacache_alternating` or slightly higher thresholds with last‑steps compute to preserve final frames.

Integration Patterns (Pipelines)
- Build signals once per step:
  1) Compute Block‑0 modulation and `mod_inp`.
  2) If FBCache uses `residual*`, run Block‑0 once to produce `x_after_block0`.
  3) `decision = cache_manager.decide(x, mod_inp, x_after_block0)`.
  4) `(x, resume_from, applied_skip) = cache_manager.apply(decision, x)`.
  5) If `decision.action == "compute"` or `not applied_skip`: run blocks (respect `resume_from`), then `cache_manager.update(decision, x_before, x_after)`.
- Begin step/branch:
  - Call `begin_step('cond')` before cond forward; `begin_step('uncond')` before uncond.
  - Manager increments `cnt` only on cond and tracks pair‑level stats.

Compatibility and Priority
- Priority: FBCache → TeaCache → Compute. Enable one at a time for predictability; if both are enabled, FBCache takes precedence by design.
- Legacy attach: per‑module TeaCache/FBCache states have been removed; CacheManager is authoritative across I2V/TI2V/S2V (non‑SP and SP).

Limitations and Future Work
- Policies: Only `linear` rescale is implemented; additional policies require calibration.
- Mixed‑batch gating: Metrics are batch‑averaged; per‑sample gating would add complexity and is not enabled.
- SP collectives: Reductions default to the global process group; explicit group wiring can be added if multiple groups coexist.
- CUDA Graphs/compile: Orthogonal perf features; independent of caching but often complementary in steady‑state loops.


- Tests (E2E): `bash tests/test.sh <local model dir> <gpu number>`.
- TeaCache smoke tests: `bash tests/test_teacache.sh <local model dir> [gpu number]`.
- TeaCache benchmarks: `bash benchmarks/bench_teacache.sh <local model dir>`.
- Format: `make format` (isort + yapf) or `black . && isort .`.

## Coding Style & Naming Conventions
- Python 3.10+, 4‑space indent, no trailing whitespace.
- Formatting: Black (line length 88) + isort (`profile=black`). Lint with Flake8; type‑check with MyPy (strict).
- Naming: modules/files `snake_case`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_SNAKE`.
- Keep functions small; prefer pure utils in `wan/utils/`; place new configs under `wan/configs/`.

## Testing Guidelines
- Primary tests are integration via `tests/test.sh`; place models in a folder (e.g., `/models`) and run: `bash tests/test.sh /models 8`.
- Optional unit tests: put under `tests/` as `test_*.py`; run with `pytest -q`.
- When adding features, include a minimal CLI example for `generate.py` in the PR.

## Commit & Pull Request Guidelines
- Commits: use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`). Keep subject ≤ 72 chars; include scope when helpful (e.g., `feat(utils): add UNI-PC solver`).
- PRs: clear description, linked issue, sample command, and (when relevant) VRAM/runtime notes and generated outputs (images/videos). Update README/INSTALL if flags or defaults change.

## Security & Configuration Tips
- Do not commit model weights or personal tokens. Use env vars for APIs: `DASH_API_KEY` (and `DASH_API_URL` for intl). Add secrets via your shell, not code.
- Large runs may require ≥80GB VRAM (single GPU); prefer FSDP + Ulysses for multi‑GPU.
