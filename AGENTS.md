# Repository Guidelines

## Project Structure & Module Organization
- Source: `wan/` — core models under `wan/modules/` (attention, tokenizers, VAE, S2V), configs in `wan/configs/`, utilities in `wan/utils/`, distributed helpers in `wan/distributed/`.
- Entry points: `generate.py` (main CLI), plus task helpers (`wan/text2video.py`, `wan/image2video.py`, `wan/speech2video.py`, `wan/textimage2video.py`).
- Assets & examples: images in `assets/`, sample inputs in `examples/`.
- Tooling: `pyproject.toml`, `requirements.txt`, `Makefile`, and tests in `tests/` (see `tests/test.sh`).

## Build, Test, and Development Commands
- Install: `pip install .` and `pip install .[dev]` (or `poetry install`).
- Run (single GPU): `python generate.py --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --prompt "..."`.
- Run (multi‑GPU): `torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8`.
  - TeaCache (I2V/TI2V/S2V): add `--teacache --teacache_thresh 0.08` to enable conditional transformer skipping for speedups (~1.3–2.0x depending on threshold). Examples:
    - Single GPU TI2V: `python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --teacache --teacache_thresh 0.08`
    - Multi-GPU I2V: `torchrun --nproc_per_node=8 generate.py --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --dit_fsdp --t5_fsdp --ulysses_size 8 --teacache --teacache_thresh 0.08`
    - Single GPU S2V: `python generate.py --task s2v-14B --ckpt_dir ./Wan2.2-S2V-14B --image examples/i2v_input.JPG --audio examples/talk.wav --teacache --teacache_thresh 0.08`
  - TeaCache flags:
    - `--teacache`: enable (disabled by default)
    - `--teacache_thresh`: skip aggressiveness (default 0.08)
    - `--teacache_policy`: rescale policy (`linear` default; unknown values fall back to `linear`)
    - `--teacache_warmup`: force compute first K steps (default 1)
    - `--teacache_last_steps`: force compute last K steps (default 1)
    - `--teacache_alternating`: alternate skip eligibility (default off). Helps stabilize long sequences of reuses.
    - Design doc: see [CFG_CACHE_DESIGN.md](CFG_CACHE_DESIGN.md) for architecture and details.
  - Notes and caveats:
    - State resets per run; cached residuals are cleared before each generation.
    - In SP/Ulysses mode, ranks synchronize a scalar decision each step; ensure ranks are aligned.
    - On shape/dtype mismatch or invalid metrics, compute is forced (failsafe) and accumulator resets.
    - Offload mode moves cached residuals to CPU when models are offloaded to release VRAM (I2V/TI2V/S2V).
    - In separate‑CFG pipelines (Wan2.2), the cond branch increments the executed step; warmup/last‑steps apply to cond; uncond shares the same timestep index.
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
