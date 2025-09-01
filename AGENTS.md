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
- Tests (E2E): `bash tests/test.sh <local model dir> <gpu number>`.
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

