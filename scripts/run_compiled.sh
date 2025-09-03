#!/usr/bin/env bash
# Run a PyTorch script with torch.compile (Inductor) and CUDA Graphs enabled.
#
# Usage:
#   scripts/run_compiled.sh [options] -- <your_script.py> [script args...]
#
# Notes:
# - This wrapper sets Inductor/Graphs env toggles. Your script should call
#   torch.compile(...) (e.g., model = torch.compile(model, mode=...)).
# - CUDA Graphs require static shapes/allocations; best results after warm‑up.
# - Verify speed/quality trade‑offs for your workload.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
MODE="reduce-overhead"   # torch.compile mode: default optimize for inference
USE_CUDAGRAPHS=1         # request CUDA Graphs in Inductor
CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:128}
LOGS=${TORCH_LOGS:-}

print_help() {
  cat <<EOF
Usage: scripts/run_compiled.sh [options] -- <your_script.py> [script args]

Options:
  --python PATH           Python interpreter (default: python)
  --mode MODE            torch.compile mode (default: ${MODE})
                         e.g., reduce-overhead | max-autotune | default
  --no-cudagraphs        Do not request CUDA Graphs in Inductor
  --connections N        Set CUDA_DEVICE_MAX_CONNECTIONS (default: ${CONNECTIONS})
  --alloc-conf STR       Set PYTORCH_CUDA_ALLOC_CONF (default: ${ALLOC_CONF})
  --logs STR             Set TORCH_LOGS (e.g., +dynamo,+inductor)
  -h, --help             Show this help

Pass your script after "--". Example:
  scripts/run_compiled.sh --mode reduce-overhead -- \
    examples/infer.py --flag foo
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --no-cudagraphs) USE_CUDAGRAPHS=0; shift ;;
    --connections) CONNECTIONS="$2"; shift 2 ;;
    --alloc-conf) ALLOC_CONF="$2"; shift 2 ;;
    --logs) LOGS="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; print_help; exit 2 ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "Error: missing script path. Use -- to separate wrapper and script args." >&2
  print_help
  exit 2
fi

SCRIPT_PATH="$1"; shift || true

# Core environment toggles
export TORCH_COMPILE=1
export TORCH_COMPILE_MODE="${MODE}"

# Request CUDA Graphs inside Inductor (effective when torch.compile is used)
if [[ "${USE_CUDAGRAPHS}" == "1" ]]; then
  export TORCHINDUCTOR_USE_CUDAGRAPHS=1
fi

# Reduce kernel launch contention and allocator fragmentation
export CUDA_DEVICE_MAX_CONNECTIONS="${CONNECTIONS}"
export PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}"

# Optional logging for debugging
if [[ -n "${LOGS}" ]]; then
  export TORCH_LOGS="${LOGS}"
fi

echo "[run_compiled] PYTHON_BIN=${PYTHON_BIN}"
echo "[run_compiled] TORCH_COMPILE_MODE=${TORCH_COMPILE_MODE} TORCHINDUCTOR_USE_CUDAGRAPHS=${TORCHINDUCTOR_USE_CUDAGRAPHS:-0}"
echo "[run_compiled] CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
echo "[run_compiled] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
if [[ -n "${TORCH_LOGS:-}" ]]; then echo "[run_compiled] TORCH_LOGS=${TORCH_LOGS}"; fi
echo "[run_compiled] Running: ${SCRIPT_PATH} $*"

"${PYTHON_BIN}" "${SCRIPT_PATH}" "$@"
