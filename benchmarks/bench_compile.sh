#!/usr/bin/env bash
set -euo pipefail

# Simple benchmark harness for torch.compile and CUDA Graphs presets.
# Usage:
#   bash benchmarks/bench_compile.sh <task> <ckpt_dir> [--size 1280*704] [--compile reduce-overhead|max-autotune|none] [--cuda_graphs]

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <task> <ckpt_dir> [--size 1280*704] [--compile reduce-overhead|max-autotune|none] [--cuda_graphs]"
  exit 1
fi

TASK="$1"; shift
CKPT_DIR="$1"; shift

SIZE="1280*704"
COMPILE="none"
CUDA_GRAPHS="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)
      SIZE="$2"; shift; shift ;;
    --compile)
      COMPILE="$2"; shift; shift ;;
    --cuda_graphs)
      CUDA_GRAPHS="true"; shift ;;
    *)
      echo "Unknown arg: $1"; exit 2 ;;
  esac
done

echo "[bench] task=$TASK size=$SIZE compile=$COMPILE cuda_graphs=$CUDA_GRAPHS"

PY=./generate.py

START_TS=$(date +%s)
if [ "$TASK" = "i2v-A14B" ]; then
  python "$PY" --task "$TASK" --size "$SIZE" --ckpt_dir "$CKPT_DIR" \
    --image examples/i2v_input.JPG \
    --compile "$COMPILE" $( [ "$CUDA_GRAPHS" = "true" ] && echo "--cuda_graphs" ) \
    --base_seed 42 --offload_model False --t5_cpu --save_file /tmp/bench_i2v.mp4
elif [ "$TASK" = "ti2v-5B" ]; then
  python "$PY" --task "$TASK" --size "$SIZE" --ckpt_dir "$CKPT_DIR" \
    --prompt "Two cats boxing on a stage." \
    --compile "$COMPILE" $( [ "$CUDA_GRAPHS" = "true" ] && echo "--cuda_graphs" ) \
    --base_seed 42 --offload_model False --t5_cpu --save_file /tmp/bench_ti2v.mp4
else
  echo "Unsupported task for bench: $TASK"; exit 3
fi
END_TS=$(date +%s)

echo "[bench] elapsed $((END_TS-START_TS)) sec"

