#!/usr/bin/env bash
set -euo pipefail

# Simple walltime benchmark for TeaCache on/off
# Usage: bash benchmarks/bench_teacache.sh <local model dir>

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <local model dir>"
  exit 1
fi

MODEL_DIR=$(realpath "$1")
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=./generate.py
OUTDIR=${OUTDIR:-./bench_outputs}
mkdir -p "$OUTDIR"

function run_case() {
  local name=$1; shift
  local outfile="$OUTDIR/${name}.mp4"
  local logfile="$OUTDIR/${name}.log"
  /usr/bin/time -f "%E real, %U user, %S sys, %M KB" \
    python "$PY" "$@" --save_file "$outfile" 2>&1 | tee "$logfile"
}

echo "[Bench] TI2V on/off (single GPU)"
TI2V_DIR="$MODEL_DIR/Wan2.2-TI2V-5B"
if [ -d "$TI2V_DIR" ]; then
  run_case ti2v_off \
    --task ti2v-5B --size 1280*704 --ckpt_dir "$TI2V_DIR" --base_seed 42 --offload_model True --t5_cpu
  run_case ti2v_on \
    --task ti2v-5B --size 1280*704 --ckpt_dir "$TI2V_DIR" --base_seed 42 --offload_model True --t5_cpu --teacache --teacache_thresh 0.08
else
  echo "[SKIP] TI2V checkpoints not found: $TI2V_DIR"
fi

echo "[Bench] I2V on/off (single GPU)"
I2V_DIR="$MODEL_DIR/Wan2.2-I2V-A14B"
if [ -d "$I2V_DIR" ]; then
  run_case i2v_off \
    --task i2v-A14B --size 832*480 --ckpt_dir "$I2V_DIR" --base_seed 42 --offload_model True --t5_cpu
  run_case i2v_on \
    --task i2v-A14B --size 832*480 --ckpt_dir "$I2V_DIR" --base_seed 42 --offload_model True --t5_cpu --teacache --teacache_thresh 0.08
else
  echo "[SKIP] I2V checkpoints not found: $I2V_DIR"
fi

echo "[Bench] S2V on/off (single GPU)"
S2V_DIR="$MODEL_DIR/Wan2.2-S2V-14B"
if [ -d "$S2V_DIR" ]; then
  run_case s2v_off \
    --task s2v-14B --ckpt_dir "$S2V_DIR" --image examples/i2v_input.JPG --audio examples/talk.wav --base_seed 42 --offload_model True --t5_cpu --infer_frames 48
  run_case s2v_on \
    --task s2v-14B --ckpt_dir "$S2V_DIR" --image examples/i2v_input.JPG --audio examples/talk.wav --base_seed 42 --offload_model True --t5_cpu --infer_frames 48 --teacache --teacache_thresh 0.08
else
  echo "[SKIP] S2V checkpoints not found: $S2V_DIR"
fi

echo "[Bench] Finished. See $OUTDIR for logs and outputs."

