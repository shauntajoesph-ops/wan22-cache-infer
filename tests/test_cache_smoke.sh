#!/usr/bin/env bash
set -euo pipefail

# Cache smoke tests covering baseline, TeaCache, and FBCache.
# Usage: bash tests/test_cache_smoke.sh <local model dir> [gpu number]

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <local model dir> [gpu number]"
  exit 1
fi

MODEL_DIR=$(realpath "$1")
GPUS=${2:-1}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=./generate.py
LOGDIR=${LOGDIR:-./cache_smoke_logs}
mkdir -p "$LOGDIR"

echo "[Baseline] TI2V baseline (no cache)"
TI2V_DIR="$MODEL_DIR/Wan2.2-TI2V-5B"
if [ -d "$TI2V_DIR" ]; then
  BASE_TTI="$LOGDIR/ti2v_baseline.log"
  python "$PY" \
    --task ti2v-5B \
    --size 1280*704 \
    --ckpt_dir "$TI2V_DIR" \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --save_file "$LOGDIR/ti2v_baseline.mp4" | tee "$BASE_TTI"
else
  echo "[SKIP] TI2V checkpoints not found at $TI2V_DIR"
fi

echo "[Baseline] I2V baseline (no cache)"
I2V_DIR="$MODEL_DIR/Wan2.2-I2V-A14B"
if [ -d "$I2V_DIR" ]; then
  BASE_I2V="$LOGDIR/i2v_baseline.log"
  python "$PY" \
    --task i2v-A14B \
    --size 832*480 \
    --ckpt_dir "$I2V_DIR" \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --save_file "$LOGDIR/i2v_baseline.mp4" | tee "$BASE_I2V"
else
  echo "[SKIP] I2V checkpoints not found at $I2V_DIR"
fi

echo "[TeaCache] TI2V with TeaCache"
if [ -d "$TI2V_DIR" ]; then
  LOG_TTI="$LOGDIR/ti2v_teacache.log"
  python "$PY" \
    --task ti2v-5B \
    --size 1280*704 \
    --ckpt_dir "$TI2V_DIR" \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --teacache --teacache_thresh 0.08 \
    --save_file "$LOGDIR/ti2v_teacache.mp4" | tee "$LOG_TTI"
fi

echo "[FBCache] TI2V with FBCache"
if [ -d "$TI2V_DIR" ]; then
  LOG_TTIF="$LOGDIR/ti2v_fbcache.log"
  python "$PY" \
    --task ti2v-5B \
    --size 1280*704 \
    --ckpt_dir "$TI2V_DIR" \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --fbcache --fb_thresh 0.08 \
    --save_file "$LOGDIR/ti2v_fbcache.mp4" | tee "$LOG_TTIF"
fi

echo "[TeaCache] I2V with TeaCache"
if [ -d "$I2V_DIR" ]; then
  LOG_I2VT="$LOGDIR/i2v_teacache.log"
  python "$PY" \
    --task i2v-A14B \
    --size 832*480 \
    --ckpt_dir "$I2V_DIR" \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --teacache --teacache_thresh 0.08 \
    --save_file "$LOGDIR/i2v_teacache.mp4" | tee "$LOG_I2VT"
fi

echo "[FBCache] I2V with FBCache"
if [ -d "$I2V_DIR" ]; then
  LOG_I2VF="$LOGDIR/i2v_fbcache.log"
  python "$PY" \
    --task i2v-A14B \
    --size 832*480 \
    --ckpt_dir "$I2V_DIR" \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --fbcache --fb_thresh 0.08 \
    --save_file "$LOGDIR/i2v_fbcache.mp4" | tee "$LOG_I2VF"
fi

echo "[Done] Smoke tests complete; logs in $LOGDIR"

