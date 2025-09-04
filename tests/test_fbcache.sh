#!/usr/bin/env bash
set -euo pipefail

# FBCache E2E smoke tests
# Usage: bash tests/test_fbcache.sh <local model dir> [gpu number]

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <local model dir> [gpu number]"
  exit 1
fi

MODEL_DIR=$(realpath "$1")
GPUS=${2:-1}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY=./generate.py
LOGDIR=${LOGDIR:-./fbcache_logs}
mkdir -p "$LOGDIR"

echo "[FBCache Test] TI2V single-GPU with FBCache"
TI2V_DIR="$MODEL_DIR/Wan2.2-TI2V-5B"
if [ -d "$TI2V_DIR" ]; then
  LOG_TTI="$LOGDIR/ti2v_fbcache.log"
  python "$PY" \
    --task ti2v-5B \
    --size 1280*704 \
    --ckpt_dir "$TI2V_DIR" \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --fbcache --fb_thresh 0.08 \
    --save_file "$LOGDIR/ti2v_fbcache.mp4" | tee "$LOG_TTI"
  if grep -E -q "FBCache\\[|FBCache skips:|CacheManager" "$LOG_TTI"; then
    echo "[OK] TI2V cache telemetry present."
  else
    echo "[WARN] TI2V telemetry not found; generation may still have succeeded."
  fi
else
  echo "[SKIP] TI2V checkpoints not found at $TI2V_DIR"
fi

echo "[FBCache Test] I2V single-GPU with FBCache"
I2V_DIR="$MODEL_DIR/Wan2.2-I2V-A14B"
if [ -d "$I2V_DIR" ]; then
  LOG_I2V="$LOGDIR/i2v_fbcache.log"
  python "$PY" \
    --task i2v-A14B \
    --size 832*480 \
    --ckpt_dir "$I2V_DIR" \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --fbcache --fb_thresh 0.08 \
    --save_file "$LOGDIR/i2v_fbcache.mp4" | tee "$LOG_I2V"
  if grep -E -q "FBCache\\[|FBCache skips:|CacheManager" "$LOG_I2V"; then
    echo "[OK] I2V cache telemetry present."
  else
    echo "[WARN] I2V telemetry not found; generation may still have succeeded."
  fi
else
  echo "[SKIP] I2V checkpoints not found at $I2V_DIR"
fi

echo "[FBCache Test] S2V single-GPU with FBCache"
S2V_DIR="$MODEL_DIR/Wan2.2-S2V-14B"
if [ -d "$S2V_DIR" ]; then
  LOG_S2V="$LOGDIR/s2v_fbcache.log"
  python "$PY" \
    --task s2v-14B \
    --ckpt_dir "$S2V_DIR" \
    --image examples/i2v_input.JPG \
    --audio examples/talk.wav \
    --base_seed 42 \
    --offload_model True \
    --t5_cpu \
    --fbcache --fb_thresh 0.08 \
    --infer_frames 48 \
    --save_file "$LOGDIR/s2v_fbcache.mp4" | tee "$LOG_S2V"
  echo "[OK] S2V run completed with FBCache flag."
else
  echo "[SKIP] S2V checkpoints not found at $S2V_DIR"
fi

echo "[FBCache Test] Done. Logs in $LOGDIR"
