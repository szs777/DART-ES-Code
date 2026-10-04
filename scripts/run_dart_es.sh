#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
NUM_ENGINES=${NUM_ENGINES:-8}
RUN_NAME=${RUN_NAME:-outputs/dart-es-qwen2.5-0.5b-seed42}

exec "${PYTHON_BIN}" train_dart_es_gsm8k.py \
  --model_name "${MODEL_NAME}" \
  --data_path gsm8k/gsm8k_main_train.json \
  --cuda_devices "${CUDA_DEVICES}" \
  --num_engines "${NUM_ENGINES}" \
  --population_size 40 \
  --sigma 0.0015 \
  --alpha 0.00025 \
  --global_seed 42 \
  --epochs 40 \
  --chunk_size 256 \
  --replay_ratio 0.20 \
  --pass_ema_decay 0.70 \
  --weight_floor 0.80 \
  --weight_exponent 3.0 \
  --replay_pass_threshold 0.50 \
  --max_replays_per_epoch 2 \
  --replay_cooldown_iterations 1 \
  --zero_pass_patience 2 \
  --experiment_dir "${RUN_NAME}" \
  "$@"
