#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python scripts/train_teacher.py \
  --model-name microsoft/deberta-v3-large \
  --output-dir outputs/deberta_v3_large_single \
  --max-length 192 \
  --batch-size 4 \
  --eval-batch-size 32 \
  --gradient-accumulation-steps 8 \
  --epochs 4 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.06 \
  --label-smoothing 0.0 \
  --gradient-checkpointing \
  --seed 42 \
  --device "${DEVICE:-cuda}" \
  --amp \
  --wandb \
  --wandb-project "${WANDB_PROJECT:-agnews-teacher}" \
  --wandb-run-name "${WANDB_RUN_NAME:-deberta-v3-large-single}"
