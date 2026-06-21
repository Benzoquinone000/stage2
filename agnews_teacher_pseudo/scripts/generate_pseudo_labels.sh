#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -f data/unlabeled_news_pool.tsv ]]; then
  python scripts/prepare_unlabeled_pool.py
fi

CHECKPOINTS=("$@")
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  CHECKPOINTS=(outputs/deberta_v3_large_single/best)
fi

python scripts/predict_teacher.py \
  --checkpoints "${CHECKPOINTS[@]}" \
  --input-file data/unlabeled_news_pool.tsv \
  --output-dir outputs/pseudo_labels \
  --max-length 192 \
  --batch-size 64 \
  --threshold "${PSEUDO_THRESHOLD:-0.98}" \
  --max-per-class "${PSEUDO_MAX_PER_CLASS:-25000}" \
  --device "${DEVICE:-cuda}" \
  --amp
