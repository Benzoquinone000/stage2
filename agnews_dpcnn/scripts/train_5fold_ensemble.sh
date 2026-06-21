#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-../agnews_classification/data/processed_clean}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-outputs/dpcnn_5fold_regularized}"

mkdir -p "$OUT_ROOT"

for fold in 0 1 2 3 4; do
  fold_dir="$OUT_ROOT/fold_${fold}"
  if [[ -f "$fold_dir/test_metrics.json" ]]; then
    echo "skip fold ${fold}: existing $fold_dir/test_metrics.json"
    continue
  fi

  python scripts/train_dpcnn.py \
    --train-file "$DATA_ROOT/folds/fold_${fold}/train.tsv" \
    --valid-file "$DATA_ROOT/folds/fold_${fold}/valid.tsv" \
    --test-file "$DATA_ROOT/agnews_test.tsv" \
    --output-dir "$fold_dir" \
    --max-vocab-size 80000 \
    --min-freq 2 \
    --max-length 128 \
    --embedding-dim 300 \
    --num-filters 250 \
    --num-blocks 4 \
    --dropout 0.6 \
    --embedding-dropout 0.3 \
    --label-smoothing 0.05 \
    --batch-size 128 \
    --epochs 20 \
    --learning-rate 5e-4 \
    --weight-decay 5e-4 \
    --max-grad-norm 5.0 \
    --warmup-ratio 0.06 \
    --patience 5 \
    --seed "$((42 + fold))" \
    --device "$DEVICE" \
    --amp
done

python scripts/ensemble_probs.py \
  --prob-files \
    "$OUT_ROOT/fold_0/test_probs.tsv" \
    "$OUT_ROOT/fold_1/test_probs.tsv" \
    "$OUT_ROOT/fold_2/test_probs.tsv" \
    "$OUT_ROOT/fold_3/test_probs.tsv" \
    "$OUT_ROOT/fold_4/test_probs.tsv" \
  --output-dir "$OUT_ROOT/ensemble"
