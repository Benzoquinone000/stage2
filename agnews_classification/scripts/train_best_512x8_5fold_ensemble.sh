#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEVICE="${DEVICE:-cuda}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-agnews-bert}"
RUN_PREFIX="${RUN_PREFIX:-best512-clean-tapt-more-5fold}"
OUT_ROOT="${OUT_ROOT:-outputs/fivefold_best512_clean_tapt_more}"
MLM_CHECKPOINT="${MLM_CHECKPOINT:-outputs/bert_mlm_news_augmented_tapt_more_512x8_len128}"

WANDB_BASE=(
  --wandb
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_BASE+=(--wandb-entity "$WANDB_ENTITY")
fi

mkdir -p "$OUT_ROOT"

for fold in 0 1 2 3 4; do
  fold_dir="$OUT_ROOT/fold_${fold}"
  if [[ -f "$fold_dir/test_metrics.json" ]]; then
    echo "skip fold ${fold}: existing $fold_dir/test_metrics.json"
    continue
  fi

  python scripts/finetune_bert_classifier.py \
    --mlm-checkpoint "$MLM_CHECKPOINT" \
    --train-file "data/processed_clean/folds/fold_${fold}/train.tsv" \
    --valid-file "data/processed_clean/folds/fold_${fold}/valid.tsv" \
    --test-file data/processed_clean/agnews_test.tsv \
    --output-dir "$fold_dir" \
    --max-length 128 \
    --batch-size 128 \
    --gradient-accumulation-steps 1 \
    --epochs 8 \
    --learning-rate 1.5e-4 \
    --weight-decay 0.01 \
    --max-grad-norm 1.0 \
    --scheduler cosine \
    --warmup-steps 200 \
    --best-metric valid_macro_f1 \
    --label-smoothing 0.02 \
    --rdrop-alpha 0.0 \
    --layerwise-lr-decay 1.0 \
    --classifier-lr-multiplier 1.0 \
    --dropout-prob 0.15 \
    --early-stopping-patience 2 \
    --seed "$((42 + fold))" \
    --device "$DEVICE" \
    --amp \
    "${WANDB_BASE[@]}" \
    --wandb-run-name "${RUN_PREFIX}-fold${fold}"
done

python scripts/ensemble_bert_classifiers.py \
  --checkpoints \
    "$OUT_ROOT/fold_0" \
    "$OUT_ROOT/fold_1" \
    "$OUT_ROOT/fold_2" \
    "$OUT_ROOT/fold_3" \
    "$OUT_ROOT/fold_4" \
  --test-file data/processed_clean/agnews_test.tsv \
  --output-dir "$OUT_ROOT/ensemble" \
  --max-length 128 \
  --batch-size 256 \
  --device "$DEVICE" \
  --amp

python scripts/summarize_experiments.py
