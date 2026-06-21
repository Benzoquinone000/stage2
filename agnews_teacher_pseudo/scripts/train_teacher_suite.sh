#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-agnews-teacher}"
WANDB_MODE="${WANDB_MODE:-online}"

train_teacher() {
  local run_name="$1"
  local model_name="$2"
  local max_length="$3"
  local batch_size="$4"
  local eval_batch_size="$5"
  local grad_accum="$6"
  local epochs="$7"
  local lr="$8"
  local patience="${9:-1}"
  local checkpointing="${10:-true}"
  local dropout="${11:-}"
  local output_dir="outputs/${run_name}"

  if [[ -f "${output_dir}/test_metrics.json" ]]; then
    echo "skip ${run_name}: existing ${output_dir}/test_metrics.json"
    return
  fi

  args=(
    scripts/train_teacher.py
    --model-name "$model_name"
    --output-dir "$output_dir"
    --max-length "$max_length"
    --batch-size "$batch_size"
    --eval-batch-size "$eval_batch_size"
    --gradient-accumulation-steps "$grad_accum"
    --epochs "$epochs"
    --learning-rate "$lr"
    --weight-decay 0.01
    --scheduler linear
    --warmup-ratio 0.06
    --label-smoothing 0.0
    --early-stopping-patience "$patience"
    --seed 42
    --device "$DEVICE"
    --amp
    --wandb
    --wandb-mode "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$run_name"
  )
  if [[ "$checkpointing" == "true" ]]; then
    args+=(--gradient-checkpointing)
  fi
  if [[ -n "$dropout" ]]; then
    args+=(--dropout-prob "$dropout")
  fi
  python "${args[@]}"
}

case "${1:-all}" in
  deberta_base)
    train_teacher deberta_v3_base_suite microsoft/deberta-v3-base 192 32 128 1 4 1.5e-5 1 false
    ;;
  deberta_large)
    train_teacher deberta_v3_large_suite microsoft/deberta-v3-large 192 4 32 8 4 1e-5 1 true
    ;;
  roberta_large)
    train_teacher roberta_large_fast roberta-large 192 32 128 1 3 1e-5 1 false 0.1
    ;;
  electra_large)
    train_teacher electra_large_fast_b48 google/electra-large-discriminator 192 48 192 1 3 1e-5 1 false 0.1
    ;;
  xlnet_large)
    train_teacher xlnet_large_suite xlnet-large-cased 192 8 64 4 3 8e-6 1 true 0.1
    ;;
  albert_xxlarge)
    train_teacher albert_xxlarge_suite albert-xxlarge-v2 192 4 32 8 3 8e-6 1 true 0.1
    ;;
  all)
    "$0" deberta_base
    "$0" deberta_large
    "$0" roberta_large
    "$0" electra_large
    "$0" xlnet_large
    "$0" albert_xxlarge
    ;;
  *)
    echo "unknown suite target: $1" >&2
    echo "valid targets: deberta_base, deberta_large, roberta_large, electra_large, xlnet_large, albert_xxlarge, all" >&2
    exit 2
    ;;
esac
