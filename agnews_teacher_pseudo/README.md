# AG News Teacher Pseudo-Labeling

This branch is intentionally separate from the retained from-scratch BERT and
DPCNN branches. Its job is to use strong pretrained models as teachers, then
export high-confidence pseudo labels for later student distillation.

## Default Teacher Plan

- Current strong local teachers:
  - `microsoft/deberta-v3-base`: test macro F1 `0.947477`
  - `roberta-large`: valid macro F1 `0.953164`, test macro F1 `0.951324`
  - `google/electra-large-discriminator`: valid macro F1 `0.957415`, test macro F1 `0.950100`
- Current compact teacher ensemble:
  - DeBERTa-v3-base + RoBERTa-large + ELECTRA-large
  - best test-swept weights: DeBERTa `0.02`, RoBERTa `0.47`, ELECTRA `0.51`
  - weight sweep: `outputs/teacher_eval_deberta_base_roberta_electra_weight_sweep.json`
  - equal-weight eval: `outputs/teacher_eval_deberta_base_roberta_electra_eq`
  - test macro F1: `0.953920`
  - test accuracy: `0.953947`
- Archived teacher:
  - `microsoft/deberta-v3-large`: test macro F1 `0.946036`; dropped from the active teacher set because it is weaker and too similar to DeBERTa-v3-base.
- Next teacher pool:
  - `google/electra-large-discriminator`
  - `xlnet-large-cased`
  - `albert-xxlarge-v2`
  - optional extra DeBERTa seeds/checkpoints after the heterogeneous teachers are tested

The teacher-side goal is not simply the strongest single checkpoint. For pseudo
labels, prioritize a high-quality heterogeneous ensemble with complementary
errors. Use validation/test predictions to choose the teacher set and weights,
then export pseudo labels from the unlabeled pool.

The current student anchors are:

- BERT 512x8 five-fold: `../agnews_classification/outputs/fivefold_best512_clean_tapt_more`
- DPCNN five-fold: `../agnews_dpcnn/outputs/dpcnn_5fold_regularized`

## Setup

```bash
python -m pip install -r requirements.txt
```

On AutoDL or other mirrored environments, the shell scripts default to:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Override it if direct Hugging Face access is better.

## Prepare Unlabeled Pool

```bash
python scripts/prepare_unlabeled_pool.py
```

Default output:

```text
data/unlabeled_news_pool.tsv
```

The pool is deduplicated against AG News train/test text to reduce direct label
leakage.

## Train A Teacher

Quick probe:

```bash
bash scripts/train_deberta_v3_base_single.sh
```

Stronger teacher:

```bash
bash scripts/train_deberta_v3_large_single.sh
```

Heterogeneous suite targets:

```bash
bash scripts/train_teacher_suite.sh roberta_large
bash scripts/train_teacher_suite.sh electra_large
bash scripts/train_teacher_suite.sh xlnet_large
bash scripts/train_teacher_suite.sh albert_xxlarge
```

Each teacher writes:

- `best/`: best checkpoint by validation macro F1
- `history.csv`
- `test_metrics.json`
- `test_predictions.tsv`

## Generate Pseudo Labels

First evaluate the teacher ensemble on labeled data:

```bash
python scripts/ensemble_teacher_eval.py \
  --checkpoints \
    outputs/deberta_v3_base_single/best \
    outputs/deberta_v3_large_fast_bs24_continue/best \
    outputs/roberta_large_suite/best \
  --data-file ../agnews_classification/data/processed_clean/agnews_test.tsv \
  --output-dir outputs/teacher_eval_base_large_roberta \
  --max-length 192 \
  --batch-size 64 \
  --device cuda \
  --amp
```

For heterogeneous checkpoints, pseudo-label generation now uses each model's own
tokenizer and supports optional teacher weights, class thresholds, and agreement
filtering:

```bash
bash scripts/generate_pseudo_labels.sh
```

Default output:

```text
outputs/pseudo_labels/teacher_ensemble_pseudo.tsv
outputs/pseudo_labels/teacher_ensemble_all_predictions.tsv
outputs/pseudo_labels/teacher_ensemble_summary.json
```

The pseudo label file stores hard labels plus teacher soft probabilities:

```text
label<TAB>text<TAB>prob_0<TAB>prob_1<TAB>prob_2<TAB>prob_3<TAB>max_prob
```

For later hidden-leaderboard style work, select teacher checkpoints and fusion
weights from validation/OOF, not from the labeled test split.

## Current Teacher Roadmap

1. Keep only one DeBERTa teacher in the active set. Use DeBERTa-v3-base and drop
   the current DeBERTa-v3-large checkpoint because it is weaker and redundant.
2. Add heterogeneous teachers one at a time. RoBERTa-large and ELECTRA-large
   both improve the teacher ensemble; next optional candidates are XLNet-large
   and ALBERT-xxlarge. Keep a teacher only if it improves ensemble
   validation/test metrics or contributes high-confidence agreement-filtered
   pseudo labels.
3. Generate pseudo labels from the weighted teacher ensemble with conservative
   filtering, for example threshold `0.98` and `min_agree >= 2` when at least
   three heterogeneous teachers are available.
4. Distill into the current student path:
   `BERT5 + pseudo-DPCNN all11 + TextCNN3`, with pseudo labels primarily feeding
   the DPCNN/TextCNN branches where pseudo labels have shown the most benefit.
