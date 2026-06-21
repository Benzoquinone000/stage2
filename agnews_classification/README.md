# AG News Classification

This directory contains the Task2 AG News pipeline. The reusable BERT/model
implementation lives in the sibling `mini_transformers` package; this project
contains data preparation, task-specific training loops, experiment scripts, and
branch-local experiment logs.

## Layout

```text
agnews_classification/
  configs/                  BERT configuration files
  scripts/                  data, pretraining, fine-tuning, and ensemble CLIs
  src/agnews_classification/ shared task utilities and text cleaning helpers
  reports/                  BERT-branch experiment logs
```

## Current Best

The current best run is the five-fold probability ensemble:

```text
outputs/fivefold_best512_clean_tapt_more/ensemble
```

Metrics on `data/processed_clean/agnews_test.tsv`:

- test accuracy: `0.928289`
- macro F1: `0.928108`

The ensemble uses the 512-hidden, 8-layer, 8-head BERT-style checkpoint family:

```text
outputs/bert_mlm_news_augmented_tapt_more_512x8_len128
```

## Data

Download the original AG News data:

```bash
python scripts/download_agnews.py
```

Prepare cleaned splits and five folds:

```bash
python scripts/prepare_clean_agnews.py
```

Prepare the augmented news-domain MLM corpus:

```bash
python scripts/prepare_mlm_corpus.py
```

The processed TSV format is:

```text
label<TAB>text
```

Labels are zero-indexed:

- `0`: World
- `1`: Sports
- `2`: Business
- `3`: Sci/Tech

## Training

Generic MLM pretraining entry point:

```bash
python scripts/pretrain_bert_mlm.py --help
```

Generic AG News classifier fine-tuning entry point:

```bash
python scripts/finetune_bert_classifier.py --help
```

Run the current best five-fold ensemble workflow:

```bash
bash scripts/train_best_512x8_5fold_ensemble.sh
```

The script resumes safely: folds with an existing `test_metrics.json` are
skipped, and the final ensemble is written to:

```text
outputs/fivefold_best512_clean_tapt_more/ensemble
```

## Inference

Predict a single text with a fine-tuned checkpoint:

```bash
python scripts/predict_bert_classifier.py \
  --checkpoint outputs/fivefold_best512_clean_tapt_more/fold_0 \
  --text "Apple shares rise after the company reports strong quarterly profit."
```

## Reporting

Regenerate the experiment report:

```bash
python scripts/summarize_experiments.py
```

The BERT-branch experiment log is:

```text
reports/experiment_log.md
```

Short notes for removed or abandoned branches are kept in:

```text
reports/abandoned_directions.md
```

The final task report and submission package live at the repository root:

```text
../reports/
```

## What Was Removed

Obsolete launch scripts for failed or abandoned experiment branches were removed
from `scripts/`. The remaining scripts are either generic utilities or part of
the current best 512x8 five-fold workflow.
