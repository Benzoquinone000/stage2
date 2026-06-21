# AG News CNN Experiments

This directory contains the CNN side of the AG News stage-2 project. The final
report uses AG News rather than CIFAR-100 and compares three retained branches:

- self-pretrained BERT-base scratch 5-fold ensemble
- DPCNN 5-fold ensemble with tuned depth/regularization
- TextCNN 5-fold ensemble with a wider multi-kernel setup

Large datasets, checkpoints, logs, and generated output directories are ignored
by Git. Project-level reports, figures, tables, and the submission archive live
in the repository-level `../reports/` directory.

## Layout

```text
agnews_dpcnn/
  scripts/                 command-line experiment entrypoints
  src/agnews_dpcnn/        reusable data/model/training/metric modules
```

The training scripts are intentionally thin. Shared implementation lives in:

- `src/agnews_dpcnn/data.py`: TSV loading, regex tokenization, vocab building,
  datasets, dataloaders
- `src/agnews_dpcnn/models.py`: DPCNN and TextCNN definitions
- `src/agnews_dpcnn/training.py`: seed control, LR schedules, train/eval loops,
  checkpoint and history writers
- `src/agnews_dpcnn/metrics.py`: accuracy, macro-F1, NLL, probability TSV IO
- `src/agnews_dpcnn/probabilities.py`: averaging and blend-search helpers

## Data

Default data root:

```text
../agnews_classification/data/processed_clean
```

Expected files:

```text
agnews_train.tsv
agnews_valid.tsv
agnews_full_train.tsv
agnews_test.tsv
folds/fold_*/train.tsv
folds/fold_*/valid.tsv
```

Pseudo-label TSV files used by the stronger CNN experiments are stored locally
under the ignored data directory and described in `../reports/experiment_log.md`.

## Run

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Run the default DPCNN five-fold experiment:

```bash
cd agnews_dpcnn
bash scripts/train_5fold_ensemble.sh
```

Run one DPCNN fold manually:

```bash
python scripts/train_dpcnn.py \
  --train-file ../agnews_classification/data/processed_clean/folds/fold_0/train.tsv \
  --valid-file ../agnews_classification/data/processed_clean/folds/fold_0/valid.tsv \
  --test-file ../agnews_classification/data/processed_clean/agnews_test.tsv \
  --output-dir outputs/dpcnn_fold0 \
  --device cuda \
  --amp
```

Run one TextCNN fold manually:

```bash
python scripts/train_textcnn.py \
  --train-file ../agnews_classification/data/processed_clean/folds/fold_0/train.tsv \
  --valid-file ../agnews_classification/data/processed_clean/folds/fold_0/valid.tsv \
  --test-file ../agnews_classification/data/processed_clean/agnews_test.tsv \
  --output-dir outputs/textcnn_fold0 \
  --device cuda \
  --amp
```

Average probability files:

```bash
python scripts/ensemble_probs.py \
  --prob-files outputs/run_a/test_probs.tsv outputs/run_b/test_probs.tsv \
  --output-dir outputs/ensemble_ab
```

Search blend weights:

```bash
python scripts/sweep_blend_probabilities.py \
  --names bert dpcnn textcnn \
  --prob-files bert.tsv dpcnn.tsv textcnn.tsv \
  --step 0.001 \
  --output-dir outputs/blend_search
```

## Final Tracked Results

The final report emphasizes validation/OOF selection, with test sweeps reported
as ablation or upper-bound analysis.

| Method | Selection | Weights | Test macro-F1 | Test accuracy |
| --- | --- | --- | ---: | ---: |
| BERT-base scratch5 | single | - | 0.928794 | 0.928947 |
| DPCNN5-b5 | single | - | 0.930365 | 0.930395 |
| TextCNN5-wide | single | - | 0.924174 | 0.924342 |
| BERT + DPCNN | OOF validation | 0.461 / 0.539 | 0.941120 | 0.941184 |
| BERT + DPCNN + TextCNN | OOF validation | 0.42 / 0.40 / 0.18 | 0.941111 | 0.941184 |
| BERT + DPCNN + TextCNN | test sweep upper bound | 0.408 / 0.580 / 0.012 | 0.942186 | 0.942237 |

Fold-level validation summary:

| Model | Mean best valid macro-F1 | Std |
| --- | ---: | ---: |
| BERT-base scratch5 | 0.926151 | 0.001225 |
| DPCNN5-b5 | 0.922543 | 0.000940 |
| TextCNN5-wide | 0.921901 | 0.001525 |

## Report Materials

- Detailed report: `../reports/agnews_final_report_detailed.md`
- Experiment log: `../reports/experiment_log.md`
- Tables: `../reports/tables/`
- Figures: `../reports/figures/`
- Submission archive: `../reports/agnews_report_package.zip`

The report states which branches used no external pretrained weights and which
experiments used pseudo labels or distillation. Keep new results in the same
tables and append concise notes to `../reports/experiment_log.md`.
