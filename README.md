# Stage 2 AG News Classification

This repository contains the stage-2 AG News text classification project. The
submitted task chooses AG News instead of CIFAR-100 and focuses on improving
simple classifiers through careful preprocessing, self-pretraining, CNN model
tuning, pseudo labels, and probability-level ensemble analysis.

## Project Structure

```text
agnews_classification/     BERT scratch pretraining and classifier experiments
agnews_dpcnn/              DPCNN/TextCNN experiments and fusion scripts
agnews_teacher_pseudo/     teacher-model pseudo-label generation utilities
mini_transformers/         local Transformer implementation used by BERT branch
reports/                   final task report, figures, tables, and package
requirements.txt           Python dependencies
```

The main report and submission materials are in:

```text
reports/
```

Important files:

- `reports/agnews_final_report_detailed.md`
- `reports/experiment_log.md`
- `reports/tables/final_results.csv`
- `reports/agnews_report_package.zip`

## Setup

```bash
python -m pip install -r requirements.txt
```

Large local artifacts are ignored by Git:

- raw/processed data
- training outputs
- checkpoints
- logs
- W&B local files

This keeps the repository suitable for code/report review while preserving the
run commands and experiment records needed for reproduction.

## Final Result Summary

The final retained three-branch track uses:

- BERT-base scratch5
- DPCNN5-b5
- TextCNN5-wide

Best validation-selected fusion:

| Method | Selection | Test macro-F1 | Test accuracy |
| --- | --- | ---: | ---: |
| BERT + DPCNN | OOF validation | 0.941120 | 0.941184 |
| BERT + DPCNN + TextCNN | OOF validation | 0.941111 | 0.941184 |

The test-sweep upper bound for the three-branch fusion reached macro-F1
`0.942186`, but the report separates that from validation-selected results.

## Reproduction Entry Points

CNN training and probability fusion scripts:

```bash
cd agnews_dpcnn
bash scripts/train_5fold_ensemble.sh
python scripts/ensemble_probs.py --help
python scripts/sweep_blend_probabilities.py --help
```

BERT scratch/self-pretraining scripts:

```bash
cd agnews_classification
python scripts/pretrain_bert_mlm.py --help
python scripts/finetune_bert_classifier.py --help
```

Teacher pseudo-label utilities:

```bash
cd agnews_teacher_pseudo
bash scripts/train_teacher_suite.sh
bash scripts/generate_pseudo_labels.sh
```
