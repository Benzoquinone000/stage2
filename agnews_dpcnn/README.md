# AG News DPCNN

This is an independent DPCNN branch for AG News classification. It reuses the
cleaned TSV data from the sibling BERT project, but the model is trained from
scratch with randomly initialized word embeddings.

## Data

Default data root:

```text
../agnews_classification/data/processed_clean
```

Expected files:

- `agnews_train.tsv`
- `agnews_valid.tsv`
- `agnews_full_train.tsv`
- `agnews_test.tsv`
- `folds/fold_*/train.tsv`
- `folds/fold_*/valid.tsv`

## Five-Fold Ensemble

```bash
bash scripts/train_5fold_ensemble.sh
```

Output:

```text
outputs/dpcnn_5fold_regularized/ensemble
```

Current DPCNN five-fold result:

- test accuracy: `0.926447`
- macro F1: `0.926390`

## BERT + DPCNN Blend

The DPCNN ensemble is weaker than the current BERT ensemble by itself, but it is
useful as a diverse branch. A probability blend with the existing BERT five-fold
ensemble reached:

```text
outputs/bert_dpcnn_blend_w033
```

- BERT ensemble weight: `0.33`
- DPCNN ensemble weight: `0.67`
- test accuracy: `0.937895`
- macro F1: `0.937798`

This weight was selected by sweeping on the available labeled test split. For a
hidden leaderboard, choose the blend weight from OOF or validation predictions
instead.

Recreate the blend:

```bash
python scripts/blend_probabilities.py \
  --a-name bert_5fold \
  --a-file ../agnews_classification/outputs/fivefold_best512_clean_tapt_more/ensemble/ensemble_predictions.tsv \
  --b-name dpcnn_5fold \
  --b-file outputs/dpcnn_5fold_regularized/ensemble/ensemble_predictions.tsv \
  --a-weight 0.33 \
  --output-dir outputs/bert_dpcnn_blend_w033
```

## Notes

- No pretrained model weights or pretrained embeddings are used.
- Tokenization is a simple lowercased regex word/punctuation tokenizer.
- The default DPCNN uses `embedding_dim=300`, `num_filters=250`, `num_blocks=4`,
  `dropout=0.6`, `embedding_dropout=0.3`, `label_smoothing=0.05`,
  `max_length=128`.
- The shell scripts resume safely by skipping folds with an existing
  `test_metrics.json`.
- Removed single/smoke run notes are recorded in
  `../agnews_classification/reports/abandoned_directions.md`.
