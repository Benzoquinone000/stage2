# Abandoned Experiment Directions

This file keeps short notes for directions that were tried and then removed from
the active workspace. The goal is to avoid repeating low-value branches while
keeping the main reports focused on retained high-score runs.

## Standard BERT-Base From Scratch

- Shape: 12 layers, hidden size 768, 12 heads.
- Observation: clean-data TAPT/classifier variants reached roughly `0.918-0.922`
  test macro F1 in single-model runs, below the retained 512x8 branch.
- Five-fold attempts were stopped early because the first fold was not
  competitive with the retained 512x8 five-fold workflow.
- Decision: do not continue this branch unless there is substantially more MLM
  data or a stronger from-scratch pretraining budget.

## Broad DAPT With Standard BERT-Base

- Observation: bolder broad-domain DAPT learning rates caused MLM loss spikes or
  loss explosion after initially reasonable epochs.
- Safer learning rates stabilized training but did not produce a better
  downstream classifier than the retained 512x8 news-DAPT -> TAPT branch.
- Decision: removed outputs and launch scripts; avoid revisiting this exact
  schedule.

## AG-News-Only MLM / Longer MLM

- Observation: AG-News-only MLM improved with longer training but remained
  weaker than adding external news-domain MLM data before TAPT.
- Downstream classifier results stayed around the low `0.919` macro F1 range.
- Decision: keep external news-domain DAPT as the default pretraining bridge.

## Smaller BERT 384x6

- Shape: 6 layers, hidden size 384, 6 heads, about `22.4M` parameters.
- Observation: the single clean TAPT-more classifier reached test accuracy
  `0.922105` and macro F1 `0.921950`.
- Five-fold probing was weaker: fold 0 macro F1 `0.912995`, fold 1 macro F1
  `0.916946`; the run was stopped during fold 2.
- Decision: not retained as a scoring branch. It is efficient, but the current
  signal suggests lower capacity and less useful ensemble value than DPCNN.

## Non-Retained 512x8 Classifier Variants

- Observation: unregularized or differently staged 512x8 fine-tuning variants
  landed below the retained clean TAPT-more classifier and five-fold ensemble.
- Label smoothing `0.02`, dropout `0.15`, cosine schedule, and `1.5e-4`
  learning rate remained the best retained classifier recipe.
- Decision: keep only the best single 512x8 checkpoint family and its five-fold
  ensemble.

## DPCNN Single/Smoke Runs

- Observation: single DPCNN runs were useful for tuning regularization but were
  weaker than DPCNN five-fold ensembling. The regularized single run was around
  macro F1 `0.920582`; the retained DPCNN five-fold ensemble is `0.926390`.
- Decision: remove single/smoke outputs and keep only the DPCNN five-fold branch
  plus the BERT+DPCNN probability blend.
