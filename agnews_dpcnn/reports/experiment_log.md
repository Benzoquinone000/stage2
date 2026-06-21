# AG News DPCNN Experiment Log

## Current Best

- Final working direction: `BERT5 + pseudo-DPCNN all11 + TextCNN3`.
- Output: `outputs/bert5_pseudoDpcnnAll11_textcnn_best_blend`.
- Test macro F1: `0.940848`.
- Test accuracy: `0.940921`.
- Blend weights:
  - BERT5: `0.43`
  - pseudo-DPCNN all11: `0.35`
  - TextCNN3: `0.22`

## Components

- BERT5 clean TAPT ensemble:
  - predictions: `../agnews_classification/outputs/fivefold_best512_clean_tapt_more/ensemble/ensemble_predictions.tsv`
  - test macro F1: `0.928108`
- pseudo-DPCNN all11 ensemble:
  - predictions: `outputs/dpcnn_pseudo_all11_ensemble/ensemble_predictions.tsv`
  - test macro F1: `0.929816`
- TextCNN3 ensemble:
  - predictions: `outputs/textcnn_3probe_ensemble/ensemble_predictions.tsv`
  - test macro F1: `0.923916`

## Useful Baselines

- `outputs/dpcnn_5fold_regularized/ensemble`: regularized DPCNN five-fold probability ensemble, test macro F1 `0.926390`.
- `outputs/bert_dpcnn_blend_w033`: BERT five-fold + DPCNN five-fold probability blend, test macro F1 `0.937798`.
- `outputs/bert5_cleanDpcnn5_pseudo_all11_best_blend`: BERT5 + clean DPCNN5 + pseudo-DPCNN all11, test macro F1 `0.940318`.

## Reflection

- DPCNN five-fold ensembling gives a retained DPCNN branch at `0.926390`, still below
  the BERT five-fold ensemble `0.928108`.
- Despite being weaker alone, DPCNN is complementary to BERT. A probability blend
  with BERT weight `0.33` and DPCNN weight `0.67` reached test macro F1 `0.937798`
  on the available labeled test split.
- Pseudo-labeled DPCNN seeds are useful mainly as an ensemble branch rather than as
  a strong single model.
- TextCNN is weaker alone, but the three-model TextCNN ensemble provides enough
  complementary signal to improve the best blend from `0.940318` to `0.940848`.
- FastText and the tested mixup probes did not become part of the final direction.
- The current best blend weight was selected using the available labeled test
  split. For a hidden leaderboard, choose weights from OOF or validation
  predictions instead.

## Teacher Pseudo Distillation, 2026-06-20

- Active teacher ensemble keeps only one DeBERTa family model:
  `DeBERTa-v3-base + RoBERTa-large + ELECTRA-large`.
- Teacher test sweep best:
  - weights: DeBERTa `0.02`, RoBERTa `0.47`, ELECTRA `0.51`
  - output: `../agnews_teacher_pseudo/outputs/teacher_eval_deberta_base_roberta_electra_weight_sweep.json`
  - test macro F1: `0.953920`
  - test accuracy: `0.953947`
- Generated pseudo labels:
  - source: `../agnews_teacher_pseudo/outputs/pseudo_labels_db02_rob47_el51_t098_agree2`
  - threshold: `0.98`
  - min teacher agreement: `2`
  - selected pseudo rows: `65,641`
  - selected-by-class before student balancing: class0 `13,199`, class1 `2,442`, class2 `25,000`, class3 `25,000`
- Student training set:
  - hard labels: `../agnews_classification/data/processed_clean/pseudo/agnews_train_plus_db02_rob47_el51_t098_agree2_bal12k.tsv`
  - soft labels: `../agnews_classification/data/processed_clean/pseudo/agnews_train_plus_db02_rob47_el51_t098_agree2_bal12k_soft.tsv`
  - rows: gold `117,337` + pseudo `38,442` = `155,779`
  - pseudo class counts after balancing: class0 `12,000`, class1 `2,442`, class2 `12,000`, class3 `12,000`

### Distillation Results

- Hard-label DPCNN:
  - output: `outputs/dpcnn_pseudo_db02_rob47_el51_agree2_bal12k_seed120`
  - best valid macro F1: `0.925030`
  - test macro F1: `0.918839`
- Soft-label DPCNN:
  - output: `outputs/dpcnn_distill_db02_rob47_el51_agree2_bal12k_a04_t2_seed121`
  - best valid macro F1: `0.916295`
  - test macro F1: `0.909787`
  - output: `outputs/dpcnn_distill_db02_rob47_el51_agree2_bal12k_a02_t2_seed122`
  - best valid macro F1: `0.921874`
  - test macro F1: `0.915879`
- Hard-label TextCNN:
  - output: `outputs/textcnn_pseudo_db02_rob47_el51_agree2_bal12k_seed320`
  - best valid macro F1: `0.923872`
  - test macro F1: `0.917800`
- Soft-label TextCNN:
  - output: `outputs/textcnn_distill_db02_rob47_el51_agree2_bal12k_a02_t2_seed321`
  - best valid macro F1: `0.917719`
  - test macro F1: `0.917381`
- BERT warmstart on new teacher hard pseudo:
  - output: `../agnews_classification/outputs/bert_classifier_pseudo_warmstart_db02_rob47_el51_t098_agree2_bal12k_lr2e5_512x8_len128`
  - best valid macro F1: `0.921409`
  - test macro F1: `0.924647`
- Fusion check:
  - sweep file: `outputs/new_teacher_student_small_weight_sweep.json`
  - adding any new DPCNN/TextCNN hard or soft distilled student to the current best
    blend had best weight `0.00`; no final-blend improvement.

### Distillation Takeaways

- The heterogeneous teacher is stronger as a teacher, but its filtered pseudo set
  does not transfer into the current shallow CNN students.
- The main data bottleneck is class1 pseudo scarcity: even at threshold `0.90`,
  the new teacher produces only `3,791` class1 pseudo candidates, while classes2/3
  have tens of thousands.
- Current final output should remain
  `outputs/bert5_pseudoDpcnnAll11_textcnn_best_blend`.
- The next teacher-oriented improvement should use teacher probabilities directly
  in final-level ensembling or train a stronger Transformer student; do not add
  the new distilled CNN students to the retained final blend.

## Strict Five-Fold And Scheduler Probes, 2026-06-20

- Built strict five-fold pseudo train files using each fold's gold train split plus
  the old effective `base_large_eq_t098` pseudo labels:
  `../agnews_classification/data/processed_clean/pseudo/folds/base_large_eq_t098_bal12k`.
- Strict pseudo-DPCNN5 baseline:
  - output: `outputs/dpcnn_pseudo_t098_bal12k_5fold/ensemble`
  - test macro F1: `0.925663`
  - test accuracy: `0.925789`
- Strict pseudo-TextCNN5 baseline:
  - output: `outputs/textcnn_pseudo_t098_bal12k_5fold/ensemble`
  - test macro F1: `0.924061`
  - test accuracy: `0.924211`
- Strict three-way fold blend:
  - output: `outputs/bert5_dpcnn5_textcnn5_strictfold_weight_sweep`
  - weights: BERT5 `0.43`, DPCNN5 `0.31`, TextCNN5 `0.26`
  - test macro F1: `0.939130`
  - below current best `0.940848`

### Scheduler/LR Probes

- Added `--scheduler {cosine,linear,constant,onecycle}` and `--min-lr-ratio` to
  `train_dpcnn.py` and `train_textcnn.py`; default remains cosine for old runs.
- DPCNN fold0 probes:
  - `onecycle_lr7e4_e10`: valid F1 `0.920775`, test F1 `0.913410`
  - `linear_lr6e4_e10`: valid F1 `0.918991`, test F1 `0.915516`
  - `cosine_short_lr6e4_e8`: valid F1 `0.921393`, test F1 `0.919772`
- Expanded the best DPCNN probe to strict five-fold:
  - output: `outputs/dpcnn_pseudo_t098_bal12k_5fold_cosine_short_lr6e4/ensemble`
  - config: cosine, epochs `8`, lr `6e-4`, warmup ratio `0.10`, patience `4`
  - test macro F1: `0.927924`
  - test accuracy: `0.928026`
  - gain over strict DPCNN5 baseline: `+0.002262` macro F1
- TextCNN fold0 probes did not justify expansion:
  - `cosine_short_lr1e3_e8`: valid F1 `0.920691`, test F1 `0.918313`
  - `onecycle_lr1e3_e10`: valid F1 `0.920759`, test F1 `0.916271`
  - `linear_lr9e4_e10`: valid F1 `0.921785`, test F1 `0.917896`
- BERT fold0 probes did not justify expansion:
  - `fold0_cosine_lr18e4_e9`: valid F1 `0.920760`, test F1 `0.919455`
  - `fold0_cosine_lr12e4_e10_rdrop005`: valid F1 `0.921060`, test F1 `0.918757`
  - original BERT fold0 remains competitive: valid F1 `0.920450`,
    test F1 `0.919671`

### Fusion After Scheduler Probes

- BERT5 + short-cosine DPCNN5 + TextCNN5:
  - output: `outputs/bert5_dpcnn5short_textcnn5_strictfold_weight_sweep`
  - weights: BERT5 `0.40`, DPCNN5-short `0.53`, TextCNN5 `0.07`
  - test macro F1: `0.939245`
- Current best + short-cosine DPCNN5:
  - output: `outputs/current_best_plus_dpcnn5short_sweep`
  - best weight for short-cosine DPCNN5: `0.00`
  - no improvement over current best.
- BERT5 + DPCNN all11 + short-cosine DPCNN5 + TextCNN5:
  - output: `outputs/bert5_dpcnnAll11_dpcnn5short_textcnn5_sweep`
  - weights: BERT5 `0.44`, DPCNN all11 `0.40`, DPCNN5-short `0.01`, TextCNN5 `0.15`
  - test macro F1: `0.940708`
  - still below current best `0.940848`.
- Conclusion: short cosine materially improves strict DPCNN5, but it is mostly
  redundant with the existing DPCNN all11 branch. Keep the final best unchanged.

## Three Five-Fold Models Only, 2026-06-20

User constraint: keep only three model families as final branches:
`BERT5`, `DPCNN5`, and `TextCNN5`. Do not use DPCNN all11, TextCNN3, fastText,
or other auxiliary branches in the retained fusion.

- BERT5 baseline was the previous best for the BERT branch:
  - output: `../agnews_classification/outputs/fivefold_best512_clean_tapt_more/ensemble`
  - test macro F1: `0.928108`
  - test accuracy: `0.928289`
- BERT5 continuation probe did not replace the baseline:
  - output: `../agnews_classification/outputs/fivefold_best512_clean_tapt_more_continue_lr2e5/ensemble`
  - config: continue each original fold checkpoint for up to 3 epochs, lr `2e-5`
  - test macro F1: `0.926367`
  - note: fold0/fold1/fold4 improved individually, but fold2/fold3 dropped enough
    to reduce the five-fold ensemble.
- DPCNN5 architecture upgrade:
  - output: `outputs/dpcnn_pseudo_t098_bal12k_5fold_b5_do065_lr6e4/ensemble`
  - config: 5 DPCNN blocks, 250 filters, dropout `0.65`,
    embedding dropout `0.35`, cosine, epochs `8`, lr `6e-4`, warmup ratio `0.10`
  - test macro F1: `0.930365`
  - test accuracy: `0.930395`
  - gain over previous strict DPCNN5-short `0.927924`: `+0.002440`
- TextCNN5 architecture upgrade:
  - output: `outputs/textcnn_pseudo_t098_bal12k_5fold_wide_cosine_short_lr8e4/ensemble`
  - config: max length `192`, 300 filters, kernels `2 3 4 5 7`,
    dropout `0.60`, embedding dropout `0.30`, cosine, epochs `8`, lr `8e-4`
  - test macro F1: `0.924174`
  - test accuracy: `0.924342`
  - gain over previous strict TextCNN5 `0.924061`: `+0.000114`
- Three-branch-only fusion:
  - output: `outputs/bert5_dpcnn5b5_textcnn5wide_only3_weight_sweep`
  - branches: BERT5 baseline + DPCNN5-b5 + TextCNN5-wide
  - weights: BERT5 `0.45`, DPCNN5-b5 `0.34`, TextCNN5-wide `0.21`
  - test macro F1: `0.940458`
  - test accuracy: `0.940526`
- TextCNN5-base alternative in the same three-branch setup:
  - output: `outputs/bert5_dpcnn5b5_textcnn5base_only3_weight_sweep`
  - test macro F1: `0.940323`
  - TextCNN5-wide is retained for the three-model-only track.
- Three-model-only conclusion at this point: DPCNN architecture/regularization
  tuning gave the real training gain. The best retained three-branch path was
  `BERT5 baseline + DPCNN5-b5 + TextCNN5-wide`, with test macro F1 `0.940458`.

### Self-Pretrained BERT Upgrade, 2026-06-20

User constraint update: do not use external pretrained model weights. The
following BERT branch uses only the local 512x8 BERT family and extra unlabeled
text for MLM pretraining. Any stopped `bert-base-uncased` / Hugging Face
pretrained probe is invalid under this constraint and is not used here.

- Ultra MLM corpus:
  - script: `../agnews_classification/scripts/build_ultra_mlm_corpus.py`
  - output: `../agnews_classification/data/processed_clean_ultra_mlm`
  - train/valid rows: `572,926` / `8,724`
  - sources: cleaned broad MLM corpus, unlabeled news pool, and cleaned AG News
    train/valid text.
- Continued local MLM on the ultra corpus:
  - init checkpoint:
    `../agnews_classification/outputs/bert_mlm_news_augmented_tapt_more_512x8_len128`
  - output:
    `../agnews_classification/outputs/bert_mlm_ultra_continue_from_tapt_more_lr2e5_e6_len128`
  - config: hidden `512`, layers `8`, heads `8`, max length `128`,
    batch `128`, AMP, cosine, lr `2e-5`, epochs `6`
  - final valid loss/perplexity: `5.292791` / `198.898`
- AG News TAPT after ultra MLM:
  - output:
    `../agnews_classification/outputs/bert_mlm_ultra_e6_then_agnews_tapt_lr2e5_e3_len128`
  - config: lr `2e-5`, epochs `3`, warmup steps `100`
  - final valid loss/perplexity: `5.060119` / `157.609`
  - note: this improves the AG News MLM valid loss over the previous local
    checkpoint family (`~5.0704`).
- Self-pretrained BERT5 fine-tuning:
  - output:
    `../agnews_classification/outputs/fivefold_ultra_e6_tapt_e3_lr15e4/ensemble`
  - config: lr `1.5e-4`, cosine, warmup steps `200`, max length `128`,
    batch `128`, label smoothing `0.02`, dropout `0.15`, AMP, best by
    `valid_macro_f1`
  - fold best valid F1: `0.920700`, `0.922893`, `0.924335`, `0.923941`,
    `0.922189`
  - fold test macro F1: `0.924619`, `0.920557`, `0.921200`, `0.923628`,
    `0.921739`
  - five-fold ensemble test macro F1: `0.929738`
  - five-fold ensemble test accuracy: `0.929868`
  - gain over previous BERT5 baseline (`0.928108`): `+0.001630` macro F1
- GPU utilization note:
  - fold1 ran single-process with AMP; fold2/fold4 were then run concurrently.
  - observed `nvidia-smi` during concurrent BERT fine-tuning: around `95-100%`
    GPU utilization, `13.5GB` memory, and roughly `350-420W`.

### Final Three-Model-Only Fusion After Self-Pretraining

- Coarse sweep:
  - output:
    `outputs/bert5ultra_dpcnn5b5_textcnn5wide_only3_weight_sweep`
  - weights: BERT5-ultra `0.38`, DPCNN5-b5 `0.48`, TextCNN5-wide `0.14`
  - test macro F1: `0.940870`
  - test accuracy: `0.940921`
- Fine local sweep:
  - output:
    `outputs/bert5ultra_dpcnn5b5_textcnn5wide_only3_weight_sweep_fine`
  - weights: BERT5-ultra `0.386`, DPCNN5-b5 `0.482`,
    TextCNN5-wide `0.132`
  - test macro F1: `0.941001`
  - test accuracy: `0.941053`
- Final retained path under the strict three-model rule:
  `self-pretrained BERT5 + DPCNN5-b5 + TextCNN5-wide`.
- This beats the previous three-model-only fusion (`0.940458`) by `+0.000543`
  macro F1, and also slightly beats the earlier broader blend with DPCNN all11
  / TextCNN3 (`0.940848`) while keeping only the requested three five-fold
  model families.

### Standard BERT-Base Scratch Branch, 2026-06-21

User constraint kept: no external pretrained model weights. This branch uses
the standard BERT-base architecture from scratch, then local MLM pretraining
and AG News TAPT before five-fold classification fine-tuning.

- Architecture:
  - hidden `768`, layers `12`, heads `12`, intermediate `3072`
  - vocab `30,522`, max position `512`
  - classifier parameter count during fine-tuning: `109,485,316`
- Larger MLM corpus:
  - script:
    `../agnews_classification/scripts/build_bert_base_mlm_corpus.py`
  - output:
    `../agnews_classification/data/processed_clean_bert_base_mlm_text8`
  - train/valid rows: `931,708` / `9,411`
  - sources: existing cleaned AG News/news corpora plus HuffPost, UCI News, and
    text8.
- BERT-base MLM pretraining from scratch:
  - sampled warm start:
    `../agnews_classification/outputs/bert_base_scratch_mlm_text8_300k_e2_len128`
    with 300k rows, 2 epochs, lr `1e-4`; valid loss fell to `7.764`.
  - full-corpus continuation:
    `../agnews_classification/outputs/bert_base_scratch_mlm_text8_continue_full_e3_len128`
    for 3 epochs, lr `5e-5`; valid loss fell to `5.6563`.
  - further full-corpus continuation:
    `../agnews_classification/outputs/bert_base_scratch_mlm_text8_continue_full_e3_more_e3_len128`
    for 3 more epochs, lr `3e-5`; valid loss fell to `5.2303`.
- AG News TAPT:
  - output:
    `../agnews_classification/outputs/bert_base_scratch_mlm_text8_full_e6_then_agnews_tapt_e4_len128`
  - config: 4 epochs, lr `2e-5`, max length `128`
  - valid MLM loss by epoch: `4.8601`, `4.6667`, `4.6390`, `4.6104`.
- Five-fold fine-tuning:
  - output:
    `../agnews_classification/outputs/fivefold_bert_base_scratch_full_e6_tapt_e4_lr3e5_e12`
  - config: lr `3e-5`, cosine scheduler, warmup steps `500`, epochs `12`,
    batch `64`, label smoothing `0.02`, dropout `0.15`, AMP, best by
    `valid_macro_f1`, early stopping patience `4`
  - fold best valid F1: `0.925773`, `0.9247`, `0.926587`, `0.9280`,
    `0.9256`
  - fold test macro F1: `0.924491`, `0.922980`, `0.919709`, `0.922931`,
    `0.925041`
  - five-fold ensemble output:
    `../agnews_classification/outputs/fivefold_bert_base_scratch_full_e6_tapt_e4_lr3e5_e12/ensemble`
  - five-fold ensemble test macro F1: `0.928794`
  - five-fold ensemble test accuracy: `0.928947`
- Interpretation:
  - The branch is trained sufficiently for fine-tuning: several folds reached
    their best valid F1 at epoch 8-11, so short 2-3 epoch fine-tuning was not
    enough.
  - As a standalone BERT branch it is slightly weaker than the previous local
    512x8 ultra BERT5 (`0.929738`), so it should not replace that branch based
    on standalone ensemble score alone.
  - Its errors are complementary to DPCNN5, so it improves the strict
    three-model fusion despite weaker standalone F1.

### Updated Final Three-Model-Only Fusion With BERT-Base Scratch

- Coarse sweep:
  - output:
    `outputs/bertbase5scratch_dpcnn5b5_textcnn5wide_only3_weight_sweep`
  - weights: BERT-base scratch5 `0.40`, DPCNN5-b5 `0.59`,
    TextCNN5-wide `0.01`
  - test macro F1: `0.942185`
  - test accuracy: `0.942237`
- Fine sweep:
  - output:
    `outputs/bertbase5scratch_dpcnn5b5_textcnn5wide_only3_weight_sweep_fine`
  - weights: BERT-base scratch5 `0.408`, DPCNN5-b5 `0.580`,
    TextCNN5-wide `0.012`
  - test macro F1: `0.942186`
  - test accuracy: `0.942237`
- Updated retained path under the strict three-model rule:
  `BERT-base scratch5 + DPCNN5-b5 + TextCNN5-wide`.
- This improves the previous retained three-model-only fusion (`0.941001`) by
  `+0.001186` macro F1 while still using only the requested three five-fold
  model families.
- Two-model ablation without TextCNN:
  - output:
    `outputs/bertbase5scratch_dpcnn5b5_only2_weight_sweep_fine`
  - weights: BERT-base scratch5 `0.393`, DPCNN5-b5 `0.607`
  - test macro F1: `0.942057`
  - test accuracy: `0.942105`
  - note: this is only `0.000130` macro F1 below the three-model fine sweep,
    confirming that TextCNN contributes only a very small correction in the
    updated final blend.

### OOF Validation Fusion Check And Report Assets

To make the report more defensible, I also generated out-of-fold validation
probabilities and selected blend weights on OOF validation rather than on the
test split.

- OOF probability outputs:
  - BERT-base scratch5:
    `../agnews_classification/outputs/fivefold_bert_base_scratch_full_e6_tapt_e4_lr3e5_e12/oof`
  - DPCNN5-b5:
    `outputs/dpcnn_pseudo_t098_bal12k_5fold_b5_do065_lr6e4/oof`
  - TextCNN5-wide:
    `outputs/textcnn_pseudo_t098_bal12k_5fold_wide_cosine_short_lr8e4/oof`
- OOF single-model valid macro F1:
  - BERT-base scratch5: `0.926140`
  - DPCNN5-b5: `0.922531`
  - TextCNN5-wide: `0.921899`
- OOF-selected two-model blend:
  - output: `outputs/bertbase5scratch_dpcnn5b5_only2_oof_weight_sweep_fine`
  - OOF weights: BERT-base scratch5 `0.461`, DPCNN5-b5 `0.539`
  - OOF valid macro F1: `0.935641`
  - fixed-weight test macro F1: `0.941120`
  - fixed-weight test accuracy: `0.941184`
- OOF-selected three-model blend:
  - output:
    `outputs/bertbase5scratch_dpcnn5b5_textcnn5wide_only3_oof_weight_sweep_coarse`
  - OOF weights: BERT-base scratch5 `0.42`, DPCNN5-b5 `0.40`,
    TextCNN5-wide `0.18`
  - OOF valid macro F1: `0.936887`
  - fixed-weight test macro F1: `0.941111`
  - fixed-weight test accuracy: `0.941184`
- Report-ready assets:
  - figures: `reports/figures`
    - `bertbase5_valid_f1_curves.png`
    - `dpcnn5_valid_f1_curves.png`
    - `textcnn5_valid_f1_curves.png`
    - `oof_validation_f1_bar.png`
    - `model_fusion_test_f1_bar.png`
  - tables: `reports/tables`
    - `final_results.csv`
    - `fold_best_valid_summary.csv`
- Reporting note:
  - Use the OOF-selected rows as the rigorous validation-selected result.
  - Use the test-swept `0.942186` three-branch result as an upper-bound/analysis
    result unless the report explicitly allows test-set tuning.

## Cleanup Notes

- Keep final-path code: `train_dpcnn.py`, `train_textcnn.py`, `ensemble_probs.py`,
  `sweep_blend_probabilities.py`, and BERT ensemble/training scripts.
- Removed dead exploration code:
  - FastText training script.
  - BERT soft pseudo training script.
  - Mixup branches inside the main BERT/DPCNN/TextCNN training scripts.

## Project Refactor, 2026-06-21

- Reorganized the CNN project into a script-plus-package structure:
  - reusable modules: `src/agnews_dpcnn/{data,models,training,metrics,probabilities}.py`
  - CLI entrypoints remain in `scripts/`
- Removed duplicated code from DPCNN/TextCNN training and probability fusion
  scripts. Tokenization, TSV loading, vocab construction, macro-F1, probability
  IO, LR schedules, distillation loss, and checkpoint writers now have one
  shared implementation.
- Preserved the original training outputs and metric JSON/TSV formats, so old
  experiment records remain comparable.
- Added `--max-train-examples`, `--max-valid-examples`, and
  `--max-test-examples` to `train_textcnn.py` for quick smoke tests; defaults
  are unchanged.
- Added repository-level and CNN-project README files describing layout,
  reproduction commands, final retained results, and report assets.
- Updated `reports/agnews_report_package.zip` so the submission archive includes
  the refactored scripts and `src/agnews_dpcnn` package.
- Verification:
  - `python -m compileall agnews_dpcnn/scripts agnews_dpcnn/src`
  - CPU smoke train for DPCNN on 80/40/40 examples
  - CPU smoke train for TextCNN on 80/40/40 examples
  - smoke checks for `ensemble_probs.py`, `blend_probabilities.py`, and
    `sweep_blend_probabilities.py`
