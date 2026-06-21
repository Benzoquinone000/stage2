# AG News Experiment Log

This report is regenerated from local output directories. Future training scripts also append structured records to `reports/experiment_runs.jsonl`.

## Current Best

- `fivefold_best512_clean_tapt_more/ensemble`: test accuracy `0.928289`, macro F1 `0.928108`.
- Reflection: five-fold probability averaging improved generalization; keep this as the current scoring anchor.

## Ensemble Runs

| rank | run | models | test acc | test f1 | test loss | reflection |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `fivefold_best512_clean_tapt_more/ensemble` | 5 | 0.928289 | 0.928108 | 0.229370 | five-fold probability averaging improved generalization; keep this as the current scoring anchor. |

## Classifier Runs

| rank | run | test acc | test f1 | best valid f1 | best epoch | reflection |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `bert_classifier_clean_tapt_more_lr15e4_sm02_do015_512x8_len128` | 0.924737 | 0.924655 | 0.924749 | 6 | cleaned data plus a smaller holdout improved test F1; prioritize clean data for folds and future pretraining. |

## MLM Runs

| run | best valid loss | best ppl | best epoch | reflection |
| --- | --- | --- | --- | --- |
| `bert_mlm_news_augmented_tapt_512x8_len128` | 5.0675 | 158.77 | 3 | TAPT lowered AG News MLM loss after DAPT; keep DAPT->TAPT structure. |
| `bert_mlm_news_augmented_tapt_more_512x8_len128` | 5.0704 | 159.24 | 2 | extra TAPT matched prior TAPT perplexity but helped test after classification; useful but watch overfit. |
| `bert_mlm_news_augmented_512x8_len128` | 5.6276 | 277.99 | 4 | DAPT alone has worse AG News validation loss than TAPT; it is a domain bridge, not final checkpoint. |

## Next Decisions

- Keep `outputs/fivefold_best512_clean_tapt_more/ensemble` as the current scoring anchor.
- Next score gains should come from calibrated/weighted ensembling or a genuinely diverse second model family, not from rerunning the failed standard BERT-base branch.
- Preserve clean data, the 512x8 TAPT checkpoint family, and five-fold probability outputs for final submission analysis.
