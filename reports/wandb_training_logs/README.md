# W&B Training Logs and Exported Histories

This directory contains reviewable training logs for the AG News experiments.

## What is included

- `training_histories/`: epoch-level CSV histories, configs, and metric JSON files for the final retained BERT/DPCNN/TextCNN runs, plus representative teacher runs.
- `training_histories/run_summary.csv`: compact table of best validation and test metrics.
- `wandb_exports/`: sanitized exports from local W&B run folders when available.

## What is intentionally excluded

Raw W&B run directories are not committed. They can contain binary `.wandb` files, debug logs, absolute machine paths, host names, email addresses, GPU UUIDs, and environment metadata. The exported JSON files keep only safe hyperparameters, final metric summaries, and a short redacted output tail.

Generated from local files under ignored `outputs/` and `wandb/` directories.
