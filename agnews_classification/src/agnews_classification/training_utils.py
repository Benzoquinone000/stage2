"""Local Task2 training utilities.

The task uses ``mini_transformers`` for BERT and tokenization, but keeps the
training loop helpers, collators, schedulers, and metrics inside this directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def add_wandb_args(parser) -> None:
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="agnews-bert", help="wandb project name.")
    parser.add_argument("--wandb-entity", default=None, help="wandb entity/team name.")
    parser.add_argument("--wandb-run-name", default=None, help="wandb run name.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="wandb mode. Use offline on machines without external network access.",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional wandb run tags.")


def init_wandb(
    args,
    run_config: dict[str, Any],
    stage: str,
    output_dir,
    extra_config: dict[str, Any] | None = None,
):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb logging was requested, but wandb is not installed. "
            "Run `python -m pip install -r requirements.txt` from the stage2 directory."
        ) from exc

    config = {"stage": stage, **run_config}
    if extra_config:
        config.update(extra_config)

    kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": config,
        "dir": str(output_dir),
    }
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    if args.wandb_mode:
        kwargs["mode"] = args.wandb_mode
    if args.wandb_tags:
        kwargs["tags"] = args.wandb_tags
    return wandb.init(**kwargs)


def log_wandb(run, metrics: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()


def record_experiment(
    stage: str,
    output_dir,
    config: dict[str, Any],
    metrics: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> None:
    """Append a lightweight experiment record for later comparison."""

    output_path = Path(output_dir)
    task_dir = Path(__file__).resolve().parents[2]
    report_dir = task_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "output_dir": str(output_path),
        "config": config,
        "metrics": metrics,
        "notes": notes or "",
    }
    if history:
        record["history_last"] = history[-1]
    with (report_dir / "experiment_runs.jsonl").open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    (output_path / "experiment_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_labels: int = 4) -> float:
    preds = logits.argmax(dim=-1)
    scores = []
    for label_id in range(num_labels):
        tp = ((preds == label_id) & (labels == label_id)).sum().item()
        fp = ((preds == label_id) & (labels != label_id)).sum().item()
        fn = ((preds != label_id) & (labels == label_id)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / max(1, len(scores))


def perplexity(loss: float) -> float:
    return float(math.exp(min(20.0, loss)))


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        remaining = num_training_steps - current_step
        decay_steps = max(1, num_training_steps - num_warmup_steps)
        return max(0.0, float(remaining) / float(decay_steps))

    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)


class DataCollatorForMaskedLanguageModeling:
    def __init__(
        self,
        mask_token_id: int,
        pad_token_id: int = 0,
        special_token_ids: tuple[int, ...] = (0, 2, 3, 4),
        mlm_probability: float = 0.15,
    ) -> None:
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.special_token_ids = special_token_ids
        self.mlm_probability = mlm_probability

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = _pad_sequences([f["input_ids"] for f in features], self.pad_token_id)
        labels = input_ids.clone()
        attention_mask = (input_ids != self.pad_token_id).long()

        probability = torch.full(input_ids.shape, self.mlm_probability)
        for token_id in self.special_token_ids:
            probability.masked_fill_(input_ids == token_id, 0.0)
        masked = torch.bernoulli(probability).bool()
        for row in range(masked.size(0)):
            if not masked[row].any():
                candidates = torch.nonzero(attention_mask[row] > 0, as_tuple=False).flatten()
                candidates = torch.tensor(
                    [
                        index.item()
                        for index in candidates
                        if input_ids[row, index].item() not in self.special_token_ids
                    ],
                    dtype=torch.long,
                )
                if len(candidates) > 0:
                    masked[row, candidates[0]] = True

        labels[~masked] = -100
        input_ids[masked] = self.mask_token_id
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class DataCollatorWithPadding:
    def __init__(self, pad_token_id: int = 0, label_pad_token_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {
            "input_ids": _pad_sequences([f["input_ids"] for f in features], self.pad_token_id),
            "attention_mask": _pad_sequences([f["attention_mask"] for f in features], 0),
        }
        labels = [f["labels"] for f in features]
        if isinstance(labels[0], list):
            batch["labels"] = _pad_sequences(labels, self.label_pad_token_id)
        else:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch
