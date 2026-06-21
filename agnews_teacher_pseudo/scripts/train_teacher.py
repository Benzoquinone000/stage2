"""Fine-tune a pretrained Hugging Face teacher on AG News."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


class TsvTextDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int, max_examples: int | None = None) -> None:
        self.rows = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    self.rows.append((int(row[0]), row[1]))
                    if max_examples is not None and len(self.rows) >= max_examples:
                        break
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        label, text = self.rows[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = label
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large")
    parser.add_argument("--train-file", default="../agnews_classification/data/processed_clean/agnews_train.tsv")
    parser.add_argument("--valid-file", default="../agnews_classification/data/processed_clean/agnews_valid.tsv")
    parser.add_argument("--test-file", default="../agnews_classification/data/processed_clean/agnews_test.tsv")
    parser.add_argument("--output-dir", default="outputs/deberta_v3_large_single")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout-prob", type=float, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="agnews-teacher")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def collate_fn(tokenizer):
    def collate(features: list[dict]) -> dict[str, torch.Tensor]:
        return tokenizer.pad(features, padding=True, return_tensors="pt")

    return collate


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 4) -> float:
    scores = []
    for cls in range(num_classes):
        pred_pos = preds == cls
        true_pos = labels == cls
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


@torch.no_grad()
def evaluate(model, loader, device: str, amp_enabled: bool) -> dict[str, float | torch.Tensor]:
    model.eval()
    logits_chunks = []
    label_chunks = []
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch["labels"]
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(**batch)
        batch_size = labels.size(0)
        total_loss += outputs.loss.item() * batch_size
        total_examples += batch_size
        logits_chunks.append(outputs.logits.float().cpu())
        label_chunks.append(labels.cpu())
    logits = torch.cat(logits_chunks)
    labels = torch.cat(label_chunks)
    preds = logits.argmax(dim=-1)
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": float((preds == labels).float().mean().item()),
        "macro_f1": macro_f1(preds, labels),
        "logits": logits,
        "labels": labels,
    }


def write_predictions(path: Path, labels: torch.Tensor, logits: torch.Tensor) -> None:
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "label", "prediction", "prob_0", "prob_1", "prob_2", "prob_3"])
        for idx, (label, pred, row) in enumerate(zip(labels.tolist(), preds.tolist(), probs.tolist())):
            writer.writerow([idx, label, pred, *[f"{value:.8f}" for value in row]])


def maybe_set_dropout(config, dropout_prob: float | None) -> None:
    if dropout_prob is None:
        return
    for name in [
        "hidden_dropout_prob",
        "attention_probs_dropout_prob",
        "classifier_dropout",
        "summary_last_dropout",
        "pooler_dropout",
        "dropout",
    ]:
        if hasattr(config, name):
            setattr(config, name, dropout_prob)


def init_wandb(args: argparse.Namespace):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    import wandb

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    return wandb


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    best_dir = output_dir / "best"
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb = init_wandb(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=4,
        id2label={idx: name for idx, name in enumerate(LABEL_NAMES)},
        label2id={name: idx for idx, name in enumerate(LABEL_NAMES)},
    )
    maybe_set_dropout(config, args.dropout_prob)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.float()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(args.device)

    train_dataset = TsvTextDataset(args.train_file, tokenizer, args.max_length, args.max_train_examples)
    valid_dataset = TsvTextDataset(args.valid_file, tokenizer, args.max_length, args.max_valid_examples)
    test_dataset = TsvTextDataset(args.test_file, tokenizer, args.max_length, args.max_test_examples)
    collate = collate_fn(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight")
    grouped_params = [
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and not any(term in n for term in no_decay_terms)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if p.requires_grad and any(term in n for term in no_decay_terms)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    amp_enabled = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"model: {args.model_name}")
    print(f"train/valid/test: {len(train_dataset):,}/{len(valid_dataset):,}/{len(test_dataset):,}")
    print(f"effective batch: {args.batch_size * args.gradient_accumulation_steps}; total_steps={total_steps}")

    best_valid_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, args.device)
            labels = batch["labels"]
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(**batch)
                loss = F.cross_entropy(outputs.logits, labels, label_smoothing=args.label_smoothing)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            should_step = step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
            total_examples += batch_size
            if args.log_every > 0 and (step % args.log_every == 0 or step == len(train_loader)):
                print(
                    f"epoch {epoch:02d} step {step:05d}/{len(train_loader):05d} "
                    f"train_loss={total_loss / max(1, total_examples):.4f} "
                    f"train_acc={total_correct / max(1, total_examples):.4f}"
                )

        valid_metrics = evaluate(model, valid_loader, args.device, amp_enabled)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_examples),
            "train_accuracy": total_correct / max(1, total_examples),
            "valid_loss": float(valid_metrics["loss"]),
            "valid_accuracy": float(valid_metrics["accuracy"]),
            "valid_macro_f1": float(valid_metrics["macro_f1"]),
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": perf_counter() - start,
        }
        history.append(row)
        if wandb is not None:
            wandb.log(row, step=epoch)
        print(
            f"epoch {epoch:02d} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} valid_loss={row['valid_loss']:.4f} "
            f"valid_acc={row['valid_accuracy']:.4f} valid_f1={row['valid_macro_f1']:.4f} "
            f"time={row['seconds']:.1f}s"
        )
        if row["valid_macro_f1"] > best_valid_f1:
            best_valid_f1 = row["valid_macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            bad_epochs += 1
            if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
                print(
                    f"early stopping after epoch {epoch:02d}; "
                    f"best_epoch={best_epoch:02d} best_valid_f1={best_valid_f1:.6f}"
                )
                break

    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(args.device)
    test_metrics = evaluate(best_model, test_loader, args.device, amp_enabled)
    metrics = {
        "model_name": args.model_name,
        "best_valid_macro_f1": best_valid_f1,
        "best_epoch": best_epoch,
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "test_predictions.tsv", test_metrics["labels"], test_metrics["logits"])
    if wandb is not None:
        wandb.log({f"test/{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
        wandb.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
