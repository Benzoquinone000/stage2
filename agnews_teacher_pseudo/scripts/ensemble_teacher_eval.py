"""Evaluate a heterogeneous teacher checkpoint ensemble on labeled TSV data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class TsvTextDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int) -> None:
        self.rows = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    self.rows.append((int(row[0]), row[1]))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        label, text = self.rows[idx]
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
        encoded["labels"] = label
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--data-file", default="../agnews_classification/data/processed_clean/agnews_test.tsv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def collate_fn(tokenizer):
    def collate(features: list[dict]) -> dict[str, torch.Tensor]:
        return tokenizer.pad(features, padding=True, return_tensors="pt")

    return collate


def move_inputs(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def predict_checkpoint(
    checkpoint: str | Path,
    data_file: str | Path,
    max_length: int,
    batch_size: int,
    device: str,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    dataset = TsvTextDataset(data_file, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn(tokenizer))
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()
    probs_chunks = []
    label_chunks = []
    for batch in loader:
        batch = move_inputs(batch, device)
        labels = batch.pop("labels")
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(**batch).logits
        probs_chunks.append(torch.softmax(logits.float(), dim=-1).cpu())
        label_chunks.append(labels.cpu())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(probs_chunks), torch.cat(label_chunks)


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


def nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    chosen = probs[torch.arange(labels.numel()), labels].clamp_min(1e-12)
    return float((-chosen.log()).mean().item())


def write_predictions(path: Path, labels: torch.Tensor, probs: torch.Tensor) -> None:
    preds = probs.argmax(dim=-1)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "label", "prediction", "prob_0", "prob_1", "prob_2", "prob_3"])
        for idx, (label, pred, row) in enumerate(zip(labels.tolist(), preds.tolist(), probs.tolist())):
            writer.writerow([idx, label, pred, *[f"{value:.8f}" for value in row]])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise ValueError("--weights must match --checkpoints")
    weights = args.weights or [1.0] * len(args.checkpoints)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("--weights must sum to a positive value")
    weights = [weight / weight_sum for weight in weights]
    amp_enabled = args.amp and args.device.startswith("cuda")

    labels = None
    probs_sum = None
    per_model = []
    for checkpoint, weight in zip(args.checkpoints, weights):
        probs, current_labels = predict_checkpoint(
            checkpoint,
            args.data_file,
            args.max_length,
            args.batch_size,
            args.device,
            amp_enabled,
        )
        if labels is None:
            labels = current_labels
        elif not torch.equal(labels, current_labels):
            raise ValueError(f"label mismatch for {checkpoint}")
        probs_sum = weight * probs if probs_sum is None else probs_sum + weight * probs
        preds = probs.argmax(dim=-1)
        row = {
            "checkpoint": checkpoint,
            "weight": weight,
            "loss": nll(probs, current_labels),
            "accuracy": float((preds == current_labels).float().mean().item()),
            "macro_f1": macro_f1(preds, current_labels),
        }
        per_model.append(row)
        print(f"{checkpoint}: acc={row['accuracy']:.6f} f1={row['macro_f1']:.6f}")

    assert labels is not None and probs_sum is not None
    preds = probs_sum.argmax(dim=-1)
    metrics = {
        "data_file": args.data_file,
        "num_examples": int(labels.numel()),
        "loss": nll(probs_sum, labels),
        "accuracy": float((preds == labels).float().mean().item()),
        "macro_f1": macro_f1(preds, labels),
        "per_model": per_model,
    }
    (output_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "ensemble_predictions.tsv", labels, probs_sum)
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_model"}, indent=2))


if __name__ == "__main__":
    main()
