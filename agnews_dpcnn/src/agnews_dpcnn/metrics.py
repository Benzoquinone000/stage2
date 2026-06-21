"""Metrics and probability-file utilities shared by CNN experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import torch


def accuracy_from_preds(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return float((preds == labels).float().mean().item())


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return accuracy_from_preds(logits.argmax(dim=-1), labels)


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


def probability_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    chosen = probs[torch.arange(labels.numel()), labels].clamp_min(1e-12)
    return float((-chosen.log()).mean().item())


def read_probs(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    labels = []
    probs = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels.append(int(row["label"]))
            probs.append([float(row[f"prob_{idx}"]) for idx in range(4)])
    return torch.tensor(labels, dtype=torch.long), torch.tensor(probs, dtype=torch.float32)


def write_predictions(path: Path, labels: torch.Tensor, probs: torch.Tensor, fold_ids: torch.Tensor | None = None) -> None:
    preds = probs.argmax(dim=-1)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        if fold_ids is None:
            writer.writerow(["id", "label", "prediction", "prob_0", "prob_1", "prob_2", "prob_3"])
            for idx, (label, pred, row) in enumerate(zip(labels.tolist(), preds.tolist(), probs.tolist())):
                writer.writerow([idx, label, pred, *[f"{value:.8f}" for value in row]])
        else:
            writer.writerow(["id", "fold", "label", "prediction", "prob_0", "prob_1", "prob_2", "prob_3"])
            for idx, (fold, label, pred, row) in enumerate(
                zip(fold_ids.tolist(), labels.tolist(), preds.tolist(), probs.tolist())
            ):
                writer.writerow([idx, fold, label, pred, *[f"{value:.8f}" for value in row]])

