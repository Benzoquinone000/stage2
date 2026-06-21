"""Blend two probability prediction files with a fixed weight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-file", required=True)
    parser.add_argument("--b-file", required=True)
    parser.add_argument("--a-name", default="a")
    parser.add_argument("--b-name", default="b")
    parser.add_argument("--a-weight", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_probs(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    labels = []
    probs = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels.append(int(row["label"]))
            probs.append([float(row[f"prob_{idx}"]) for idx in range(4)])
    return torch.tensor(labels, dtype=torch.long), torch.tensor(probs, dtype=torch.float32)


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
    labels_a, probs_a = read_probs(args.a_file)
    labels_b, probs_b = read_probs(args.b_file)
    if not torch.equal(labels_a, labels_b):
        raise ValueError("label/order mismatch between probability files")
    if not 0.0 <= args.a_weight <= 1.0:
        raise ValueError("--a-weight must be in [0, 1]")
    probs = args.a_weight * probs_a + (1.0 - args.a_weight) * probs_b
    preds = probs.argmax(dim=-1)
    metrics = {
        "a_name": args.a_name,
        "b_name": args.b_name,
        "a_file": args.a_file,
        "b_file": args.b_file,
        "a_weight": args.a_weight,
        "b_weight": 1.0 - args.a_weight,
        "test_loss": nll(probs, labels_a),
        "test_accuracy": float((preds == labels_a).float().mean().item()),
        "test_macro_f1": macro_f1(preds, labels_a),
    }
    (output_dir / "blend_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "blend_predictions.tsv", labels_a, probs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
