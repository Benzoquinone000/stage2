"""Average saved DPCNN probability files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-files", nargs="+", required=True)
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
    labels = None
    probs_sum = None
    per_model = []
    for prob_file in args.prob_files:
        current_labels, probs = read_probs(prob_file)
        if labels is None:
            labels = current_labels
        else:
            assert torch.equal(labels, current_labels), f"label mismatch: {prob_file}"
        probs_sum = probs if probs_sum is None else probs_sum + probs
        preds = probs.argmax(dim=-1)
        per_model.append(
            {
                "prob_file": str(prob_file),
                "test_loss": nll(probs, current_labels),
                "test_accuracy": float((preds == current_labels).float().mean().item()),
                "test_macro_f1": macro_f1(preds, current_labels),
            }
        )
        print(
            f"{prob_file}: acc={per_model[-1]['test_accuracy']:.6f} "
            f"f1={per_model[-1]['test_macro_f1']:.6f}"
        )
    assert labels is not None and probs_sum is not None
    ensemble_probs = probs_sum / len(args.prob_files)
    preds = ensemble_probs.argmax(dim=-1)
    metrics = {
        "num_models": len(args.prob_files),
        "test_loss": nll(ensemble_probs, labels),
        "test_accuracy": float((preds == labels).float().mean().item()),
        "test_macro_f1": macro_f1(preds, labels),
        "per_model": per_model,
    }
    (output_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "ensemble_predictions.tsv", labels, ensemble_probs)
    print(
        f"ensemble test_loss={metrics['test_loss']:.6f} "
        f"test_acc={metrics['test_accuracy']:.6f} "
        f"test_f1={metrics['test_macro_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
