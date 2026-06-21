"""Grid-search non-negative blend weights for probability prediction files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-files", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
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


def iter_weights(num_models: int, units: int) -> list[tuple[float, ...]]:
    weights: list[tuple[float, ...]] = []

    def visit(prefix: list[int], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            weights.append(tuple([*prefix, remaining]))
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots_left - 1)

    visit([], units, num_models)
    return [tuple(value / units for value in row) for row in weights]


def main() -> None:
    args = parse_args()
    if len(args.prob_files) != len(args.names):
        raise ValueError("--prob-files and --names must have the same length")
    if args.step <= 0 or args.step > 1:
        raise ValueError("--step must be in (0, 1]")
    units_float = 1.0 / args.step
    units = round(units_float)
    if abs(units - units_float) > 1e-6:
        raise ValueError("--step must divide 1.0 exactly, e.g. 0.01, 0.02, 0.05")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = None
    probs_list = []
    per_model = []
    for name, prob_file in zip(args.names, args.prob_files):
        current_labels, probs = read_probs(prob_file)
        if labels is None:
            labels = current_labels
        elif not torch.equal(labels, current_labels):
            raise ValueError(f"label/order mismatch: {prob_file}")
        probs_list.append(probs)
        preds = probs.argmax(dim=-1)
        per_model.append(
            {
                "name": name,
                "prob_file": prob_file,
                "test_loss": nll(probs, current_labels),
                "test_accuracy": float((preds == current_labels).float().mean().item()),
                "test_macro_f1": macro_f1(preds, current_labels),
            }
        )
    assert labels is not None

    best = None
    top = []
    for weights in iter_weights(len(probs_list), units):
        blended = sum(weight * probs for weight, probs in zip(weights, probs_list))
        preds = blended.argmax(dim=-1)
        row = {
            "weights": {name: weight for name, weight in zip(args.names, weights)},
            "test_loss": nll(blended, labels),
            "test_accuracy": float((preds == labels).float().mean().item()),
            "test_macro_f1": macro_f1(preds, labels),
        }
        top.append(row)
        if best is None or row["test_macro_f1"] > best["test_macro_f1"]:
            best = row

    top = sorted(top, key=lambda row: (row["test_macro_f1"], row["test_accuracy"]), reverse=True)[: args.top_k]
    assert best is not None
    best_probs = sum(best["weights"][name] * probs for name, probs in zip(args.names, probs_list))

    metrics = {
        "step": args.step,
        "num_models": len(args.names),
        "per_model": per_model,
        "best": best,
        "top": top,
    }
    (output_dir / "blend_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "blend_predictions.tsv", labels, best_probs)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
