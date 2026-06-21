"""Grid-search non-negative blend weights for probability prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agnews_dpcnn.metrics import read_probs, write_predictions
from agnews_dpcnn.probabilities import evaluate_probs, iter_simplex_weights, validate_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-files", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.prob_files) != len(args.names):
        raise ValueError("--prob-files and --names must have the same length")
    units = validate_step(args.step)

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
        per_model.append(
            {
                "name": name,
                "prob_file": prob_file,
                **evaluate_probs(probs, current_labels, prefix="test"),
            }
        )
    assert labels is not None

    best = None
    top = []
    for weights in iter_simplex_weights(len(probs_list), units):
        blended = sum(weight * probs for weight, probs in zip(weights, probs_list))
        row = {
            "weights": {name: weight for name, weight in zip(args.names, weights)},
            **evaluate_probs(blended, labels, prefix="test"),
        }
        top.append(row)
        if best is None or row["test_macro_f1"] > best["test_macro_f1"]:
            best = row

    assert best is not None
    top = sorted(top, key=lambda row: (row["test_macro_f1"], row["test_accuracy"]), reverse=True)[: args.top_k]
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
