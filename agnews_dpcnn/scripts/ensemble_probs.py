"""Average saved probability files and report test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agnews_dpcnn.metrics import read_probs, write_predictions
from agnews_dpcnn.probabilities import average_probs, evaluate_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = None
    probs_list = []
    per_model = []
    for prob_file in args.prob_files:
        current_labels, probs = read_probs(prob_file)
        if labels is None:
            labels = current_labels
        elif not torch.equal(labels, current_labels):
            raise ValueError(f"label mismatch: {prob_file}")
        probs_list.append(probs)
        row = {"prob_file": str(prob_file), **evaluate_probs(probs, current_labels, prefix="test")}
        per_model.append(row)
        print(f"{prob_file}: acc={row['test_accuracy']:.6f} f1={row['test_macro_f1']:.6f}")

    assert labels is not None
    ensemble_probs = average_probs(probs_list)
    metrics = {
        "num_models": len(args.prob_files),
        **evaluate_probs(ensemble_probs, labels, prefix="test"),
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
