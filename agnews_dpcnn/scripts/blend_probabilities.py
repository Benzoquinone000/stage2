"""Blend two probability prediction files with a fixed weight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agnews_dpcnn.metrics import read_probs, write_predictions
from agnews_dpcnn.probabilities import evaluate_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-file", required=True)
    parser.add_argument("--b-file", required=True)
    parser.add_argument("--a-name", default="a")
    parser.add_argument("--b-name", default="b")
    parser.add_argument("--a-weight", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


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
    metrics = {
        "a_name": args.a_name,
        "b_name": args.b_name,
        "a_file": args.a_file,
        "b_file": args.b_file,
        "a_weight": args.a_weight,
        "b_weight": 1.0 - args.a_weight,
        **evaluate_probs(probs, labels_a, prefix="test"),
    }
    (output_dir / "blend_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "blend_predictions.tsv", labels_a, probs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
