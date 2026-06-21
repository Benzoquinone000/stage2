"""Write out-of-fold validation probabilities for a 5-fold BERT run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[1]
TASK_SRC = TASK_DIR / "src"
PACKAGE_SRC = ROOT / "mini_transformers" / "src"
sys.path.insert(0, str(TASK_SRC))
sys.path.insert(0, str(PACKAGE_SRC))

from ensemble_bert_classifiers import TextClassificationDataset, collate, predict_probs, probability_nll
from mini_transformers.tokenization import load_tokenizer
from agnews_classification.training_utils import accuracy, macro_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--data-root", default="data/processed_clean")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def write_predictions(path: Path, labels: torch.Tensor, probs: torch.Tensor, fold_ids: torch.Tensor) -> None:
    preds = probs.argmax(dim=-1)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "fold", "label", "prediction", "prob_0", "prob_1", "prob_2", "prob_3"])
        for idx, (fold, label, pred, row) in enumerate(
            zip(fold_ids.tolist(), labels.tolist(), preds.tolist(), probs.tolist())
        ):
            writer.writerow([idx, fold, label, pred, *[f"{value:.8f}" for value in row]])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_enabled = args.amp and args.device.startswith("cuda")

    probs_chunks = []
    label_chunks = []
    fold_chunks = []
    per_fold = []
    for fold in range(args.num_folds):
        checkpoint = Path(args.fold_root) / f"fold_{fold}"
        valid_file = Path(args.data_root) / "folds" / f"fold_{fold}" / "valid.tsv"
        dataset = TextClassificationDataset(valid_file, load_tokenizer(checkpoint), args.max_length)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        probs, labels = predict_probs(checkpoint, loader, args.device, amp_enabled)
        preds = probs.argmax(dim=-1)
        row = {
            "fold": fold,
            "checkpoint": str(checkpoint),
            "valid_file": str(valid_file),
            "valid_loss": probability_nll(probs, labels),
            "valid_accuracy": accuracy(probs, labels),
            "valid_macro_f1": macro_f1(probs, labels, num_labels=4),
            "num_examples": int(labels.numel()),
        }
        per_fold.append(row)
        probs_chunks.append(probs)
        label_chunks.append(labels)
        fold_chunks.append(torch.full_like(labels, fold))
        print(
            f"fold_{fold}: valid_acc={row['valid_accuracy']:.6f} "
            f"valid_f1={row['valid_macro_f1']:.6f}"
        )

    probs_all = torch.cat(probs_chunks)
    labels_all = torch.cat(label_chunks)
    fold_ids = torch.cat(fold_chunks)
    metrics = {
        "num_folds": args.num_folds,
        "num_examples": int(labels_all.numel()),
        "valid_loss": probability_nll(probs_all, labels_all),
        "valid_accuracy": accuracy(probs_all, labels_all),
        "valid_macro_f1": macro_f1(probs_all, labels_all, num_labels=4),
        "per_fold": per_fold,
    }
    (output_dir / "oof_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "oof_predictions.tsv", labels_all, probs_all, fold_ids)
    print(
        f"oof valid_loss={metrics['valid_loss']:.6f} "
        f"valid_acc={metrics['valid_accuracy']:.6f} "
        f"valid_f1={metrics['valid_macro_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
