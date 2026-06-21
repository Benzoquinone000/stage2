"""Write out-of-fold validation probabilities for a 5-fold TextCNN run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agnews_dpcnn.data import load_vocab, make_loader, read_tsv
from agnews_dpcnn.metrics import accuracy_from_preds, macro_f1, probability_nll, write_predictions
from agnews_dpcnn.models import TextCNN
from agnews_dpcnn.training import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", required=True)
    parser.add_argument("--data-root", default="../agnews_classification/data/processed_clean")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def load_model(fold_dir: Path, device: str) -> tuple[TextCNN, dict[str, int], dict]:
    config = json.loads((fold_dir / "train_config.json").read_text(encoding="utf-8"))
    vocab = load_vocab(fold_dir / "vocab.jsonl")
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=int(config["embedding_dim"]),
        num_filters=int(config["num_filters"]),
        kernel_sizes=[int(value) for value in config["kernel_sizes"]],
        dropout=float(config["dropout"]),
        embedding_dropout=float(config["embedding_dropout"]),
    ).to(device)
    checkpoint = torch.load(fold_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocab, config


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
        fold_dir = Path(args.fold_root) / f"fold_{fold}"
        valid_file = Path(args.data_root) / "folds" / f"fold_{fold}" / "valid.tsv"
        model, vocab, config = load_model(fold_dir, args.device)
        rows = read_tsv(valid_file)
        loader = make_loader(rows, vocab, int(config["max_length"]), args.batch_size, shuffle=False)
        valid = evaluate(model, loader, args.device, amp_enabled)
        logits = valid["logits"]
        labels = valid["labels"]
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        row = {
            "fold": fold,
            "checkpoint": str(fold_dir),
            "valid_file": str(valid_file),
            "valid_loss": probability_nll(probs, labels),
            "valid_accuracy": accuracy_from_preds(preds, labels),
            "valid_macro_f1": macro_f1(preds, labels),
            "num_examples": int(labels.numel()),
        }
        per_fold.append(row)
        probs_chunks.append(probs)
        label_chunks.append(labels)
        fold_chunks.append(torch.full_like(labels, fold))
        print(f"fold_{fold}: valid_acc={row['valid_accuracy']:.6f} valid_f1={row['valid_macro_f1']:.6f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    probs_all = torch.cat(probs_chunks)
    labels_all = torch.cat(label_chunks)
    fold_ids = torch.cat(fold_chunks)
    preds_all = probs_all.argmax(dim=-1)
    metrics = {
        "num_folds": args.num_folds,
        "num_examples": int(labels_all.numel()),
        "valid_loss": probability_nll(probs_all, labels_all),
        "valid_accuracy": accuracy_from_preds(preds_all, labels_all),
        "valid_macro_f1": macro_f1(preds_all, labels_all),
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
