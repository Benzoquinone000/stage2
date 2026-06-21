"""Average probabilities from multiple fine-tuned BERT classifiers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[1]
PACKAGE_SRC = ROOT / "mini_transformers" / "src"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(PACKAGE_SRC))

from mini_transformers.models import BertForSequenceClassification
from mini_transformers.tokenization import load_tokenizer
from task2_utils import accuracy, macro_f1, record_experiment


class TextClassificationDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int) -> None:
        self.rows = self.read_rows(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    @staticmethod
    def read_rows(path: str | Path) -> list[tuple[int, str]]:
        rows: list[tuple[int, str]] = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    rows.append((int(row[0]), row[1]))
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        label, text = self.rows[idx]
        encoded = self.tokenizer.encode(text, max_length=self.max_length)
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "labels": label,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--test-file", default="data/processed_clean/agnews_test.tsv")
    parser.add_argument("--output-dir", default="outputs/fivefold_best512_clean_tapt_more/ensemble")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def pad_sequences(sequences: list[list[int]], pad_value: int = 0) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    return torch.tensor([seq + [pad_value] * (max_len - len(seq)) for seq in sequences], dtype=torch.long)


def collate(features: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": pad_sequences([feature["input_ids"] for feature in features], 0),
        "attention_mask": pad_sequences([feature["attention_mask"] for feature in features], 0),
        "labels": torch.tensor([feature["labels"] for feature in features], dtype=torch.long),
    }


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def predict_probs(
    checkpoint: str | Path,
    loader: DataLoader,
    device: str,
    amp_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = BertForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()
    probs_chunks = []
    label_chunks = []
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch.pop("labels")
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(**batch)["logits"]
        probs_chunks.append(torch.softmax(logits, dim=-1).cpu())
        label_chunks.append(labels.cpu())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(probs_chunks), torch.cat(label_chunks)


def probability_nll(probs: torch.Tensor, labels: torch.Tensor) -> float:
    chosen = probs[torch.arange(labels.numel()), labels].clamp_min(1e-12)
    return float((-chosen.log()).mean().item())


def write_predictions(path: Path, labels: torch.Tensor, probs: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = probs.argmax(dim=-1)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "label", "prediction", "prob_0", "prob_1", "prob_2", "prob_3"])
        for idx, (label, pred, row) in enumerate(zip(labels.tolist(), preds.tolist(), probs.tolist())):
            writer.writerow([idx, label, pred, *[f"{value:.8f}" for value in row]])


def main() -> None:
    args = parse_args()
    checkpoints = [str(Path(path)) for path in args.checkpoints]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(checkpoints[0])
    dataset = TextClassificationDataset(args.test_file, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    amp_enabled = args.amp and args.device.startswith("cuda")

    probs_sum = None
    labels = None
    per_model = []
    for checkpoint in checkpoints:
        probs, current_labels = predict_probs(checkpoint, loader, args.device, amp_enabled)
        labels = current_labels if labels is None else labels
        probs_sum = probs if probs_sum is None else probs_sum + probs
        per_model.append(
            {
                "checkpoint": checkpoint,
                "test_loss": probability_nll(probs, current_labels),
                "test_accuracy": accuracy(probs, current_labels),
                "test_macro_f1": macro_f1(probs, current_labels, num_labels=4),
            }
        )
        print(
            f"{checkpoint}: acc={per_model[-1]['test_accuracy']:.6f} "
            f"f1={per_model[-1]['test_macro_f1']:.6f}"
        )

    assert probs_sum is not None and labels is not None
    ensemble_probs = probs_sum / len(checkpoints)
    metrics = {
        "num_models": len(checkpoints),
        "test_loss": probability_nll(ensemble_probs, labels),
        "test_accuracy": accuracy(ensemble_probs, labels),
        "test_macro_f1": macro_f1(ensemble_probs, labels, num_labels=4),
        "per_model": per_model,
    }
    (output_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "ensemble_predictions.tsv", labels, ensemble_probs)
    record_experiment(
        stage="classification_ensemble",
        output_dir=output_dir,
        config={
            "checkpoints": checkpoints,
            "test_file": args.test_file,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "amp": amp_enabled,
            "device": args.device,
        },
        metrics={key: value for key, value in metrics.items() if key != "per_model"},
        notes="five-fold probability ensemble from current best 512x8 checkpoint family",
    )
    print(
        f"ensemble test_loss={metrics['test_loss']:.6f} "
        f"test_acc={metrics['test_accuracy']:.6f} "
        f"test_f1={metrics['test_macro_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
