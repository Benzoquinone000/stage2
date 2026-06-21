"""Average teacher checkpoints and export high-confidence pseudo labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class UnlabeledTextDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int, max_examples: int | None = None) -> None:
        self.rows = []
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames and "text" in reader.fieldnames:
                for row in reader:
                    self.rows.append((row.get("id", str(len(self.rows))), row["text"]))
                    if max_examples is not None and len(self.rows) >= max_examples:
                        break
            else:
                f.seek(0)
                raw_reader = csv.reader(f, delimiter="\t")
                for row in raw_reader:
                    if len(row) >= 2:
                        self.rows.append((str(len(self.rows)), row[1]))
                        if max_examples is not None and len(self.rows) >= max_examples:
                            break
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row_id, text = self.rows[idx]
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length, padding=False)
        encoded["row_id"] = row_id
        encoded["text"] = text
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--input-file", default="data/unlabeled_news_pool.tsv")
    parser.add_argument("--output-dir", default="outputs/pseudo_labels")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--class-thresholds", type=float, nargs=4, default=None)
    parser.add_argument("--min-agree", type=int, default=1)
    parser.add_argument("--max-per-class", type=int, default=25000)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def collate_fn(tokenizer):
    def collate(features: list[dict]) -> dict:
        row_ids = [feature.pop("row_id") for feature in features]
        texts = [feature.pop("text") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["row_ids"] = row_ids
        batch["texts"] = texts
        return batch

    return collate


def move_model_inputs(batch: dict, device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items() if isinstance(value, torch.Tensor)}


@torch.no_grad()
def predict_one_checkpoint(
    checkpoint: str,
    loader: DataLoader,
    device: str,
    amp_enabled: bool,
    log_every: int,
) -> torch.Tensor:
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()
    probs_chunks = []
    for step, batch in enumerate(loader, start=1):
        inputs = move_model_inputs(batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(**inputs).logits
        probs_chunks.append(torch.softmax(logits.float(), dim=-1).cpu())
        if log_every > 0 and (step % log_every == 0 or step == len(loader)):
            print(f"{checkpoint}: batch {step:05d}/{len(loader):05d}")
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return torch.cat(probs_chunks)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_enabled = args.amp and args.device.startswith("cuda")
    if args.weights is not None and len(args.weights) != len(args.checkpoints):
        raise ValueError("--weights must have the same length as --checkpoints")
    weights = args.weights or [1.0] * len(args.checkpoints)
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("--weights must sum to a positive value")
    weights = [weight / weight_sum for weight in weights]

    probs_sum = None
    teacher_preds = []
    per_checkpoint = []
    rows = None
    for checkpoint, weight in zip(args.checkpoints, weights):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
        dataset = UnlabeledTextDataset(args.input_file, tokenizer, args.max_length, args.max_examples)
        if rows is None:
            rows = dataset.rows
        elif rows != dataset.rows:
            raise ValueError(f"input row mismatch while reading {args.input_file}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn(tokenizer))
        probs = predict_one_checkpoint(checkpoint, loader, args.device, amp_enabled, args.log_every)
        probs_sum = weight * probs if probs_sum is None else probs_sum + weight * probs
        teacher_preds.append(probs.argmax(dim=-1))
        per_checkpoint.append({"checkpoint": checkpoint, "weight": weight})
    assert probs_sum is not None
    assert rows is not None
    probs = probs_sum
    max_probs, preds = probs.max(dim=-1)
    teacher_pred_tensor = torch.stack(teacher_preds)
    agree_counts = (teacher_pred_tensor == preds.unsqueeze(0)).sum(dim=0)

    all_path = output_dir / "teacher_ensemble_all_predictions.tsv"
    pseudo_path = output_dir / "teacher_ensemble_pseudo.tsv"
    records = []
    candidates_by_class = {idx: 0 for idx in range(4)}
    for (row_id, text), pred, max_prob, agree_count, row_probs in zip(
        rows,
        preds.tolist(),
        max_probs.tolist(),
        agree_counts.tolist(),
        probs.tolist(),
    ):
        record = {
            "row_id": row_id,
            "text": text,
            "prediction": pred,
            "max_prob": max_prob,
            "agree_count": agree_count,
            "probs": row_probs,
        }
        records.append(record)
        threshold = args.class_thresholds[pred] if args.class_thresholds is not None else args.threshold
        if max_prob >= threshold and agree_count >= args.min_agree:
            candidates_by_class[pred] += 1

    selected_records = []
    selected_by_class = {idx: 0 for idx in range(4)}
    selected_min_prob_by_class = {}
    for cls in range(4):
        threshold = args.class_thresholds[cls] if args.class_thresholds is not None else args.threshold
        class_records = [
            record
            for record in records
            if record["prediction"] == cls
            and record["max_prob"] >= threshold
            and record["agree_count"] >= args.min_agree
        ]
        class_records.sort(key=lambda record: record["max_prob"], reverse=True)
        if args.max_per_class is not None:
            class_records = class_records[: args.max_per_class]
        selected_records.extend(class_records)
        selected_by_class[cls] = len(class_records)
        if class_records:
            selected_min_prob_by_class[cls] = min(record["max_prob"] for record in class_records)
    selected_records.sort(key=lambda record: (record["prediction"], -record["max_prob"]))

    with all_path.open("w", encoding="utf-8", newline="") as all_f, pseudo_path.open(
        "w", encoding="utf-8", newline=""
    ) as pseudo_f:
        all_writer = csv.writer(all_f, delimiter="\t", lineterminator="\n")
        pseudo_writer = csv.writer(pseudo_f, delimiter="\t", lineterminator="\n")
        all_writer.writerow(
            ["id", "prediction", "max_prob", "agree_count", "prob_0", "prob_1", "prob_2", "prob_3", "text"]
        )
        pseudo_writer.writerow(
            ["label", "text", "prob_0", "prob_1", "prob_2", "prob_3", "max_prob", "agree_count"]
        )
        for record in records:
            all_writer.writerow(
                [
                    record["row_id"],
                    record["prediction"],
                    f"{record['max_prob']:.8f}",
                    record["agree_count"],
                    *[f"{value:.8f}" for value in record["probs"]],
                    record["text"],
                ]
            )
        for record in selected_records:
            pseudo_writer.writerow(
                [
                    record["prediction"],
                    record["text"],
                    *[f"{value:.8f}" for value in record["probs"]],
                    f"{record['max_prob']:.8f}",
                    record["agree_count"],
                ]
            )

    summary = {
        "checkpoints": per_checkpoint,
        "input_file": args.input_file,
        "num_input_rows": len(rows),
        "threshold": args.threshold,
        "class_thresholds": args.class_thresholds,
        "min_agree": args.min_agree,
        "max_per_class": args.max_per_class,
        "num_selected": len(selected_records),
        "candidates_by_class": candidates_by_class,
        "selected_by_class": selected_by_class,
        "selected_min_prob_by_class": selected_min_prob_by_class,
        "all_predictions_file": str(all_path),
        "pseudo_label_file": str(pseudo_path),
    }
    (output_dir / "teacher_ensemble_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
