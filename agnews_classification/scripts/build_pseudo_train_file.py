"""Combine gold AG News TSV with high-confidence teacher pseudo labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/processed_clean/agnews_train.tsv")
    parser.add_argument("--pseudo-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-pseudo-per-class", type=int, default=12000)
    parser.add_argument(
        "--include-teacher-probs",
        action="store_true",
        help="Write a header TSV with prob_0..prob_3 and is_pseudo columns for soft distillation.",
    )
    return parser.parse_args()


def read_gold(path: str | Path) -> list[tuple[int, str]]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                rows.append((int(row[0]), row[1]))
    return rows


def read_pseudo(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            item = {
                "label": int(row["label"]),
                "text": row["text"],
                "max_prob": float(row.get("max_prob", "0")),
                "agree_count": row.get("agree_count", ""),
            }
            for class_id in range(4):
                key = f"prob_{class_id}"
                if key in row and row[key] != "":
                    item[key] = float(row[key])
            rows.append(item)
    return rows


def main() -> None:
    args = parse_args()
    gold_rows = read_gold(args.train_file)
    pseudo_rows = read_pseudo(args.pseudo_file)
    selected = []
    pseudo_min_prob_by_class = {}
    for label in range(4):
        class_rows = [row for row in pseudo_rows if row["label"] == label]
        class_rows.sort(key=lambda row: row["max_prob"], reverse=True)
        class_rows = class_rows[: args.max_pseudo_per_class]
        selected.extend(class_rows)
        if class_rows:
            pseudo_min_prob_by_class[str(label)] = min(row["max_prob"] for row in class_rows)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        if args.include_teacher_probs:
            writer.writerow(
                ["label", "text", "prob_0", "prob_1", "prob_2", "prob_3", "max_prob", "agree_count", "is_pseudo"]
            )
            for label, text in gold_rows:
                probs = [1.0 if class_id == label else 0.0 for class_id in range(4)]
                writer.writerow([label, text, *[f"{value:.8f}" for value in probs], "1.00000000", "", 0])
            for row in selected:
                probs = [float(row.get(f"prob_{class_id}", 1.0 if class_id == row["label"] else 0.0)) for class_id in range(4)]
                writer.writerow(
                    [
                        row["label"],
                        row["text"],
                        *[f"{value:.8f}" for value in probs],
                        f"{row['max_prob']:.8f}",
                        row.get("agree_count", ""),
                        1,
                    ]
                )
        else:
            for label, text in gold_rows:
                writer.writerow([label, text])
            for row in selected:
                writer.writerow([row["label"], row["text"]])

    summary = {
        "train_file": args.train_file,
        "pseudo_file": args.pseudo_file,
        "output_file": args.output_file,
        "max_pseudo_per_class": args.max_pseudo_per_class,
        "gold_rows": len(gold_rows),
        "pseudo_rows": len(selected),
        "combined_rows": len(gold_rows) + len(selected),
        "include_teacher_probs": args.include_teacher_probs,
        "counts": {
            "gold": dict(Counter(str(label) for label, _ in gold_rows)),
            "pseudo": dict(Counter(str(row["label"]) for row in selected)),
            "combined": dict(
                Counter(str(label) for label, _ in gold_rows)
                + Counter(str(row["label"]) for row in selected)
            ),
        },
        "pseudo_min_prob_by_class": pseudo_min_prob_by_class,
    }
    summary_path = output_file.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
