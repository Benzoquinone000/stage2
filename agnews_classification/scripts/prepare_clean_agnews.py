"""Prepare cleaned AG News splits and stratified CV folds."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import random
import sys
from typing import Iterable


TASK_DIR = Path(__file__).resolve().parents[1]
TASK_SRC = TASK_DIR / "src"
sys.path.insert(0, str(TASK_SRC))

from agnews_classification.text_cleaning import clean_text, dedupe_key


LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", default="data/raw/train.jsonl")
    parser.add_argument("--test-jsonl", default="data/raw/test.jsonl")
    parser.add_argument("--output-dir", default="data/processed_clean")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-conflicting-duplicates",
        action="store_true",
        help="Keep the first row when duplicate cleaned texts have conflicting labels.",
    )
    parser.add_argument(
        "--dedupe-test",
        action="store_true",
        help="Dedupe official test too. Disabled by default to preserve official test size.",
    )
    return parser.parse_args()


def infer_label_offset(values: Iterable[int]) -> int:
    labels = {int(value) for value in values}
    if labels <= {0, 1, 2, 3}:
        return 0
    if labels <= {1, 2, 3, 4}:
        return 1
    raise ValueError(f"unsupported AG News label set: {sorted(labels)}")


def normalize_label(value, offset: int) -> int:
    return int(value) - offset


def read_agnews_jsonl(path: str | Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    label_offset = infer_label_offset(item["label"] for item in items)
    for item in items:
        label = normalize_label(item["label"], label_offset)
        if "text" in item:
            text = clean_text(item["text"])
        else:
            text = clean_text(item.get("title", ""), item.get("description", ""))
        if text:
            rows.append((label, text))
    return rows


def count_changed(raw_path: str | Path) -> int:
    changed = 0
    with Path(raw_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "text" in item:
                original = " ".join(str(item.get("text", "")).split())
                cleaned = clean_text(item.get("text", ""))
            else:
                original = " ".join(
                    part.strip()
                    for part in [item.get("title", ""), item.get("description", "")]
                    if part and part.strip()
                )
                original = " ".join(original.split())
                cleaned = clean_text(item.get("title", ""), item.get("description", ""))
            if original != cleaned:
                changed += 1
    return changed


def dedupe_rows(
    rows: Iterable[tuple[int, str]],
    keep_conflicting_duplicates: bool = False,
) -> tuple[list[tuple[int, str]], dict[str, int]]:
    by_text: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for label, text in rows:
        by_text[dedupe_key(text)].append((label, text))

    deduped: list[tuple[int, str]] = []
    exact_duplicate_groups = 0
    exact_duplicate_extra_rows = 0
    conflicting_groups = 0
    conflicting_rows = 0
    same_label_extra_rows = 0
    for group in by_text.values():
        if len(group) > 1:
            exact_duplicate_groups += 1
            exact_duplicate_extra_rows += len(group) - 1
        labels = {label for label, _ in group}
        if len(labels) > 1:
            conflicting_groups += 1
            conflicting_rows += len(group)
            if not keep_conflicting_duplicates:
                continue
        else:
            same_label_extra_rows += len(group) - 1
        deduped.append(group[0])

    return deduped, {
        "duplicate_groups": exact_duplicate_groups,
        "duplicate_extra_rows": exact_duplicate_extra_rows,
        "conflicting_duplicate_groups": conflicting_groups,
        "conflicting_duplicate_rows": conflicting_rows,
        "same_label_duplicate_extra_rows": same_label_extra_rows,
        "kept_rows": len(deduped),
    }


def label_counts(rows: Iterable[tuple[int, str]]) -> dict[str, int]:
    counts = collections.Counter(label for label, _ in rows)
    return {str(label): counts[label] for label in sorted(counts)}


def stratified_split(
    rows: list[tuple[int, str]],
    valid_ratio: float,
    seed: int,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    rng = random.Random(seed)
    train: list[tuple[int, str]] = []
    valid: list[tuple[int, str]] = []
    by_label: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for row in rows:
        by_label[row[0]].append(row)
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        valid_size = max(1, round(len(group) * valid_ratio))
        valid.extend(group[:valid_size])
        train.extend(group[valid_size:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def make_stratified_folds(
    rows: list[tuple[int, str]],
    folds: int,
    seed: int,
) -> list[tuple[int, str, int]]:
    rng = random.Random(seed)
    folded: list[tuple[int, str, int]] = []
    by_label: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for row in rows:
        by_label[row[0]].append(row)
    for label in sorted(by_label):
        group = list(by_label[label])
        rng.shuffle(group)
        for index, (row_label, text) in enumerate(group):
            folded.append((row_label, text, index % folds))
    rng.shuffle(folded)
    return folded


def write_tsv(path: Path, rows: Iterable[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for label, text in rows:
            writer.writerow([label, text])
    print(f"wrote {path}")


def write_fold_tsv(path: Path, rows: Iterable[tuple[int, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for label, text, fold in rows:
            writer.writerow([label, text, fold])
    print(f"wrote {path}")


def write_fold_splits(output_dir: Path, rows: list[tuple[int, str, int]], folds: int) -> dict[str, dict[str, int]]:
    fold_summary: dict[str, dict[str, int]] = {}
    for fold in range(folds):
        valid_rows = [(label, text) for label, text, row_fold in rows if row_fold == fold]
        train_rows = [(label, text) for label, text, row_fold in rows if row_fold != fold]
        fold_dir = output_dir / "folds" / f"fold_{fold}"
        write_tsv(fold_dir / "train.tsv", train_rows)
        write_tsv(fold_dir / "valid.tsv", valid_rows)
        fold_summary[str(fold)] = {
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
        }
    return fold_summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_train_rows = read_agnews_jsonl(args.train_jsonl)
    raw_test_rows = read_agnews_jsonl(args.test_jsonl)
    train_rows, train_dedupe_stats = dedupe_rows(
        raw_train_rows,
        keep_conflicting_duplicates=args.keep_conflicting_duplicates,
    )
    test_rows = raw_test_rows
    test_dedupe_stats = None
    if args.dedupe_test:
        test_rows, test_dedupe_stats = dedupe_rows(raw_test_rows)

    train_split, valid_split = stratified_split(train_rows, args.valid_ratio, args.seed)
    folded_rows = make_stratified_folds(train_rows, args.folds, args.seed)

    write_tsv(output_dir / "agnews_full_train.tsv", train_rows)
    write_tsv(output_dir / "agnews_train.tsv", train_split)
    write_tsv(output_dir / "agnews_valid.tsv", valid_split)
    write_tsv(output_dir / "agnews_test.tsv", test_rows)
    write_fold_tsv(output_dir / "agnews_full_train_folds.tsv", folded_rows)
    fold_split_summary = write_fold_splits(output_dir, folded_rows, args.folds)

    report = {
        "label_names": LABEL_NAMES,
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "folds": args.folds,
        "raw_train_rows": len(raw_train_rows),
        "raw_test_rows": len(raw_test_rows),
        "clean_changed_train_rows": count_changed(args.train_jsonl),
        "clean_changed_test_rows": count_changed(args.test_jsonl),
        "train_dedupe": train_dedupe_stats,
        "test_dedupe": test_dedupe_stats,
        "full_train_rows": len(train_rows),
        "train_rows": len(train_split),
        "valid_rows": len(valid_split),
        "test_rows": len(test_rows),
        "fold_split_rows": fold_split_summary,
        "full_train_label_counts": label_counts(train_rows),
        "train_label_counts": label_counts(train_split),
        "valid_label_counts": label_counts(valid_split),
        "test_label_counts": label_counts(test_rows),
    }
    (output_dir / "cleaning_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
