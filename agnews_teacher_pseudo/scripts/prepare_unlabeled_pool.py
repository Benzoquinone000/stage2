"""Prepare a deduplicated unlabeled news pool for teacher pseudo-labeling."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGNEWS_SRC = ROOT / "agnews_classification" / "src"
sys.path.insert(0, str(AGNEWS_SRC))

from agnews_classification.text_cleaning import clean_text, dedupe_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlm-train-file",
        default="../agnews_classification/data/processed_clean/mlm_news_augmented_train.tsv",
    )
    parser.add_argument(
        "--mlm-valid-file",
        default="../agnews_classification/data/processed_clean/mlm_news_augmented_valid.tsv",
    )
    parser.add_argument(
        "--exclude-files",
        nargs="*",
        default=[
            "../agnews_classification/data/processed_clean/agnews_full_train.tsv",
            "../agnews_classification/data/processed_clean/agnews_test.tsv",
        ],
    )
    parser.add_argument("--output-file", default="data/unlabeled_news_pool.tsv")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def read_text_column(path: str | Path) -> list[str]:
    texts = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                text = clean_text(row[1])
                if text:
                    texts.append(text)
    return texts


def main() -> None:
    args = parse_args()
    excluded_keys = set()
    for path in args.exclude_files:
        if not Path(path).exists():
            continue
        for text in read_text_column(path):
            excluded_keys.add(dedupe_key(text))

    candidates = []
    for path in [args.mlm_train_file, args.mlm_valid_file]:
        candidates.extend(read_text_column(path))

    seen = set(excluded_keys)
    selected = []
    skipped_short = 0
    skipped_duplicate = 0
    for text in candidates:
        word_count = len(text.split())
        if word_count < args.min_words or word_count > args.max_words:
            skipped_short += 1
            continue
        key = dedupe_key(text)
        if key in seen:
            skipped_duplicate += 1
            continue
        seen.add(key)
        selected.append(text)
        if args.max_examples is not None and len(selected) >= args.max_examples:
            break

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "text"])
        for idx, text in enumerate(selected):
            writer.writerow([idx, text])

    print(f"excluded labeled/test keys: {len(excluded_keys):,}")
    print(f"candidate texts: {len(candidates):,}")
    print(f"skipped by length: {skipped_short:,}")
    print(f"skipped duplicates/excluded: {skipped_duplicate:,}")
    print(f"wrote {len(selected):,} rows: {output_file}")


if __name__ == "__main__":
    main()
