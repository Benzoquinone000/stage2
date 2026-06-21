"""Build a larger news-domain MLM corpus from local news sources."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys


TASK_DIR = Path(__file__).resolve().parents[1]
TASK_SRC = TASK_DIR / "src"
sys.path.insert(0, str(TASK_SRC))

from agnews_classification.text_cleaning import clean_text, dedupe_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=220)
    return parser.parse_args()


def read_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return texts
        text_idx = 1
        if "text" in header:
            text_idx = header.index("text")
        elif header[0].isdigit():
            texts.append(clean_text(header[1] if len(header) > 1 else ""))
        for row in reader:
            if len(row) > text_idx:
                texts.append(clean_text(row[text_idx]))
    print(f"loaded {len(texts):,}: {path}")
    return texts


def write_tsv(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for text in texts:
            writer.writerow([0, text])
    print(f"wrote {len(texts):,}: {path}")


def main() -> None:
    args = parse_args()
    texts: list[str] = []
    for input_path in args.inputs:
        texts.extend(read_texts(Path(input_path)))

    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        words = text.split()
        if len(words) < args.min_words or len(words) > args.max_words:
            continue
        key = dedupe_key(text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)

    rng = random.Random(args.seed)
    rng.shuffle(unique)
    valid_size = int(len(unique) * args.valid_ratio)
    output_dir = Path(args.output_dir)
    write_tsv(output_dir / "mlm_news_ultra_train.tsv", unique[valid_size:])
    write_tsv(output_dir / "mlm_news_ultra_valid.tsv", unique[:valid_size])


if __name__ == "__main__":
    main()
