"""Prepare an augmented news-domain corpus for MLM training.

The output format matches the existing MLM script input:

    label<TAB>text

Labels are dummy zeros because masked language modeling only reads the text
column. The corpus combines AG News train text with external news-domain text
from UCI News Aggregator and HuffPost News Category.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import random
import sys
import urllib.request
import zipfile


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from task2_text_cleaning import clean_text, dedupe_key


UCI_URL = "https://archive.ics.uci.edu/static/public/359/news+aggregator.zip"
HUFFPOST_URL = "https://hf-mirror.com/datasets/heegyu/news-category-dataset/resolve/main/data.json"

HUFFPOST_KEEP_CATEGORIES = {
    "WORLD NEWS",
    "WORLDPOST",
    "THE WORLDPOST",
    "POLITICS",
    "SPORTS",
    "BUSINESS",
    "MONEY",
    "SCIENCE",
    "TECH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agnews-train", default="data/processed/agnews_train.tsv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw/mlm_external")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-uci-rows", type=int, default=300000)
    parser.add_argument("--max-uci-per-category", type=int, default=None)
    parser.add_argument("--uci-categories", default="b,t")
    parser.add_argument("--max-huffpost-rows", type=int, default=180000)
    parser.add_argument("--max-huffpost-per-category", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        print(f"exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"downloading: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "agnews-mlm-corpus"})
    with urllib.request.urlopen(request, timeout=60) as response, tmp_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(path)
    print(f"saved: {path}")


def normalize_text(*parts: str) -> str:
    return clean_text(*parts)


def read_agnews(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                texts.append(normalize_text(row[1]))
    print(f"loaded {len(texts):,} AG News texts")
    return texts


def read_uci_titles(
    zip_path: Path,
    allowed_categories: set[str],
    max_rows: int | None = None,
    max_per_category: int | None = None,
) -> list[str]:
    texts: list[str] = []
    category_counts: collections.Counter[str] = collections.Counter()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        csv_name = next(name for name in names if name.endswith("newsCorpora.csv"))
        with zf.open(csv_name) as raw:
            wrapper = (line.decode("utf-8", errors="ignore") for line in raw)
            reader = csv.reader(wrapper, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                category = row[4].strip().lower()
                if category not in allowed_categories:
                    continue
                if max_per_category is not None and category_counts[category] >= max_per_category:
                    continue
                text = normalize_text(row[1])
                if len(text.split()) >= 4:
                    texts.append(text)
                    category_counts[category] += 1
                if max_rows is not None and len(texts) >= max_rows:
                    break
    print(f"loaded {len(texts):,} UCI News Aggregator titles")
    print(f"UCI category counts: {dict(sorted(category_counts.items()))}")
    return texts


def read_huffpost(
    path: Path,
    max_rows: int | None = None,
    max_per_category: int | None = None,
) -> list[str]:
    texts: list[str] = []
    category_counts: collections.Counter[str] = collections.Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            category = item.get("category", "")
            if category not in HUFFPOST_KEEP_CATEGORIES:
                continue
            if max_per_category is not None and category_counts[category] >= max_per_category:
                continue
            text = normalize_text(item.get("headline", ""), item.get("short_description", ""))
            if len(text.split()) >= 6:
                texts.append(text)
                category_counts[category] += 1
            if max_rows is not None and len(texts) >= max_rows:
                break
    print(f"loaded {len(texts):,} HuffPost news texts")
    print(f"HuffPost category counts: {dict(sorted(category_counts.items()))}")
    return texts


def dedupe(texts: list[str], max_words: int | None = None) -> list[str]:
    seen = set()
    unique = []
    for text in texts:
        if max_words is not None and len(text.split()) > max_words:
            continue
        key = dedupe_key(text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    print(f"deduped corpus: {len(unique):,} texts")
    return unique


def write_tsv(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for text in texts:
            writer.writerow([0, text])
    print(f"wrote {len(texts):,} rows: {path}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    uci_zip = raw_dir / "news_aggregator.zip"
    huffpost_json = raw_dir / "huffpost_news_category.jsonl"
    download(UCI_URL, uci_zip, force=args.force)
    download(HUFFPOST_URL, huffpost_json, force=args.force)

    texts = []
    texts.extend(read_agnews(Path(args.agnews_train)))
    uci_categories = {category.strip().lower() for category in args.uci_categories.split(",") if category.strip()}
    texts.extend(
        read_uci_titles(
            uci_zip,
            allowed_categories=uci_categories,
            max_rows=args.max_uci_rows,
            max_per_category=args.max_uci_per_category,
        )
    )
    texts.extend(
        read_huffpost(
            huffpost_json,
            max_rows=args.max_huffpost_rows,
            max_per_category=args.max_huffpost_per_category,
        )
    )
    texts = dedupe(texts, max_words=args.max_words)

    rng = random.Random(args.seed)
    rng.shuffle(texts)
    valid_size = int(len(texts) * args.valid_ratio)
    valid = texts[:valid_size]
    train = texts[valid_size:]

    write_tsv(output_dir / "mlm_news_augmented_train.tsv", train)
    write_tsv(output_dir / "mlm_news_augmented_valid.tsv", valid)


if __name__ == "__main__":
    main()
