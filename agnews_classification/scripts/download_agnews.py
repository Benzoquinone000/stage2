"""Download and prepare AG News for text classification.

The script downloads JSONL files from Hugging Face and converts them to TSV:

    label<TAB>title + description

Labels are converted to zero-based ids.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import urllib.request


URLS = {
    "train": (
        "https://hf-mirror.com/datasets/sh0416/ag_news/resolve/main/train.jsonl",
        "https://huggingface.co/datasets/sh0416/ag_news/resolve/main/train.jsonl",
    ),
    "test": (
        "https://hf-mirror.com/datasets/sh0416/ag_news/resolve/main/test.jsonl",
        "https://huggingface.co/datasets/sh0416/ag_news/resolve/main/test.jsonl",
    ),
}

CSV_URLS = {
    "train": "https://cdn.jsdelivr.net/gh/mhjabreel/CharCnn_Keras@master/data/ag_news_csv/train.csv",
    "test": "https://cdn.jsdelivr.net/gh/mhjabreel/CharCnn_Keras@master/data/ag_news_csv/test.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(url: str, path: Path, force: bool = False) -> bool:
    if path.exists() and not force:
        print(f"exists: {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading: {url}")
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        with urllib.request.urlopen(url, timeout=30) as response, tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp_path.replace(path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"download failed: {url} ({exc})")
        return False
    print(f"saved: {path}")
    return True


def normalize_label(value) -> int:
    label = int(value)
    return label - 1 if label in {1, 2, 3, 4} else label


def normalize_text(*parts: str) -> str:
    text = " ".join(part.strip() for part in parts if part and part.strip())
    return " ".join(text.split())


def read_jsonl(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            label = normalize_label(item.get("label"))
            title = item.get("title", "")
            description = item.get("description", item.get("text", ""))
            rows.append((label, normalize_text(title, description)))
    return rows


def read_csv_rows(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            label = normalize_label(row[0])
            rows.append((label, normalize_text(row[1], row[2])))
    return rows


def write_jsonl(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for label, text in rows:
            json.dump({"label": label, "text": text}, f, ensure_ascii=False)
            f.write("\n")
    print(f"wrote {len(rows):,} rows: {path}")


def load_split(split: str, raw_dir: Path, force: bool = False) -> list[tuple[int, str]]:
    jsonl_path = raw_dir / f"{split}.jsonl"
    for url in URLS[split]:
        if download(url, jsonl_path, force=force):
            return read_jsonl(jsonl_path)

    print(f"falling back to CSV source for {split}")
    csv_path = raw_dir / f"{split}.csv"
    if not download(CSV_URLS[split], csv_path, force=force):
        raise RuntimeError(f"failed to download AG News {split} split")
    rows = read_csv_rows(csv_path)
    write_jsonl(jsonl_path, rows)
    return rows


def split_train_valid(
    rows: list[tuple[int, str]],
    valid_ratio: float,
    seed: int,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    valid_size = int(len(shuffled) * valid_ratio)
    valid = shuffled[:valid_size]
    train = shuffled[valid_size:]
    return train, valid


def write_tsv(path: Path, rows: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for label, text in rows:
            writer.writerow([label, text])
    print(f"wrote {len(rows):,} rows: {path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"

    train_rows = load_split("train", raw_dir, force=args.force)
    test_rows = load_split("test", raw_dir, force=args.force)
    train_rows, valid_rows = split_train_valid(train_rows, args.valid_ratio, args.seed)

    write_tsv(processed_dir / "agnews_train.tsv", train_rows)
    write_tsv(processed_dir / "agnews_valid.tsv", valid_rows)
    write_tsv(processed_dir / "agnews_test.tsv", test_rows)

    print("label mapping: 0=World, 1=Sports, 2=Business, 3=Sci/Tech")


if __name__ == "__main__":
    main()
