"""Download and convert a few small public NLP datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import urllib.request
import zipfile


URLS = {
    "tiny_shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "sms_spam": "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    "manythings_en_de": "http://www.manythings.org/anki/deu-eng.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(URLS))
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-examples", type=int, default=2000)
    return parser.parse_args()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "mini-transformers"})
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())


def write_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def split_rows(rows: list[list[str]], valid_ratio: float = 0.1) -> tuple[list[list[str]], list[list[str]]]:
    split = int(len(rows) * (1 - valid_ratio))
    return rows[:split], rows[split:]


def prepare_tiny_shakespeare(output_dir: Path) -> None:
    download(URLS["tiny_shakespeare"], output_dir / "tiny_shakespeare.txt")


def prepare_sms_spam(output_dir: Path, max_examples: int) -> None:
    zip_path = output_dir / "sms_spam.zip"
    download(URLS["sms_spam"], zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        lines = zf.read("SMSSpamCollection").decode("utf-8").splitlines()

    rows = []
    for line in lines[:max_examples]:
        label, text = line.split("\t", 1)
        rows.append(["1" if label == "spam" else "0", text])

    train_rows, valid_rows = split_rows(rows)
    write_rows(output_dir / "sms_spam_train.tsv", train_rows)
    write_rows(output_dir / "sms_spam_valid.tsv", valid_rows)


def prepare_manythings_en_de(output_dir: Path, max_examples: int) -> None:
    zip_path = output_dir / "deu-eng.zip"
    download(URLS["manythings_en_de"], zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        filename = next(name for name in zf.namelist() if name.endswith(".txt"))
        lines = zf.read(filename).decode("utf-8").splitlines()

    rows = []
    for line in lines[:max_examples]:
        english, german, *_ = line.split("\t")
        rows.append([english, german])

    train_rows, valid_rows = split_rows(rows)
    write_rows(output_dir / "manythings_en_de_train.tsv", train_rows)
    write_rows(output_dir / "manythings_en_de_valid.tsv", valid_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.dataset

    if args.dataset == "tiny_shakespeare":
        prepare_tiny_shakespeare(output_dir)
    elif args.dataset == "sms_spam":
        prepare_sms_spam(output_dir, args.max_examples)
    elif args.dataset == "manythings_en_de":
        prepare_manythings_en_de(output_dir, args.max_examples)

    print(f"prepared {args.dataset} in {output_dir}")


if __name__ == "__main__":
    main()
