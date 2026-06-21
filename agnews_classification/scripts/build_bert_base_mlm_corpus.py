"""Build a larger corpus for BERT-base-from-scratch MLM pretraining.

The output format matches ``pretrain_bert_mlm.py``:

    0<TAB>text

This script deliberately downloads text corpora only. It does not download or
load any pretrained model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import tarfile
import urllib.request
import zipfile


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from task2_text_cleaning import clean_text, dedupe_key


WIKITEXT_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
DBPEDIA_URL = "https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz"
TEXT8_URL = "https://mattmahoney.net/dc/text8.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/processed_clean_bert_base_mlm")
    parser.add_argument("--raw-dir", default="data/raw/mlm_external")
    parser.add_argument("--inputs", nargs="*", default=[])
    parser.add_argument("--huffpost-jsonl", default="data/raw/mlm_external/huffpost_news_category.jsonl")
    parser.add_argument("--uci-zip", default="data/raw/mlm_external/news_aggregator.zip")
    parser.add_argument("--download-wikitext", action="store_true")
    parser.add_argument("--download-dbpedia", action="store_true")
    parser.add_argument("--download-text8", action="store_true")
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=220)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--text8-block-words", type=int, default=80)
    parser.add_argument("--force-download", action="store_true")
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
    request = urllib.request.Request(url, headers={"User-Agent": "agnews-bert-base-mlm"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp_path.replace(path)
    print(f"saved: {path}")


def maybe_add(texts: list[str], *parts: str) -> None:
    text = clean_text(*parts)
    if text:
        texts.append(text)


def read_tsv(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                maybe_add(texts, row[1])
    print(f"loaded {len(texts):,} TSV rows: {path}")
    return texts


def read_jsonl(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if "text" in item:
                maybe_add(texts, str(item.get("text", "")))
            elif "title" in item or "description" in item:
                maybe_add(texts, str(item.get("title", "")), str(item.get("description", "")))
            elif "headline" in item or "short_description" in item:
                maybe_add(texts, str(item.get("headline", "")), str(item.get("short_description", "")))
    print(f"loaded {len(texts):,} JSONL rows: {path}")
    return texts


def read_csv(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8", newline="", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) >= 3:
                maybe_add(texts, row[1], row[2])
            elif len(row) >= 2:
                maybe_add(texts, row[1])
            elif row:
                maybe_add(texts, row[0])
    print(f"loaded {len(texts):,} CSV rows: {path}")
    return texts


def read_huffpost(path: Path) -> list[str]:
    if not path.exists():
        print(f"missing HuffPost file: {path}")
        return []
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            maybe_add(texts, item.get("headline", ""), item.get("short_description", ""))
    print(f"loaded {len(texts):,} HuffPost rows: {path}")
    return texts


def read_uci_news(zip_path: Path) -> list[str]:
    if not zip_path.exists():
        print(f"missing UCI zip: {zip_path}")
        return []
    texts: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(name for name in zf.namelist() if name.endswith("newsCorpora.csv"))
        with zf.open(csv_name) as raw:
            wrapper = (line.decode("utf-8", errors="ignore") for line in raw)
            reader = csv.reader(wrapper, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    maybe_add(texts, row[1])
    print(f"loaded {len(texts):,} UCI News rows: {zip_path}")
    return texts


def read_wikitext(zip_path: Path) -> list[str]:
    if not zip_path.exists():
        print(f"missing WikiText zip: {zip_path}")
        return []
    texts: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            name for name in zf.namelist()
            if name.endswith(".raw") or name.endswith(".tokens")
        ]
        for name in sorted(names):
            if "wiki.test" in name:
                continue
            with zf.open(name) as raw:
                for line in raw:
                    text = line.decode("utf-8", errors="ignore").strip()
                    if not text or text.startswith("="):
                        continue
                    maybe_add(texts, text)
    print(f"loaded {len(texts):,} WikiText rows: {zip_path}")
    return texts


def read_dbpedia(tgz_path: Path) -> list[str]:
    if not tgz_path.exists():
        print(f"missing DBpedia tgz: {tgz_path}")
        return []
    texts: list[str] = []
    with tarfile.open(tgz_path, "r:gz") as tf:
        names = [
            member for member in tf.getmembers()
            if member.isfile() and member.name.endswith((".csv", ".txt"))
        ]
        for member in names:
            if member.name.endswith("test.csv"):
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            wrapper = (line.decode("utf-8", errors="ignore") for line in extracted)
            reader = csv.reader(wrapper)
            for row in reader:
                if len(row) >= 3:
                    maybe_add(texts, row[1], row[2])
    print(f"loaded {len(texts):,} DBpedia rows: {tgz_path}")
    return texts


def read_text8(zip_path: Path, block_words: int) -> list[str]:
    if not zip_path.exists():
        print(f"missing text8 zip: {zip_path}")
        return []
    texts: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        name = next(name for name in zf.namelist() if name.endswith("text8"))
        with zf.open(name) as raw:
            words = raw.read().decode("utf-8", errors="ignore").split()
    for start in range(0, len(words), block_words):
        block = words[start : start + block_words]
        if len(block) >= 8:
            maybe_add(texts, " ".join(block))
    print(f"loaded {len(texts):,} text8 blocks: {zip_path}")
    return texts


def read_path(path: Path) -> list[str]:
    if path.suffix == ".tsv":
        return read_tsv(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".csv":
        return read_csv(path)
    if path.suffix == ".zip":
        if "wiki" in path.name.lower():
            return read_wikitext(path)
        if "text8" in path.name.lower():
            return read_text8(path, block_words=80)
        return read_uci_news(path)
    if path.suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".tgz":
        return read_dbpedia(path)
    print(f"skipping unsupported input: {path}")
    return []


def filter_and_dedupe(
    texts: list[str],
    min_words: int,
    max_words: int,
    max_examples: int | None,
    seed: int,
) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        words = text.split()
        if len(words) < min_words or len(words) > max_words:
            continue
        key = dedupe_key(text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    rng = random.Random(seed)
    rng.shuffle(unique)
    if max_examples is not None:
        unique = unique[:max_examples]
    print(f"filtered/deduped corpus: {len(unique):,} rows")
    return unique


def write_tsv(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for text in texts:
            writer.writerow([0, text])
    print(f"wrote {len(texts):,}: {path}")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    if args.download_wikitext:
        download(WIKITEXT_URL, raw_dir / "wikitext-103-raw-v1.zip", args.force_download)
    if args.download_dbpedia:
        download(DBPEDIA_URL, raw_dir / "dbpedia_csv.tgz", args.force_download)
    if args.download_text8:
        download(TEXT8_URL, raw_dir / "text8.zip", args.force_download)

    texts: list[str] = []
    for input_path in args.inputs:
        texts.extend(read_path(Path(input_path)))
    texts.extend(read_huffpost(Path(args.huffpost_jsonl)))
    texts.extend(read_uci_news(Path(args.uci_zip)))
    texts.extend(read_wikitext(raw_dir / "wikitext-103-raw-v1.zip"))
    texts.extend(read_dbpedia(raw_dir / "dbpedia_csv.tgz"))
    texts.extend(read_text8(raw_dir / "text8.zip", args.text8_block_words))

    unique = filter_and_dedupe(
        texts,
        min_words=args.min_words,
        max_words=args.max_words,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    valid_size = int(len(unique) * args.valid_ratio)
    output_dir = Path(args.output_dir)
    write_tsv(output_dir / "mlm_bert_base_train.tsv", unique[valid_size:])
    write_tsv(output_dir / "mlm_bert_base_valid.tsv", unique[:valid_size])


if __name__ == "__main__":
    main()
