"""Shared TSV loading, tokenization, vocabulary, and dataset helpers."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?|[^\w\s]", re.UNICODE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class TextRow:
    label: int
    text: str
    teacher_probs: list[float] | None = None
    is_pseudo: bool = False


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def read_tsv(path: str | Path, max_examples: int | None = None) -> list[TextRow]:
    rows: list[TextRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header: list[str] | None = None
        for raw_row in reader:
            if len(raw_row) < 2:
                continue
            if header is None and raw_row[0] == "label":
                header = raw_row
                continue
            if header is None:
                if raw_row[0].isdigit():
                    rows.append(TextRow(label=int(raw_row[0]), text=raw_row[1]))
            else:
                values = {key: raw_row[idx] if idx < len(raw_row) else "" for idx, key in enumerate(header)}
                teacher_probs = None
                if all(values.get(f"prob_{class_id}", "") != "" for class_id in range(4)):
                    teacher_probs = [float(values[f"prob_{class_id}"]) for class_id in range(4)]
                    total = sum(teacher_probs)
                    if total > 0:
                        teacher_probs = [value / total for value in teacher_probs]
                rows.append(
                    TextRow(
                        label=int(values["label"]),
                        text=values["text"],
                        teacher_probs=teacher_probs,
                        is_pseudo=values.get("is_pseudo", "0") == "1",
                    )
                )
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def build_vocab(rows: list[TextRow], max_size: int, min_freq: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(tokenize(row.text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def save_vocab(path: Path, vocab: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda item: item[1]):
            f.write(json.dumps({"token": token, "id": idx}, ensure_ascii=False) + "\n")


def load_vocab(path: str | Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            vocab[row["token"]] = int(row["id"])
    return vocab


def encode(text: str, vocab: dict[str, int], max_length: int) -> tuple[list[int], int]:
    ids = [vocab.get(token, 1) for token in tokenize(text)] or [1]
    ids = ids[:max_length]
    length = len(ids)
    if length < max_length:
        ids.extend([0] * (max_length - length))
    return ids, length


class AGNewsDataset(Dataset):
    def __init__(self, rows: list[TextRow], vocab: dict[str, int], max_length: int, include_lengths: bool) -> None:
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length
        self.include_lengths = include_lengths

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        input_ids, length = encode(row.text, self.vocab, self.max_length)
        teacher_probs = row.teacher_probs
        if teacher_probs is None:
            teacher_probs = [1.0 if class_id == row.label else 0.0 for class_id in range(4)]
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(row.label, dtype=torch.long),
            "teacher_probs": torch.tensor(teacher_probs, dtype=torch.float),
            "teacher_mask": torch.tensor(1 if row.is_pseudo and row.teacher_probs is not None else 0, dtype=torch.bool),
        }
        if self.include_lengths:
            item["lengths"] = torch.tensor(length, dtype=torch.long)
        return item


def make_loader(
    rows: list[TextRow],
    vocab: dict[str, int],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    include_lengths: bool = False,
) -> DataLoader:
    return DataLoader(
        AGNewsDataset(rows, vocab, max_length, include_lengths=include_lengths),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

