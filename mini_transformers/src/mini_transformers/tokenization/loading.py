"""Tokenizer save/load helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .basic_tokenizer import BasicTokenizer
from .bpe_tokenizer import BPETokenizer
from .vocab import Vocab
from .wordpiece_tokenizer import WordPieceTokenizer


def save_tokenizer(tokenizer, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.vocab.save(output_dir / "vocab.txt")
    config = {"type": "basic"}
    if isinstance(tokenizer, BPETokenizer):
        tokenizer.save_merges(output_dir / "merges.txt")
        config["type"] = "bpe"
    elif isinstance(tokenizer, WordPieceTokenizer):
        config["type"] = "wordpiece"
    with (output_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_tokenizer(model_dir: str | Path):
    model_dir = Path(model_dir)
    config_path = model_dir / "tokenizer_config.json"
    tokenizer_type = "bpe" if (model_dir / "merges.txt").exists() else "basic"
    if config_path.exists():
        tokenizer_type = json.loads(config_path.read_text(encoding="utf-8")).get("type", tokenizer_type)
    if tokenizer_type == "bpe":
        return BPETokenizer.from_files(model_dir / "vocab.txt", model_dir / "merges.txt")
    if tokenizer_type == "wordpiece":
        return WordPieceTokenizer.from_file(model_dir / "vocab.txt")
    return BasicTokenizer(Vocab.from_file(model_dir / "vocab.txt"))
