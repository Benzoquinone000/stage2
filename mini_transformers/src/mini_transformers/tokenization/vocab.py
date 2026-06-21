"""Vocabulary helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path


class Vocab:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def id_for_token(self, token: str, unk_token: str = "[UNK]") -> int:
        return self.token_to_id.get(token, self.token_to_id[unk_token])

    def token_for_id(self, idx: int) -> str:
        return self.tokens[idx]

    @classmethod
    def from_tokens(
        cls,
        tokens: list[str],
        special_tokens: list[str] | None = None,
        min_freq: int = 1,
    ) -> "Vocab":
        special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        counts = Counter(tokens)
        words = sorted(token for token, count in counts.items() if count >= min_freq)
        words = [word for word in words if word not in special_tokens]
        return cls([*special_tokens, *words])

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocab":
        with Path(path).open("r", encoding="utf-8") as f:
            tokens = [line.rstrip("\n") for line in f if line.strip()]
        return cls(tokens)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            f.write("\n".join(self.tokens))
