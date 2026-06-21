"""A simple rule-based tokenizer."""

from __future__ import annotations

import re

from .tokenizer_base import BaseTokenizer
from .vocab import Vocab


class BasicTokenizer(BaseTokenizer):
    def __init__(self, vocab: Vocab, do_lower_case: bool = True) -> None:
        self.vocab = vocab
        self.do_lower_case = do_lower_case

    def tokenize(self, text: str) -> list[str]:
        if self.do_lower_case:
            text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.id_for_token(token, self.unk_token) for token in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.vocab.token_for_id(idx) for idx in ids]
