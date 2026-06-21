"""Optional SentencePiece wrapper."""

from __future__ import annotations

from .tokenizer_base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_file: str) -> None:
        import sentencepiece as spm

        self.processor = spm.SentencePieceProcessor(model_file=model_file)

    def tokenize(self, text: str) -> list[str]:
        return self.processor.encode(text, out_type=str)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.processor.piece_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.processor.id_to_piece(idx) for idx in ids]
