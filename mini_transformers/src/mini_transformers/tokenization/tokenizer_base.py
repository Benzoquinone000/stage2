"""Base tokenizer interfaces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EncodedInput:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int] | None = None


class BaseTokenizer:
    pad_token = "[PAD]"
    unk_token = "[UNK]"
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    mask_token = "[MASK]"

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab.id_for_token(token, self.unk_token) for token in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.vocab.token_for_id(idx) for idx in ids]

    def encode(
        self,
        text: str,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> EncodedInput:
        tokens = self.tokenize(text)
        if max_length is not None and add_special_tokens:
            tokens = tokens[: max(0, max_length - 2)]
        if add_special_tokens:
            tokens = [self.cls_token, *tokens, self.sep_token]
        elif max_length is not None:
            tokens = tokens[:max_length]
        input_ids = self.convert_tokens_to_ids(tokens)
        return EncodedInput(input_ids=input_ids, attention_mask=[1] * len(input_ids))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            specials = {
                self.pad_token,
                self.unk_token,
                self.cls_token,
                self.sep_token,
                self.mask_token,
            }
            tokens = [token for token in tokens if token not in specials]
        return " ".join(tokens)
