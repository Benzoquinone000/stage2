"""A minimal WordPiece tokenizer."""

from __future__ import annotations

from collections import Counter

from .basic_tokenizer import BasicTokenizer
from .vocab import Vocab


class WordPieceTokenizer(BasicTokenizer):
    def __init__(
        self,
        vocab: Vocab,
        do_lower_case: bool = True,
        max_input_chars_per_word: int = 100,
    ) -> None:
        super().__init__(vocab=vocab, do_lower_case=do_lower_case)
        self.max_input_chars_per_word = max_input_chars_per_word

    @classmethod
    def from_file(cls, path: str) -> "WordPieceTokenizer":
        return cls(Vocab.from_file(path))

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 1000,
        min_freq: int = 1,
        do_lower_case: bool = True,
    ) -> "WordPieceTokenizer":
        basic = BasicTokenizer(Vocab(["[PAD]", "[UNK]"]), do_lower_case=do_lower_case)
        counts = Counter(token for text in texts for token in basic.tokenize(text))
        pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        char_pieces = set()
        for token in counts:
            if not token:
                continue
            char_pieces.add(token[0])
            char_pieces.update("##" + char for char in token[1:])
        full_words = [token for token, count in counts.most_common() if count >= min_freq]
        for piece in [*sorted(char_pieces), *full_words]:
            if piece not in pieces:
                pieces.append(piece)
            if len(pieces) >= vocab_size:
                break
        return cls(Vocab(pieces), do_lower_case=do_lower_case)

    def tokenize(self, text: str) -> list[str]:
        output_tokens: list[str] = []
        for token in super().tokenize(text):
            output_tokens.extend(self._tokenize_word(token))
        return output_tokens

    def _tokenize_word(self, token: str) -> list[str]:
        if len(token) > self.max_input_chars_per_word:
            return [self.unk_token]
        if token in self.vocab:
            return [token]

        chars = list(token)
        sub_tokens: list[str] = []
        start = 0
        while start < len(chars):
            end = len(chars)
            current_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    current_substr = substr
                    break
                end -= 1
            if current_substr is None:
                return [self.unk_token]
            sub_tokens.append(current_substr)
            start = end
        return sub_tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens()]

        words: list[str] = []
        for token in tokens:
            if token.startswith("##") and words:
                words[-1] += token[2:]
            else:
                words.append(token)
        return " ".join(words)

    def special_tokens(self) -> set[str]:
        return {self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token}
