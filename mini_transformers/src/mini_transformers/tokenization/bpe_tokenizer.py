"""A small byte-pair encoding tokenizer."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .basic_tokenizer import BasicTokenizer
from .vocab import Vocab


END_WORD = "</w>"


class BPETokenizer(BasicTokenizer):
    def __init__(
        self,
        vocab: Vocab,
        merges: list[tuple[str, str]] | None = None,
        do_lower_case: bool = True,
    ) -> None:
        super().__init__(vocab=vocab, do_lower_case=do_lower_case)
        self.merges = merges or []
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def train(
        cls,
        texts: list[str],
        vocab_size: int = 1000,
        min_pair_freq: int = 2,
        special_tokens: list[str] | None = None,
        do_lower_case: bool = True,
    ) -> "BPETokenizer":
        special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        basic = BasicTokenizer(Vocab(["[PAD]", "[UNK]"]), do_lower_case=do_lower_case)
        words = Counter()
        for text in texts:
            for token in basic.tokenize(text):
                words[tuple(token) + (END_WORD,)] += 1

        merges: list[tuple[str, str]] = []
        while len(cls._pieces_from_words(words, special_tokens)) < vocab_size:
            pairs = cls._count_pairs(words)
            if not pairs:
                break
            best_pair, best_count = pairs.most_common(1)[0]
            if best_count < min_pair_freq:
                break
            merges.append(best_pair)
            words = cls._merge_pair(words, best_pair)

        pieces = cls._pieces_from_words(words, special_tokens)
        return cls(Vocab(pieces), merges=merges, do_lower_case=do_lower_case)

    @classmethod
    def from_files(cls, vocab_file: str | Path, merges_file: str | Path) -> "BPETokenizer":
        merges = []
        with Path(merges_file).open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
        return cls(Vocab.from_file(vocab_file), merges=merges)

    def save_merges(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for left, right in self.merges:
                f.write(f"{left}\t{right}\n")

    def tokenize(self, text: str) -> list[str]:
        pieces = []
        for token in super().tokenize(text):
            pieces.extend(self._tokenize_word(token))
        return pieces

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens()]
        text = ""
        for token in tokens:
            if token.endswith(END_WORD):
                text += token[: -len(END_WORD)] + " "
            else:
                text += token
        return text.strip()

    def special_tokens(self) -> set[str]:
        return {self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token}

    def _tokenize_word(self, token: str) -> list[str]:
        symbols = tuple(token) + (END_WORD,)
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self.merge_ranks[pair], pair) for pair in pairs if pair in self.merge_ranks]
            if not ranked:
                break
            _, best_pair = min(ranked)
            symbols = self._merge_symbols(symbols, best_pair)
        pieces = self._output_pieces(symbols)
        return [piece if piece in self.vocab else self.unk_token for piece in pieces]

    @staticmethod
    def _count_pairs(words: Counter[tuple[str, ...]]) -> Counter[tuple[str, str]]:
        pairs = Counter()
        for symbols, count in words.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += count
        return pairs

    @classmethod
    def _merge_pair(
        cls,
        words: Counter[tuple[str, ...]],
        pair: tuple[str, str],
    ) -> Counter[tuple[str, ...]]:
        merged_words = Counter()
        for symbols, count in words.items():
            merged_words[cls._merge_symbols(symbols, pair)] += count
        return merged_words

    @staticmethod
    def _merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        merged = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return tuple(merged)

    @classmethod
    def _pieces_from_words(cls, words: Counter[tuple[str, ...]], special_tokens: list[str]) -> list[str]:
        pieces = set()
        for symbols in words:
            pieces.update(cls._output_pieces(symbols))
        pieces.discard("")
        pieces = {piece for piece in pieces if piece not in special_tokens}
        return [*special_tokens, *sorted(pieces)]

    @staticmethod
    def _output_pieces(symbols: tuple[str, ...]) -> list[str]:
        pieces: list[str] = []
        for symbol in symbols:
            if symbol == END_WORD:
                if pieces:
                    pieces[-1] += END_WORD
            else:
                pieces.append(symbol)
        return pieces
