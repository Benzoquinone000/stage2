"""BLEU metric wrapper."""

from __future__ import annotations


def corpus_bleu(predictions: list[str], references: list[str]) -> float:
    import sacrebleu

    return sacrebleu.corpus_bleu(predictions, [references]).score
