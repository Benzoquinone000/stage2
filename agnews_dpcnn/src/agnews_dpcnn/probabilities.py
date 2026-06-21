"""Probability ensembling and blend-search helpers."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from .metrics import accuracy_from_preds, macro_f1, probability_nll


def evaluate_probs(probs: torch.Tensor, labels: torch.Tensor, prefix: str = "test") -> dict[str, float]:
    preds = probs.argmax(dim=-1)
    return {
        f"{prefix}_loss": probability_nll(probs, labels),
        f"{prefix}_accuracy": accuracy_from_preds(preds, labels),
        f"{prefix}_macro_f1": macro_f1(preds, labels),
    }


def average_probs(probs_list: list[torch.Tensor]) -> torch.Tensor:
    if not probs_list:
        raise ValueError("probs_list must not be empty")
    return sum(probs_list) / len(probs_list)


def iter_simplex_weights(num_models: int, units: int) -> Iterator[tuple[float, ...]]:
    def visit(prefix: list[int], remaining: int, slots_left: int):
        if slots_left == 1:
            yield tuple([*prefix, remaining])
            return
        for value in range(remaining + 1):
            yield from visit([*prefix, value], remaining - value, slots_left - 1)

    for row in visit([], units, num_models):
        yield tuple(value / units for value in row)


def validate_step(step: float) -> int:
    if step <= 0 or step > 1:
        raise ValueError("--step must be in (0, 1]")
    units_float = 1.0 / step
    units = round(units_float)
    if abs(units - units_float) > 1e-6:
        raise ValueError("--step must divide 1.0 exactly, e.g. 0.01, 0.02, 0.05")
    return units

