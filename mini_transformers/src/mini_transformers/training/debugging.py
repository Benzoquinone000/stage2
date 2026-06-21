"""Debug helpers for tensor shape and gradient checks."""

from __future__ import annotations

import torch


def describe_batch(batch: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in batch.items()}


def has_nan_or_inf(tensor: torch.Tensor) -> bool:
    return bool(torch.isnan(tensor).any() or torch.isinf(tensor).any())
