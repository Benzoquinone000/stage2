"""Normalization helpers."""

from __future__ import annotations

from torch import nn


def build_layer_norm(hidden_size: int, eps: float = 1e-5) -> nn.LayerNorm:
    return nn.LayerNorm(hidden_size, eps=eps)
