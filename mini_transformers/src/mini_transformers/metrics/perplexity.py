"""Perplexity metric."""

from __future__ import annotations

import math


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))
