"""Common loss functions."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )
