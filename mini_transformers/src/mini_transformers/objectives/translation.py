"""Machine translation losses."""

from __future__ import annotations

import torch

from .losses import token_cross_entropy


def seq2seq_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return token_cross_entropy(logits, labels, ignore_index=-100)
