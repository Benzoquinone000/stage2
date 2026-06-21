"""Optimizer builders."""

from __future__ import annotations

import torch


def build_adamw(model, learning_rate: float, weight_decay: float = 0.0):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
