"""Transformer feed-forward networks."""

from __future__ import annotations

from torch import nn


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)
