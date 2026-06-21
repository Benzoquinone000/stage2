"""BERT task heads."""

from __future__ import annotations

import torch
from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        return self.out_proj(self.dropout(pooled_output))


class MaskedLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, layer_norm_eps: float = 1e-12) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.functional.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.decoder(hidden_states)


class NextSentenceHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output):
        return self.classifier(pooled_output)
