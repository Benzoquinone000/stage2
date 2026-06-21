"""Multi-head self-attention implemented with basic PyTorch ops."""

from __future__ import annotations

from math import sqrt

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_value_states = hidden_states if key_value_states is None else key_value_states
        query = self.split_heads(self.q_proj(hidden_states))
        key = self.split_heads(self.k_proj(key_value_states))
        value = self.split_heads(self.v_proj(key_value_states))

        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]

        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)

        attention_probs = torch.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = self.merge_heads(context)
        return self.out_proj(context), attention_probs
