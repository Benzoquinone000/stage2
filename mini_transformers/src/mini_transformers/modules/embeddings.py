"""Embedding layers."""

from __future__ import annotations

import torch
from torch import nn


class TokenPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        pad_token_id: int = 0,
        dropout: float = 0.1,
        type_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = (
            nn.Embedding(type_vocab_size, hidden_size) if type_vocab_size is not None else None
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        embeddings = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)
        return self.dropout(self.layer_norm(embeddings))
