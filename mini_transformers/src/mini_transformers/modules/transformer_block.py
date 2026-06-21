"""Transformer blocks."""

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .feed_forward import FeedForward


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_probs = self.attention(hidden_states, attention_mask)
        hidden_states = self.attn_layer_norm(hidden_states + self.dropout(attn_output))
        ffn_output = self.feed_forward(hidden_states)
        hidden_states = self.ffn_layer_norm(hidden_states + ffn_output)
        return hidden_states, attn_probs


class TransformerDecoderOnlyBlock(TransformerEncoderBlock):
    """Decoder-only block; causal masking is supplied by the caller."""


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cross_attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_output, self_probs = self.self_attention(hidden_states, self_attention_mask)
        hidden_states = self.self_attn_layer_norm(hidden_states + self.dropout(self_output))

        cross_output, cross_probs = self.cross_attention(
            hidden_states,
            encoder_attention_mask,
            key_value_states=encoder_hidden_states,
        )
        hidden_states = self.cross_attn_layer_norm(hidden_states + self.dropout(cross_output))

        ffn_output = self.feed_forward(hidden_states)
        hidden_states = self.ffn_layer_norm(hidden_states + ffn_output)
        return hidden_states, self_probs, cross_probs
