"""GPT-2 task heads."""

from __future__ import annotations

from torch import nn


class CausalLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.proj(hidden_states)
