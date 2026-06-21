"""Text generation helpers."""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_search(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    attention_mask: torch.Tensor | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    generated = input_ids
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        next_token_logits = outputs["logits"][:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=-1)
        if eos_token_id is not None and (next_tokens == eos_token_id).all():
            break
    return generated
