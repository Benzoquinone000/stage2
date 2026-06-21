"""Tensor helpers."""

from __future__ import annotations

import torch


def shift_tokens_right(input_ids: torch.Tensor, start_token_id: int, pad_token_id: int) -> torch.Tensor:
    shifted = input_ids.new_full(input_ids.shape, pad_token_id)
    shifted[:, 1:] = input_ids[:, :-1]
    shifted[:, 0] = start_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted
