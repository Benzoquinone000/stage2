"""GPT-2-style decoder-only models."""

from __future__ import annotations

import torch
from torch import nn

from mini_transformers.configs import GPT2Config
from mini_transformers.models.modeling_utils import PreTrainedModel
from mini_transformers.modules import TokenPositionEmbedding, TransformerDecoderOnlyBlock

from .heads import CausalLMHead


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.long))[None, None, :, :]


class GPT2Model(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.embeddings = TokenPositionEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            dropout=config.embd_dropout,
        )
        self.layers = nn.ModuleList(
            [
                TransformerDecoderOnlyBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.resid_dropout,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, seq_len = input_ids.shape
        causal_mask = build_causal_mask(seq_len, input_ids.device)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask * causal_mask
        else:
            attention_mask = causal_mask
        hidden_states = self.embeddings(input_ids)
        attentions = []
        for layer in self.layers:
            hidden_states, attention = layer(hidden_states, attention_mask)
            attentions.append(attention)
        return {
            "last_hidden_state": self.final_layer_norm(hidden_states),
            "attentions": attentions,
        }


class GPT2ForCausalLM(PreTrainedModel):
    config_class = GPT2Config

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = CausalLMHead(config.hidden_size, config.vocab_size)
        self.lm_head.proj.weight = self.transformer.embeddings.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.transformer(input_ids, attention_mask)
        logits = self.lm_head(outputs["last_hidden_state"])
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return result
