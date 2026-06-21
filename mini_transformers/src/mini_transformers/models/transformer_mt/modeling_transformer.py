"""A small Transformer encoder-decoder for machine translation."""

from __future__ import annotations

import torch
from torch import nn

from mini_transformers.configs import TransformerMTConfig
from mini_transformers.models.modeling_utils import PreTrainedModel
from mini_transformers.modules import (
    TokenPositionEmbedding,
    TransformerDecoderBlock,
    TransformerEncoderBlock,
)

from .heads import Seq2SeqLMHead


def build_decoder_mask(
    decoder_attention_mask: torch.Tensor | None,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.long))
    causal_mask = causal_mask[None, None, :, :]
    if decoder_attention_mask is None:
        return causal_mask
    padding_mask = decoder_attention_mask[:, None, None, :]
    return causal_mask * padding_mask


class TransformerForMachineTranslation(PreTrainedModel):
    config_class = TransformerMTConfig

    def __init__(self, config: TransformerMTConfig) -> None:
        super().__init__(config)
        self.src_embeddings = TokenPositionEmbedding(
            config.src_vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
        )
        self.tgt_embeddings = TokenPositionEmbedding(
            config.tgt_vocab_size,
            config.hidden_size,
            config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
        )
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                    config.dropout,
                    config.layer_norm_eps,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.hidden_size,
                    config.num_attention_heads,
                    config.intermediate_size,
                    config.dropout,
                    config.layer_norm_eps,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = Seq2SeqLMHead(config.hidden_size, config.tgt_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_hidden_states = self.src_embeddings(input_ids)
        for layer in self.encoder_layers:
            encoder_hidden_states, _ = layer(encoder_hidden_states, attention_mask)

        decoder_hidden_states = self.tgt_embeddings(decoder_input_ids)
        decoder_mask = build_decoder_mask(
            decoder_attention_mask,
            decoder_input_ids.size(1),
            decoder_input_ids.device,
        )
        for layer in self.decoder_layers:
            decoder_hidden_states, _, _ = layer(
                decoder_hidden_states,
                encoder_hidden_states,
                self_attention_mask=decoder_mask,
                encoder_attention_mask=attention_mask,
            )

        logits = self.lm_head(self.final_layer_norm(decoder_hidden_states))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return result
