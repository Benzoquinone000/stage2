"""BERT-style encoder models."""

from __future__ import annotations

import torch
from torch import nn

from mini_transformers.configs import BertConfig
from mini_transformers.models.modeling_utils import PreTrainedModel
from mini_transformers.modules import TokenPositionEmbedding, TransformerEncoderBlock

from .heads import ClassificationHead, MaskedLMHead, NextSentenceHead


class BertModel(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.embeddings = TokenPositionEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            pad_token_id=config.pad_token_id,
            dropout=config.hidden_dropout_prob,
            type_vocab_size=config.type_vocab_size,
        )
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.hidden_dropout_prob,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.embeddings(input_ids, token_type_ids)
        attentions = []
        for layer in self.layers:
            hidden_states, attention = layer(hidden_states, attention_mask)
            attentions.append(attention)
        pooled_output = self.pooler(hidden_states[:, 0])
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
            "attentions": attentions,
        }


class BertForSequenceClassification(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config: BertConfig, num_labels: int | None = None) -> None:
        super().__init__(config)
        self.num_labels = num_labels or config.num_labels
        self.bert = BertModel(config)
        self.classifier = ClassificationHead(
            config.hidden_size,
            self.num_labels,
            config.hidden_dropout_prob,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(outputs["pooler_output"])
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result


class BertForMaskedLM(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.mlm_head = MaskedLMHead(config.hidden_size, config.vocab_size, config.layer_norm_eps)
        self.mlm_head.decoder.weight = self.bert.embeddings.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.bert(input_ids, attention_mask, token_type_ids)["last_hidden_state"]
        logits = self.mlm_head(hidden_states)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return result


class BertForPreTraining(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = BertModel(config)
        self.mlm_head = MaskedLMHead(config.hidden_size, config.vocab_size, config.layer_norm_eps)
        self.nsp_head = NextSentenceHead(config.hidden_size)
        self.mlm_head.decoder.weight = self.bert.embeddings.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        next_sentence_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        prediction_logits = self.mlm_head(outputs["last_hidden_state"])
        relationship_logits = self.nsp_head(outputs["pooler_output"])
        result = {
            "logits": prediction_logits,
            "prediction_logits": prediction_logits,
            "seq_relationship_logits": relationship_logits,
        }
        losses = []
        if labels is not None:
            losses.append(
                nn.functional.cross_entropy(
                    prediction_logits.reshape(-1, prediction_logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            )
        if next_sentence_labels is not None:
            losses.append(nn.functional.cross_entropy(relationship_logits, next_sentence_labels))
        if losses:
            result["loss"] = sum(losses)
        return result
