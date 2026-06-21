"""BERT configuration."""

from dataclasses import dataclass

from .configuration_utils import PretrainedConfig


@dataclass
class BertConfig(PretrainedConfig):
    model_type: str = "bert"
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    num_labels: int = 2
