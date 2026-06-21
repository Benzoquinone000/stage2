"""XLNet configuration."""

from dataclasses import dataclass

from .configuration_utils import PretrainedConfig


@dataclass
class XLNetConfig(PretrainedConfig):
    model_type: str = "xlnet"
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    mem_len: int = 0
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
