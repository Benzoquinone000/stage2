"""GPT-2 configuration."""

from dataclasses import dataclass

from .configuration_utils import PretrainedConfig


@dataclass
class GPT2Config(PretrainedConfig):
    model_type: str = "gpt2"
    vocab_size: int = 50257
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    resid_dropout: float = 0.1
    embd_dropout: float = 0.1
    attn_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
