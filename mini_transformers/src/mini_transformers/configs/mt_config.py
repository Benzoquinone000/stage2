"""Transformer machine translation configuration."""

from dataclasses import dataclass

from .configuration_utils import PretrainedConfig


@dataclass
class TransformerMTConfig(PretrainedConfig):
    model_type: str = "transformer_mt"
    src_vocab_size: int = 32000
    tgt_vocab_size: int = 32000
    hidden_size: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
