import torch

from mini_transformers.modules.attention import MultiHeadSelfAttention


def test_attention_shape():
    layer = MultiHeadSelfAttention(hidden_size=8, num_heads=2)
    output, probs = layer(torch.randn(2, 3, 8))
    assert output.shape == (2, 3, 8)
    assert probs.shape == (2, 2, 3, 3)
