import torch

from mini_transformers.configs import BertConfig, GPT2Config, XLNetConfig
from mini_transformers.configs import TransformerMTConfig
from mini_transformers.models import (
    BertForMaskedLM,
    BertForPreTraining,
    BertForSequenceClassification,
    GPT2ForCausalLM,
    TransformerForMachineTranslation,
    XLNetLMHeadModel,
)
from mini_transformers.models.xlnet import build_permutation_mask
from mini_transformers.models.xlnet.modeling_xlnet import prepend_memory_mask
from mini_transformers.modules import TransformerDecoderOnlyBlock


def test_small_bert_forward():
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = BertForSequenceClassification(config, num_labels=2)
    out = model(input_ids=torch.ones(2, 4, dtype=torch.long), labels=torch.tensor([0, 1]))
    assert out["logits"].shape == (2, 2)
    assert "loss" in out


def test_small_bert_mlm_forward():
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = BertForMaskedLM(config)
    out = model(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        labels=torch.ones(2, 4, dtype=torch.long),
    )
    assert out["logits"].shape == (2, 4, 32)
    assert "loss" in out


def test_small_bert_pretraining_forward():
    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = BertForPreTraining(config)
    out = model(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        labels=torch.ones(2, 4, dtype=torch.long),
        next_sentence_labels=torch.tensor([0, 1]),
    )
    assert out["prediction_logits"].shape == (2, 4, 32)
    assert out["seq_relationship_logits"].shape == (2, 2)
    assert "loss" in out


def test_small_gpt2_forward():
    config = GPT2Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = GPT2ForCausalLM(config)
    out = model(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        labels=torch.ones(2, 4, dtype=torch.long),
    )
    assert out["logits"].shape == (2, 4, 32)
    assert "loss" in out


def test_gpt2_uses_decoder_only_blocks():
    config = GPT2Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = GPT2ForCausalLM(config)
    assert isinstance(model.transformer.layers[0], TransformerDecoderOnlyBlock)


def test_small_translation_forward():
    config = TransformerMTConfig(
        src_vocab_size=32,
        tgt_vocab_size=32,
        hidden_size=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = TransformerForMachineTranslation(config)
    out = model(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        decoder_input_ids=torch.ones(2, 5, dtype=torch.long),
        decoder_attention_mask=torch.ones(2, 5, dtype=torch.long),
        labels=torch.ones(2, 5, dtype=torch.long),
    )
    assert out["logits"].shape == (2, 5, 32)
    assert "loss" in out


def test_small_xlnet_forward():
    config = XLNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = XLNetLMHeadModel(config)
    out = model(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        labels=torch.ones(2, 4, dtype=torch.long),
    )
    assert out["logits"].shape == (2, 4, 32)
    assert "loss" in out


def test_xlnet_ties_lm_head_to_word_embeddings():
    config = XLNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = XLNetLMHeadModel(config)
    assert model.lm_head.proj.weight is model.xlnet.word_embeddings.weight


def test_xlnet_memory_mask_keeps_memory_visible():
    mask = torch.tensor([[[1, 0], [1, 1]]])
    expanded = prepend_memory_mask(mask, mem_len=2)
    assert expanded.tolist() == [[[1, 1, 1, 0], [1, 1, 1, 1]]]


def test_xlnet_segment_attention_forward():
    config = XLNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        type_vocab_size=2,
    )
    model = XLNetLMHeadModel(config)
    out = model(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        token_type_ids=torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
        labels=torch.ones(2, 4, dtype=torch.long),
    )
    assert out["logits"].shape == (2, 4, 32)
    assert "loss" in out


def test_xlnet_query_stream_uses_mask_embedding():
    config = XLNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = XLNetLMHeadModel(config)
    model.eval()
    query_states = model.xlnet.init_query_stream(batch_size=2, seq_len=4, device=torch.device("cpu"))
    assert query_states.shape == (2, 4, 16)
    assert torch.allclose(query_states[0, 0], query_states[1, 3])


def test_xlnet_permutation_mask_and_target_mapping():
    config = XLNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = XLNetLMHeadModel(config)
    input_ids = torch.ones(2, 4, dtype=torch.long)
    perm_mask = build_permutation_mask(torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]))
    target_mapping = torch.tensor(
        [
            [[0, 0, 0, 1]],
            [[0, 0, 0, 1]],
        ],
        dtype=torch.float,
    )
    labels = torch.ones(2, 1, dtype=torch.long)
    out = model(
        input_ids=input_ids,
        perm_mask=perm_mask,
        target_mapping=target_mapping,
        labels=labels,
    )
    assert out["logits"].shape == (2, 1, 32)
    assert "loss" in out


def test_xlnet_returns_memory():
    config = XLNetConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        mem_len=2,
    )
    model = XLNetLMHeadModel(config)
    out = model(input_ids=torch.ones(2, 4, dtype=torch.long))
    assert len(out["mems"]) == 2
    assert out["mems"][0].shape == (2, 2, 16)

    out_with_memory = model(input_ids=torch.ones(2, 4, dtype=torch.long), mems=out["mems"])
    assert out_with_memory["logits"].shape == (2, 4, 32)
