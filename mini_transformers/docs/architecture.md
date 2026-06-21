# Architecture

The project is organized around three Transformer language model families:
BERT, GPT-2, and XLNet.

## Core Models

### BERT

`src/mini_transformers/models/bert` implements a bidirectional Transformer
encoder. It reuses `TransformerEncoderBlock` and exposes:

- `BertModel`
- `BertForMaskedLM`
- `BertForPreTraining`
- `BertForSequenceClassification`

BERT batches normally contain `input_ids`, `attention_mask`, optional
`token_type_ids`, and task labels.

### GPT-2

`src/mini_transformers/models/gpt2` implements a decoder-only causal language
model. It uses `TransformerDecoderOnlyBlock`; the causal behavior comes from the
lower-triangular mask built in `build_causal_mask`.

GPT-2 batches contain `input_ids`, `attention_mask`, and next-token `labels`.
The dataset shifts labels before batching, so the model loss can be computed
directly over aligned logits and labels.

### XLNet

`src/mini_transformers/models/xlnet` implements a compact XLNet-style language
model with:

- permutation masks
- content and query streams
- relative position scores
- segment-aware attention scores
- optional memory
- target mapping for partial prediction
- tied input/output token embeddings

XLNet batches contain `input_ids`, `attention_mask`, `perm_mask`, and `labels`.
Optional `token_type_ids`, `target_mapping`, and `mems` are supported for
segment-aware attention, selected-position prediction, and recurrent memory.

## Shared Layers

`src/mini_transformers/modules` contains reusable building blocks:

- `MultiHeadSelfAttention`
- `TokenPositionEmbedding`
- `TransformerEncoderBlock`
- `TransformerDecoderOnlyBlock`
- `TransformerDecoderBlock`
- `FeedForward`

`TransformerDecoderBlock` includes cross-attention and is only needed by the
optional encoder-decoder extension.

## Training Contract

Task scripts build datasets and collators, then pass batches to `Trainer`.
Every trainable model follows the same contract:

```python
outputs = model(**batch)
loss = outputs["loss"]
```

The trainer handles device placement, optimizer steps, schedulers, evaluation,
callbacks, checkpointing, and final `save_pretrained` output.
