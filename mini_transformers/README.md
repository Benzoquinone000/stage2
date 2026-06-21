# Mini Transformers

A compact PyTorch project for implementing and testing three Transformer language
model families:

- BERT: bidirectional encoder for MLM, NSP, and sequence classification.
- GPT-2: decoder-only causal language model.
- XLNet: permutation language model with two-stream relative attention.

The code is intentionally small and readable. It is organized like a tiny
version of `transformers`, but the model layers, heads, tokenizers, datasets,
trainer, and metrics are implemented in this repository.

## Project Map

- `src/mini_transformers/configs`: dataclass model configs.
- `src/mini_transformers/modules`: reusable attention, embeddings, and blocks.
- `src/mini_transformers/models/bert`: BERT encoder and task heads.
- `src/mini_transformers/models/gpt2`: GPT-2 decoder-only model and LM head.
- `src/mini_transformers/models/xlnet`: XLNet relative attention, memory, and LM head.
- `src/mini_transformers/tokenization`: Basic, BPE, WordPiece, and SentencePiece wrappers.
- `src/mini_transformers/data`: datasets and collators for MLM, causal LM, permutation LM, and classification.
- `src/mini_transformers/training`: trainer, checkpointing, callbacks, optimizer, and schedulers.
- `tests`: fast contract tests for model shapes, tokenizers, datasets, trainer, and registries.

The machine-translation encoder-decoder code is kept as an optional extension,
but the main project scope is BERT, GPT-2, and XLNet.

## Quick Smoke Runs

```bash
python scripts/train.py --task gpt2 --epochs 1
python scripts/train.py --task bert --epochs 1
python scripts/train.py --task xlnet --epochs 1
```

Equivalent direct scripts:

```bash
python examples/language_modeling/train_gpt2_lm.py --epochs 1
python examples/language_modeling/train_bert_pretraining.py --epochs 1
python examples/language_modeling/train_xlnet_lm.py --epochs 1
python examples/sentiment_analysis/train_bert_classifier.py --epochs 1
```

## Common Workflows

Prepare small public datasets:

```bash
python scripts/prepare_dataset.py --dataset tiny_shakespeare
python scripts/prepare_dataset.py --dataset sms_spam --max-examples 2000
```

Train on prepared data:

```bash
python examples/language_modeling/train_gpt2_lm.py \
  --text-file data/processed/tiny_shakespeare/tiny_shakespeare.txt \
  --max-tokens 2000

python examples/sentiment_analysis/train_bert_classifier.py \
  --data-file data/processed/sms_spam/sms_spam_train.tsv \
  --valid-file data/processed/sms_spam/sms_spam_valid.tsv
```

Evaluate or predict:

```bash
python scripts/evaluate.py --task lm --checkpoint outputs/checkpoints/gpt2_tiny --data-file examples/language_modeling/tiny_corpus.txt
python scripts/predict.py --task lm --checkpoint outputs/checkpoints/gpt2_tiny --text "language models"
```

Training scripts also accept:

```text
--scheduler-type linear|cosine|none
--eval-steps N
--save-steps N
--resume-from-checkpoint PATH
```
