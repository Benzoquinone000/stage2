"""Train a tiny XLNet-style permutation language model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformers.configs import XLNetConfig
from mini_transformers.data import (
    DataCollatorForPermutationLanguageModeling,
    PermutationLanguageModelingDataset,
)
from mini_transformers.metrics import perplexity
from mini_transformers.models import XLNetLMHeadModel
from mini_transformers.tokenization import BasicTokenizer, Vocab, save_tokenizer
from mini_transformers.training import Trainer, TrainingArguments
from mini_transformers.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default="examples/language_modeling/tiny_corpus.txt")
    parser.add_argument("--output-dir", default="outputs/checkpoints/xlnet_tiny")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--mem-len", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--scheduler-type", choices=["linear", "cosine", "none"], default="linear")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def build_tokenizer(text: str) -> BasicTokenizer:
    rough_tokenizer = BasicTokenizer(Vocab(["[PAD]", "[UNK]"]))
    return BasicTokenizer(Vocab.from_tokens(rough_tokenizer.tokenize(text)))


def split_ids(token_ids: list[int], valid_ratio: float = 0.2) -> tuple[list[int], list[int]]:
    split = int(len(token_ids) * (1 - valid_ratio))
    return token_ids[:split], token_ids[split:]


def main() -> None:
    args = parse_args()
    set_seed(42)

    text = Path(args.text_file).read_text(encoding="utf-8")
    tokenizer = build_tokenizer(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    if args.max_tokens is not None:
        token_ids = token_ids[: args.max_tokens]
    train_ids, valid_ids = split_ids(token_ids)

    train_dataset = PermutationLanguageModelingDataset(train_ids, args.block_size)
    valid_dataset = PermutationLanguageModelingDataset(valid_ids, args.block_size)
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Text is too short. Use a smaller block size or more text.")

    config = XLNetConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.block_size,
        mem_len=args.mem_len,
    )
    model = XLNetLMHeadModel(config)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            scheduler_type=args.scheduler_type,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            resume_from_checkpoint=args.resume_from_checkpoint,
        ),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForPermutationLanguageModeling(),
    )
    trainer.train()
    save_tokenizer(tokenizer, args.output_dir)

    metrics = trainer.evaluate()
    print(f"validation loss: {metrics['loss']:.4f}")
    print(f"validation perplexity: {perplexity(metrics['loss']):.2f}")


if __name__ == "__main__":
    main()
