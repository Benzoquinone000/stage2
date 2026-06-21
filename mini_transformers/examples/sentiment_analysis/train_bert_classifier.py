"""Train a tiny BERT-style classifier from a TSV file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformers.configs import BertConfig
from mini_transformers.data import DataCollatorWithPadding, TextClassificationDataset
from mini_transformers.data.preprocessing import train_valid_split
from mini_transformers.data.readers import read_classification_tsv
from mini_transformers.metrics import accuracy, macro_f1
from mini_transformers.models import BertForSequenceClassification
from mini_transformers.tokenization import BasicTokenizer, Vocab, save_tokenizer
from mini_transformers.training import Trainer, TrainingArguments
from mini_transformers.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="examples/sentiment_analysis/tiny_sentiment.tsv")
    parser.add_argument("--valid-file", default=None)
    parser.add_argument("--output-dir", default="outputs/checkpoints/bert_sentiment_tiny")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--scheduler-type", choices=["linear", "cosine", "none"], default="linear")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--text", default="this movie is good")
    return parser.parse_args()


def build_tokenizer(texts: list[str]) -> BasicTokenizer:
    rough_tokenizer = BasicTokenizer(Vocab(["[PAD]", "[UNK]"]))
    tokens = []
    for text in texts:
        tokens.extend(rough_tokenizer.tokenize(text))
    return BasicTokenizer(Vocab.from_tokens(tokens))


def compute_classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    return {
        "accuracy": accuracy(logits, labels),
        "macro_f1": macro_f1(logits, labels),
    }


def main() -> None:
    args = parse_args()
    set_seed(42)

    train_examples = read_classification_tsv(args.data_file)
    if args.valid_file:
        valid_examples = read_classification_tsv(args.valid_file)
        examples = train_examples + valid_examples
    else:
        train_examples, valid_examples = train_valid_split(train_examples, valid_ratio=0.25, seed=42)
        examples = train_examples + valid_examples
    tokenizer = build_tokenizer([example.text for example in examples])

    train_dataset = TextClassificationDataset(train_examples, tokenizer, max_length=args.max_length)
    valid_dataset = TextClassificationDataset(valid_examples, tokenizer, max_length=args.max_length)

    config = BertConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_length,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)

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
        data_collator=DataCollatorWithPadding(),
        compute_metrics=compute_classification_metrics,
    )
    trainer.train()
    save_tokenizer(tokenizer, args.output_dir)

    metrics = trainer.evaluate()
    print({key: round(value, 4) for key, value in metrics.items()})

    encoded = tokenizer.encode(args.text, max_length=args.max_length)
    batch = {
        "input_ids": torch.tensor([encoded.input_ids], device=trainer.args.device),
        "attention_mask": torch.tensor([encoded.attention_mask], device=trainer.args.device),
    }
    model.eval()
    with torch.no_grad():
        label = model(**batch)["logits"].argmax(dim=-1).item()
    print(f"prediction: {label}")


if __name__ == "__main__":
    main()
