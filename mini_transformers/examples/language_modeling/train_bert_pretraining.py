"""Train BERT with masked language modeling and next sentence prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformers.configs import BertConfig
from mini_transformers.data import BertPreTrainingDataset, DataCollatorForMaskedLanguageModeling
from mini_transformers.models import BertForPreTraining
from mini_transformers.tokenization import WordPieceTokenizer, save_tokenizer
from mini_transformers.training import Trainer, TrainingArguments
from mini_transformers.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default="examples/language_modeling/tiny_corpus.txt")
    parser.add_argument("--output-dir", default="outputs/checkpoints/bert_pretraining_tiny")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    sentences = Path(args.text_file).read_text(encoding="utf-8").splitlines()
    tokenizer = WordPieceTokenizer.train(sentences, vocab_size=args.vocab_size)
    dataset = BertPreTrainingDataset(sentences, tokenizer, args.max_length)
    split = max(1, int(len(dataset) * 0.8))
    train_dataset = torch_subset(dataset, 0, split)
    valid_dataset = torch_subset(dataset, split, len(dataset))

    config = BertConfig(
        vocab_size=len(tokenizer.vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_length,
    )
    model = BertForPreTraining(config)
    collator = DataCollatorForMaskedLanguageModeling(
        mask_token_id=tokenizer.vocab.id_for_token("[MASK]"),
        special_token_ids=tuple(
            tokenizer.vocab.id_for_token(token)
            for token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        ),
    )
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
        data_collator=collator,
    )
    trainer.train()
    save_tokenizer(tokenizer, args.output_dir)
    print({key: round(value, 4) for key, value in trainer.evaluate().items()})


def torch_subset(dataset, start: int, end: int):
    from torch.utils.data import Subset

    indices = list(range(start, end))
    return Subset(dataset, indices or [0])


if __name__ == "__main__":
    main()
