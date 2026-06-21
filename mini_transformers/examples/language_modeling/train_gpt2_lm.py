"""Train a tiny GPT-style language model on a plain text file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformers.configs import GPT2Config
from mini_transformers.data import DataCollatorForLanguageModeling, LanguageModelingDataset
from mini_transformers.metrics import perplexity
from mini_transformers.models import GPT2ForCausalLM
from mini_transformers.modules.generation import greedy_search
from mini_transformers.tokenization import BPETokenizer, BasicTokenizer, Vocab, save_tokenizer
from mini_transformers.training import Trainer, TrainingArguments
from mini_transformers.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", default="examples/language_modeling/tiny_corpus.txt")
    parser.add_argument("--output-dir", default="outputs/checkpoints/gpt2_tiny")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tokenizer", choices=["basic", "bpe"], default="bpe")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--scheduler-type", choices=["linear", "cosine", "none"], default="linear")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--prompt", default="language models")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_tokenizer(text: str, tokenizer_type: str, vocab_size: int):
    if tokenizer_type == "bpe":
        return BPETokenizer.train([text], vocab_size=vocab_size)
    rough_tokenizer = BasicTokenizer(Vocab(["[PAD]", "[UNK]"]))
    vocab = Vocab.from_tokens(rough_tokenizer.tokenize(text))
    return BasicTokenizer(vocab)


def split_ids(token_ids: list[int], valid_ratio: float = 0.2) -> tuple[list[int], list[int]]:
    split = int(len(token_ids) * (1 - valid_ratio))
    return token_ids[:split], token_ids[split:]


def main() -> None:
    args = parse_args()
    set_seed(42)

    text = load_text(args.text_file)
    tokenizer = build_tokenizer(text, args.tokenizer, args.vocab_size)
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    if args.max_tokens is not None:
        token_ids = token_ids[: args.max_tokens]
    train_ids, valid_ids = split_ids(token_ids)

    if len(train_ids) <= args.block_size or len(valid_ids) <= args.block_size:
        raise ValueError("Text is too short. Use a smaller block size or a longer text file.")

    train_dataset = LanguageModelingDataset(train_ids, block_size=args.block_size)
    valid_dataset = LanguageModelingDataset(valid_ids, block_size=args.block_size)

    config = GPT2Config(
        vocab_size=len(tokenizer.vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.block_size + args.max_new_tokens,
    )
    model = GPT2ForCausalLM(config)

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
        data_collator=DataCollatorForLanguageModeling(),
    )
    trainer.train()
    save_tokenizer(tokenizer, args.output_dir)

    metrics = trainer.evaluate()
    print(f"validation loss: {metrics['loss']:.4f}")
    print(f"validation perplexity: {perplexity(metrics['loss']):.2f}")

    prompt_ids = tokenizer.encode(args.prompt, max_length=args.block_size, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=trainer.args.device)
    output_ids = greedy_search(model, input_ids, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
