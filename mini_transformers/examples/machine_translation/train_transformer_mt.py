"""Train a tiny Transformer encoder-decoder for translation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformers.configs import TransformerMTConfig
from mini_transformers.data import DataCollatorForSeq2Seq, TranslationDataset
from mini_transformers.data.preprocessing import train_valid_split
from mini_transformers.data.readers import read_translation_tsv
from mini_transformers.metrics import corpus_bleu
from mini_transformers.models import TransformerForMachineTranslation
from mini_transformers.tokenization import BasicTokenizer, Vocab
from mini_transformers.training import Trainer, TrainingArguments
from mini_transformers.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="examples/machine_translation/tiny_en_de.tsv")
    parser.add_argument("--valid-file", default=None)
    parser.add_argument("--output-dir", default="outputs/checkpoints/mt_tiny")
    parser.add_argument("--max-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--scheduler-type", choices=["linear", "cosine", "none"], default="linear")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--source", default="i like apples")
    return parser.parse_args()


def build_tokenizer(texts: list[str]) -> BasicTokenizer:
    rough_tokenizer = BasicTokenizer(Vocab(["[PAD]", "[UNK]"]))
    tokens = []
    for text in texts:
        tokens.extend(rough_tokenizer.tokenize(text))
    return BasicTokenizer(Vocab.from_tokens(tokens))


@torch.no_grad()
def translate(
    model: TransformerForMachineTranslation,
    source_tokenizer: BasicTokenizer,
    target_tokenizer: BasicTokenizer,
    text: str,
    max_length: int,
    device: str,
) -> str:
    source = source_tokenizer.encode(text, max_length=max_length)
    input_ids = torch.tensor([source.input_ids], device=device)
    attention_mask = torch.tensor([source.attention_mask], device=device)

    cls_id = target_tokenizer.vocab.id_for_token("[CLS]")
    sep_id = target_tokenizer.vocab.id_for_token("[SEP]")
    decoder_input_ids = torch.tensor([[cls_id]], device=device)

    model.eval()
    for _ in range(max_length - 1):
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )["logits"]
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=-1)
        if next_id.item() == sep_id:
            break
    return target_tokenizer.decode(decoder_input_ids[0].tolist())


def main() -> None:
    args = parse_args()
    set_seed(42)

    train_examples = read_translation_tsv(args.data_file)
    if args.valid_file:
        valid_examples = read_translation_tsv(args.valid_file)
        examples = train_examples + valid_examples
    else:
        train_examples, valid_examples = train_valid_split(train_examples, valid_ratio=0.25, seed=42)
        examples = train_examples + valid_examples
    source_tokenizer = build_tokenizer([example.source for example in examples])
    target_tokenizer = build_tokenizer([example.target for example in examples])

    train_dataset = TranslationDataset(
        train_examples,
        source_tokenizer,
        target_tokenizer,
        args.max_length,
        args.max_length,
    )
    valid_dataset = TranslationDataset(
        valid_examples,
        source_tokenizer,
        target_tokenizer,
        args.max_length,
        args.max_length,
    )

    config = TransformerMTConfig(
        src_vocab_size=len(source_tokenizer.vocab),
        tgt_vocab_size=len(target_tokenizer.vocab),
        hidden_size=args.hidden_size,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_length,
    )
    model = TransformerForMachineTranslation(config)

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
        data_collator=DataCollatorForSeq2Seq(
            start_token_id=target_tokenizer.vocab.id_for_token("[CLS]")
        ),
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    source_tokenizer.vocab.save(output_dir / "src_vocab.txt")
    target_tokenizer.vocab.save(output_dir / "tgt_vocab.txt")

    metrics = trainer.evaluate()
    predictions = [
        translate(model, source_tokenizer, target_tokenizer, example.source, args.max_length, trainer.args.device)
        for example in valid_examples
    ]
    metrics["bleu"] = corpus_bleu(predictions, [example.target for example in valid_examples])
    print({key: round(value, 4) for key, value in metrics.items()})
    print(translate(model, source_tokenizer, target_tokenizer, args.source, args.max_length, trainer.args.device))


if __name__ == "__main__":
    main()
