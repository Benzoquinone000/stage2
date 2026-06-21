"""Evaluate a saved checkpoint on a TSV or text file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformers.data import (
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    LanguageModelingDataset,
    PermutationLanguageModelingDataset,
    TextClassificationDataset,
    TranslationDataset,
    read_classification_tsv,
    read_translation_tsv,
)
from mini_transformers.metrics import accuracy, corpus_bleu, macro_f1, perplexity
from mini_transformers.models import (
    BertForSequenceClassification,
    GPT2ForCausalLM,
    TransformerForMachineTranslation,
    XLNetLMHeadModel,
)
from mini_transformers.tokenization import BasicTokenizer, Vocab, load_tokenizer
from mini_transformers.training import Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["lm", "xlnet_lm", "classification", "translation"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def decode_batch(tokenizer: BasicTokenizer, ids: torch.Tensor) -> list[str]:
    pad_id = tokenizer.vocab.id_for_token("[PAD]")
    texts = []
    for row in ids.tolist():
        row = [pad_id if token_id == -100 else token_id for token_id in row]
        texts.append(tokenizer.decode(row))
    return texts


def evaluate_lm(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    tokenizer = load_tokenizer(checkpoint)
    model = GPT2ForCausalLM.from_pretrained(checkpoint)
    text = Path(args.data_file).read_text(encoding="utf-8")
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    dataset = LanguageModelingDataset(token_ids, args.block_size)

    trainer = Trainer(
        model,
        TrainingArguments(batch_size=args.batch_size),
        eval_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(),
    )
    metrics = trainer.evaluate()
    metrics["perplexity"] = perplexity(metrics["loss"])
    print({key: round(value, 4) for key, value in metrics.items()})


def evaluate_xlnet_lm(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    tokenizer = load_tokenizer(checkpoint)
    model = XLNetLMHeadModel.from_pretrained(checkpoint)
    text = Path(args.data_file).read_text(encoding="utf-8")
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    dataset = PermutationLanguageModelingDataset(token_ids, args.block_size)

    trainer = Trainer(
        model,
        TrainingArguments(batch_size=args.batch_size),
        eval_dataset=dataset,
        data_collator=DataCollatorForPermutationLanguageModeling(),
    )
    metrics = trainer.evaluate()
    metrics["perplexity"] = perplexity(metrics["loss"])
    print({key: round(value, 4) for key, value in metrics.items()})


def evaluate_classification(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    tokenizer = load_tokenizer(checkpoint)
    model = BertForSequenceClassification.from_pretrained(checkpoint)
    examples = read_classification_tsv(args.data_file)
    dataset = TextClassificationDataset(examples, tokenizer, args.max_length)

    def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        return {"accuracy": accuracy(logits, labels), "macro_f1": macro_f1(logits, labels)}

    trainer = Trainer(
        model,
        TrainingArguments(batch_size=args.batch_size),
        eval_dataset=dataset,
        data_collator=DataCollatorWithPadding(),
        compute_metrics=compute_metrics,
    )
    print({key: round(value, 4) for key, value in trainer.evaluate().items()})


def evaluate_translation(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    src_tokenizer = BasicTokenizer(Vocab.from_file(checkpoint / "src_vocab.txt"))
    tgt_tokenizer = BasicTokenizer(Vocab.from_file(checkpoint / "tgt_vocab.txt"))
    model = TransformerForMachineTranslation.from_pretrained(checkpoint)
    examples = read_translation_tsv(args.data_file)
    dataset = TranslationDataset(examples, src_tokenizer, tgt_tokenizer, args.max_length, args.max_length)

    trainer = Trainer(
        model,
        TrainingArguments(batch_size=args.batch_size),
        eval_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(start_token_id=tgt_tokenizer.vocab.id_for_token("[CLS]")),
    )
    metrics = trainer.evaluate()
    predictions = [
        generate_translation(model, src_tokenizer, tgt_tokenizer, example.source, args.max_length)
        for example in examples
    ]
    references = [example.target for example in examples]
    metrics["bleu"] = corpus_bleu(predictions, references)
    print({key: round(value, 4) for key, value in metrics.items()})


@torch.no_grad()
def generate_translation(model, src_tokenizer, tgt_tokenizer, text: str, max_length: int) -> str:
    source = src_tokenizer.encode(text, max_length=max_length)
    input_ids = torch.tensor([source.input_ids])
    attention_mask = torch.tensor([source.attention_mask])
    cls_id = tgt_tokenizer.vocab.id_for_token("[CLS]")
    sep_id = tgt_tokenizer.vocab.id_for_token("[SEP]")
    decoder_input_ids = torch.tensor([[cls_id]])

    model.eval()
    for _ in range(max_length - 1):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=torch.ones_like(decoder_input_ids),
        )["logits"]
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        decoder_input_ids = torch.cat([decoder_input_ids, next_id], dim=-1)
        if next_id.item() == sep_id:
            break
    return tgt_tokenizer.decode(decoder_input_ids[0].tolist())


def main() -> None:
    args = parse_args()
    if args.task == "lm":
        evaluate_lm(args)
    elif args.task == "xlnet_lm":
        evaluate_xlnet_lm(args)
    elif args.task == "classification":
        evaluate_classification(args)
    elif args.task == "translation":
        evaluate_translation(args)


if __name__ == "__main__":
    main()
