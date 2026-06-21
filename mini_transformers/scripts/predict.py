"""Run a single prediction from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_transformers.models import (
    BertForSequenceClassification,
    GPT2ForCausalLM,
    TransformerForMachineTranslation,
    XLNetLMHeadModel,
)
from mini_transformers.modules.generation import greedy_search
from mini_transformers.tokenization import BasicTokenizer, Vocab, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["lm", "xlnet_lm", "classification", "translation"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    return parser.parse_args()


def predict_lm(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    tokenizer = load_tokenizer(checkpoint)
    model = GPT2ForCausalLM.from_pretrained(checkpoint)
    input_ids = tokenizer.encode(args.text, max_length=args.max_length, add_special_tokens=False).input_ids
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    output_ids = greedy_search(model, input_ids, args.max_new_tokens)
    print(tokenizer.decode(output_ids[0].tolist()))


def predict_xlnet_lm(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    tokenizer = load_tokenizer(checkpoint)
    model = XLNetLMHeadModel.from_pretrained(checkpoint)
    encoded = tokenizer.encode(args.text, max_length=args.max_length, add_special_tokens=False)
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids)["logits"]
    predicted_ids = logits.argmax(dim=-1)[0].tolist()
    print(tokenizer.decode(predicted_ids))


def predict_classification(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    tokenizer = load_tokenizer(checkpoint)
    model = BertForSequenceClassification.from_pretrained(checkpoint)
    encoded = tokenizer.encode(args.text, max_length=args.max_length)
    batch = {
        "input_ids": torch.tensor([encoded.input_ids]),
        "attention_mask": torch.tensor([encoded.attention_mask]),
    }
    with torch.no_grad():
        label = model(**batch)["logits"].argmax(dim=-1).item()
    print(label)


@torch.no_grad()
def predict_translation(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    src_tokenizer = BasicTokenizer(Vocab.from_file(checkpoint / "src_vocab.txt"))
    tgt_tokenizer = BasicTokenizer(Vocab.from_file(checkpoint / "tgt_vocab.txt"))
    model = TransformerForMachineTranslation.from_pretrained(checkpoint)

    source = src_tokenizer.encode(args.text, max_length=args.max_length)
    input_ids = torch.tensor([source.input_ids])
    attention_mask = torch.tensor([source.attention_mask])
    cls_id = tgt_tokenizer.vocab.id_for_token("[CLS]")
    sep_id = tgt_tokenizer.vocab.id_for_token("[SEP]")
    decoder_input_ids = torch.tensor([[cls_id]])

    for _ in range(args.max_new_tokens):
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
    print(tgt_tokenizer.decode(decoder_input_ids[0].tolist()))


def main() -> None:
    args = parse_args()
    if args.task == "lm":
        predict_lm(args)
    elif args.task == "xlnet_lm":
        predict_xlnet_lm(args)
    elif args.task == "classification":
        predict_classification(args)
    elif args.task == "translation":
        predict_translation(args)


if __name__ == "__main__":
    main()
