"""Predict AG News labels with a fine-tuned BERT classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SRC = ROOT / "mini_transformers" / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from mini_transformers.models import BertForSequenceClassification
from mini_transformers.tokenization import load_tokenizer


LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/bert_classifier")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.checkpoint)
    model = BertForSequenceClassification.from_pretrained(args.checkpoint).to(args.device)
    encoded = tokenizer.encode(args.text, max_length=args.max_length)
    batch = {
        "input_ids": torch.tensor([encoded.input_ids], device=args.device),
        "attention_mask": torch.tensor([encoded.attention_mask], device=args.device),
    }
    model.eval()
    with torch.no_grad():
        logits = model(**batch)["logits"]
        probs = torch.softmax(logits, dim=-1)[0]
    label_id = int(probs.argmax().item())
    print(f"label={label_id} name={LABELS[label_id]} confidence={probs[label_id].item():.4f}")
    print({LABELS[idx]: round(float(value), 4) for idx, value in enumerate(probs.tolist())})


if __name__ == "__main__":
    main()
