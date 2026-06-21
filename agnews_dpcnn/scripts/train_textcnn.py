"""Train a multi-kernel TextCNN classifier on AG News TSV files."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from agnews_dpcnn.data import build_vocab, make_loader, read_tsv, save_vocab
from agnews_dpcnn.models import TextCNN
from agnews_dpcnn.training import (
    evaluate,
    save_checkpoint,
    save_test_outputs,
    set_seed,
    train_epoch,
    write_history,
)


@dataclass
class TrainConfig:
    train_file: str
    valid_file: str
    test_file: str
    output_dir: str
    max_vocab_size: int
    min_freq: int
    max_length: int
    embedding_dim: int
    num_filters: int
    kernel_sizes: list[int]
    dropout: float
    embedding_dropout: float
    label_smoothing: float
    distill_alpha: float
    distill_temperature: float
    scheduler: str
    min_lr_ratio: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    warmup_ratio: float
    patience: int
    max_train_examples: int | None
    max_valid_examples: int | None
    max_test_examples: int | None
    seed: int
    amp: bool
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-vocab-size", type=int, default=100000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--num-filters", type=int, default=256)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--dropout", type=float, default=0.55)
    parser.add_argument("--embedding-dropout", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--distill-alpha", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["cosine", "linear", "constant", "onecycle"], default="cosine")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    amp_enabled = args.amp and args.device.startswith("cuda")
    return TrainConfig(
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        kernel_sizes=args.kernel_sizes,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
        label_smoothing=args.label_smoothing,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        scheduler=args.scheduler,
        min_lr_ratio=args.min_lr_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        max_train_examples=args.max_train_examples,
        max_valid_examples=args.max_valid_examples,
        max_test_examples=args.max_test_examples,
        seed=args.seed,
        amp=amp_enabled,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    train_rows = read_tsv(args.train_file, args.max_train_examples)
    valid_rows = read_tsv(args.valid_file, args.max_valid_examples)
    test_rows = read_tsv(args.test_file, args.max_test_examples)
    vocab = build_vocab(train_rows, args.max_vocab_size, args.min_freq)
    save_vocab(output_dir / "vocab.jsonl", vocab)

    train_loader = make_loader(train_rows, vocab, args.max_length, args.batch_size, shuffle=True)
    valid_loader = make_loader(valid_rows, vocab, args.max_length, args.batch_size, shuffle=False)
    test_loader = make_loader(test_rows, vocab, args.max_length, args.batch_size, shuffle=False)

    amp_enabled = args.amp and args.device.startswith("cuda")
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        kernel_sizes=args.kernel_sizes,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for group in optimizer.param_groups:
        group["initial_lr"] = args.learning_rate
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None

    print(f"device: {args.device}; amp={amp_enabled}")
    print(f"train/valid/test: {len(train_rows):,}/{len(valid_rows):,}/{len(test_rows):,}")
    print(f"vocab size: {len(vocab):,}; max_length={args.max_length}; kernels={args.kernel_sizes}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    history: list[dict[str, float]] = []
    best_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            scaler,
            amp_enabled,
            args.max_grad_norm,
            total_steps,
            warmup_steps,
            global_step,
            args.label_smoothing,
            args.distill_alpha,
            args.distill_temperature,
            args.scheduler,
            args.min_lr_ratio,
        )
        valid = evaluate(model, valid_loader, args.device, amp_enabled)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "valid_loss": float(valid["loss"]),
            "valid_accuracy": float(valid["accuracy"]),
            "valid_macro_f1": float(valid["macro_f1"]),
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"valid_loss={row['valid_loss']:.4f} valid_acc={row['valid_accuracy']:.4f} "
            f"valid_f1={row['valid_macro_f1']:.4f} time={row['seconds']:.1f}s"
        )
        if row["valid_macro_f1"] > best_f1:
            best_f1 = row["valid_macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(output_dir / "best_model.pt", model, config, vocab)
        else:
            bad_epochs += 1
            if args.patience > 0 and bad_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}; best_epoch={best_epoch} best_valid_f1={best_f1:.6f}")
                break

    write_history(output_dir / "history.csv", history)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, test_loader, args.device, amp_enabled)
    metrics = {
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_f1,
        "test_loss": float(test["loss"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "epochs_completed": len(history),
    }
    save_test_outputs(output_dir, metrics, test["labels"], test["logits"])
    print(
        f"test_loss={metrics['test_loss']:.4f} test_acc={metrics['test_accuracy']:.4f} "
        f"test_f1={metrics['test_macro_f1']:.4f}"
    )
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
