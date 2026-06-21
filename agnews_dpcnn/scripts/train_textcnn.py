"""Train a multi-kernel TextCNN classifier on AG News TSV files."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?|[^\w\s]", re.UNICODE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


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
    seed: int
    amp: bool
    device: str


@dataclass
class TextRow:
    label: int
    text: str
    teacher_probs: list[float] | None = None
    is_pseudo: bool = False


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def read_tsv(path: str | Path) -> list[TextRow]:
    rows: list[TextRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header: list[str] | None = None
        for raw_row in reader:
            if len(raw_row) < 2:
                continue
            if header is None and raw_row[0] == "label":
                header = raw_row
                continue
            if header is None:
                if raw_row[0].isdigit():
                    rows.append(TextRow(label=int(raw_row[0]), text=raw_row[1]))
                continue
            values = {key: raw_row[idx] if idx < len(raw_row) else "" for idx, key in enumerate(header)}
            label = int(values["label"])
            teacher_probs = None
            if all(values.get(f"prob_{class_id}", "") != "" for class_id in range(4)):
                teacher_probs = [float(values[f"prob_{class_id}"]) for class_id in range(4)]
                total = sum(teacher_probs)
                if total > 0:
                    teacher_probs = [value / total for value in teacher_probs]
            rows.append(
                TextRow(
                    label=label,
                    text=values["text"],
                    teacher_probs=teacher_probs,
                    is_pseudo=values.get("is_pseudo", "0") == "1",
                )
            )
    return rows


def build_vocab(rows: list[TextRow], max_size: int, min_freq: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(tokenize(row.text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counts.most_common():
        if count < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def save_vocab(path: Path, vocab: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda item: item[1]):
            f.write(json.dumps({"token": token, "id": idx}, ensure_ascii=False) + "\n")


def encode(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    ids = [vocab.get(token, 1) for token in tokenize(text)]
    if not ids:
        ids = [1]
    ids = ids[:max_length]
    if len(ids) < max_length:
        ids.extend([0] * (max_length - len(ids)))
    return ids


class AGNewsDataset(Dataset):
    def __init__(self, rows: list[TextRow], vocab: dict[str, int], max_length: int) -> None:
        self.rows = rows
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        teacher_probs = row.teacher_probs
        if teacher_probs is None:
            teacher_probs = [1.0 if class_id == row.label else 0.0 for class_id in range(4)]
        return {
            "input_ids": torch.tensor(encode(row.text, self.vocab, self.max_length), dtype=torch.long),
            "labels": torch.tensor(row.label, dtype=torch.long),
            "teacher_probs": torch.tensor(teacher_probs, dtype=torch.float),
            "teacher_mask": torch.tensor(1 if row.is_pseudo and row.teacher_probs is not None else 0, dtype=torch.bool),
        }


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        dropout: float,
        embedding_dropout: float,
        num_classes: int = 4,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding_dropout(self.embedding(input_ids)).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            hidden = F.relu(conv(x))
            pooled.append(F.max_pool1d(hidden, kernel_size=hidden.size(-1)).squeeze(-1))
        x = self.dropout(torch.cat(pooled, dim=-1))
        return self.classifier(x)


def make_loader(
    rows: list[TextRow],
    vocab: dict[str, int],
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        AGNewsDataset(rows, vocab, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def lr_scale(
    step: int,
    total_steps: int,
    warmup_steps: int,
    scheduler: str,
    min_lr_ratio: float,
) -> float:
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    if scheduler == "constant":
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0
    if scheduler == "onecycle":
        total_steps = max(2, total_steps)
        pct_start = max(1.0 / total_steps, warmup_steps / total_steps)
        progress = min(1.0, step / max(1, total_steps - 1))
        if progress < pct_start:
            rise = progress / max(pct_start, 1e-8)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 - math.cos(math.pi * rise))
        decay = (progress - pct_start) / max(1e-8, 1.0 - pct_start)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, decay)))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    if scheduler == "linear":
        return min_lr_ratio + (1.0 - min_lr_ratio) * max(0.0, 1.0 - progress)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 4) -> float:
    scores = []
    for cls in range(num_classes):
        pred_pos = preds == cls
        true_pos = labels == cls
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def train_epoch(
    model: TextCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    total_steps: int,
    warmup_steps: int,
    global_step: int,
    max_grad_norm: float,
    label_smoothing: float,
    distill_alpha: float,
    distill_temperature: float,
    scheduler: str,
    min_lr_ratio: float,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(batch["input_ids"])
            hard_loss = F.cross_entropy(logits, batch["labels"], label_smoothing=label_smoothing)
            loss = hard_loss
            if distill_alpha > 0:
                teacher_mask = batch["teacher_mask"]
                if bool(teacher_mask.any().item()):
                    temperature = max(distill_temperature, 1e-6)
                    teacher_probs = batch["teacher_probs"][teacher_mask].float()
                    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    if temperature != 1.0:
                        teacher_probs = torch.softmax(torch.log(teacher_probs.clamp_min(1e-8)) / temperature, dim=-1)
                    student_log_probs = F.log_softmax(logits[teacher_mask] / temperature, dim=-1)
                    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
                        temperature * temperature
                    )
                    loss = (1.0 - distill_alpha) * hard_loss + distill_alpha * soft_loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scale = lr_scale(global_step, total_steps, warmup_steps, scheduler, min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * scale
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        global_step += 1
        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == batch["labels"]).sum().item()
        total_examples += batch_size
    return total_loss / total_examples, total_correct / total_examples, global_step


@torch.no_grad()
def evaluate(model: TextCNN, loader: DataLoader, device: str, amp_enabled: bool) -> dict[str, float | torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_chunks = []
    label_chunks = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(batch["input_ids"])
            loss = F.cross_entropy(logits, batch["labels"])
        total_loss += loss.item() * batch["labels"].size(0)
        total_examples += batch["labels"].size(0)
        logits_chunks.append(logits.float().cpu())
        label_chunks.append(batch["labels"].cpu())
    logits = torch.cat(logits_chunks)
    labels = torch.cat(label_chunks)
    preds = logits.argmax(dim=-1)
    return {
        "loss": total_loss / total_examples,
        "accuracy": float((preds == labels).float().mean().item()),
        "macro_f1": macro_f1(preds, labels),
        "logits": logits,
        "labels": labels,
    }


def write_probs(path: Path, labels: torch.Tensor, logits: torch.Tensor) -> None:
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "label", "prediction", "prob_0", "prob_1", "prob_2", "prob_3"])
        for idx, (label, pred, row) in enumerate(zip(labels.tolist(), preds.tolist(), probs.tolist())):
            writer.writerow([idx, label, pred, *[f"{value:.8f}" for value in row]])


def save_checkpoint(path: Path, model: TextCNN, config: TrainConfig, vocab: dict[str, int]) -> None:
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "vocab_size": len(vocab)}, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
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
        seed=args.seed,
        amp=args.amp and args.device.startswith("cuda"),
        device=args.device,
    )
    (output_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    train_rows = read_tsv(args.train_file)
    valid_rows = read_tsv(args.valid_file)
    test_rows = read_tsv(args.test_file)
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
            total_steps,
            warmup_steps,
            global_step,
            args.max_grad_norm,
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

    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

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
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_probs(output_dir / "test_probs.tsv", test["labels"], test["logits"])
    print(
        f"test_loss={metrics['test_loss']:.4f} test_acc={metrics['test_accuracy']:.4f} "
        f"test_f1={metrics['test_macro_f1']:.4f}"
    )
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
