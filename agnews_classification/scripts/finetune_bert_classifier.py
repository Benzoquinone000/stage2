"""Fine-tune a BERT classifier on AG News from an MLM checkpoint."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[1]
PACKAGE_SRC = ROOT / "mini_transformers" / "src"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(PACKAGE_SRC))

from mini_transformers.configs import BertConfig
from mini_transformers.models import BertForMaskedLM, BertForSequenceClassification
from mini_transformers.tokenization import load_tokenizer, save_tokenizer
from task2_utils import (
    DataCollatorWithPadding,
    add_wandb_args,
    accuracy,
    finish_wandb,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    init_wandb,
    log_wandb,
    macro_f1,
    record_experiment,
    set_seed,
)


@dataclass
class FineTuneConfig:
    mlm_checkpoint: str
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    scheduler: str
    warmup_steps: int
    best_metric: str
    label_smoothing: float
    rdrop_alpha: float
    layerwise_lr_decay: float
    classifier_lr_multiplier: float
    dropout_prob: float | None
    early_stopping_patience: int
    amp: bool
    seed: int
    device: str


class AGNewsClassificationDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_length: int,
        max_examples: int | None = None,
    ) -> None:
        self.rows = read_rows(path, max_examples=max_examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        label, text = self.rows[idx]
        encoded = self.tokenizer.encode(text, max_length=self.max_length)
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "labels": label,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm-checkpoint", default="outputs/bert_mlm")
    parser.add_argument(
        "--classifier-checkpoint",
        default=None,
        help="Optional existing sequence-classification checkpoint to continue fine-tuning from.",
    )
    parser.add_argument("--train-file", default="data/processed/agnews_train.tsv")
    parser.add_argument("--valid-file", default="data/processed/agnews_valid.tsv")
    parser.add_argument("--test-file", default="data/processed/agnews_test.tsv")
    parser.add_argument("--output-dir", default="outputs/bert_classifier")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["linear", "cosine", "constant", "none"], default="linear")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--best-metric",
        choices=["valid_accuracy", "valid_macro_f1", "valid_loss"],
        default="valid_accuracy",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0)
    parser.add_argument("--layerwise-lr-decay", type=float, default=1.0)
    parser.add_argument("--classifier-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--dropout-prob", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    add_wandb_args(parser)
    return parser.parse_args()


def read_rows(path: str | Path, max_examples: int | None = None) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            rows.append((int(row[0]), row[1]))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def build_scheduler(args: argparse.Namespace, optimizer, total_steps: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "constant":
        return get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    if args.scheduler == "linear":
        return get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    return get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_classifier_from_mlm(
    mlm_checkpoint: str | Path,
    dropout_prob: float | None = None,
) -> BertForSequenceClassification:
    checkpoint = Path(mlm_checkpoint)
    mlm_model = BertForMaskedLM.from_pretrained(checkpoint)
    config = BertConfig.from_json(checkpoint / "config.json")
    config.num_labels = 4
    if dropout_prob is not None:
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob
    classifier = BertForSequenceClassification(config)
    classifier.bert.load_state_dict(mlm_model.bert.state_dict())
    return classifier


def _symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_probs_a = F.log_softmax(logits_a, dim=-1)
    log_probs_b = F.log_softmax(logits_b, dim=-1)
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()
    return 0.5 * (
        F.kl_div(log_probs_a, probs_b, reduction="batchmean")
        + F.kl_div(log_probs_b, probs_a, reduction="batchmean")
    )


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def build_optimizer(args: argparse.Namespace, model: BertForSequenceClassification):
    no_decay_terms = ("bias", "layer_norm", "LayerNorm", "layernorm")
    num_layers = model.config.num_hidden_layers
    grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = {}

    def layer_id_for_name(name: str) -> int:
        if name.startswith("bert.embeddings."):
            return 0
        if name.startswith("bert.layers."):
            parts = name.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                return int(parts[2]) + 1
        return num_layers + 1

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        weight_decay = 0.0 if any(term in name for term in no_decay_terms) else args.weight_decay
        if name.startswith("classifier."):
            lr = args.learning_rate * args.classifier_lr_multiplier
        else:
            layer_id = layer_id_for_name(name)
            lr_scale = args.layerwise_lr_decay ** (num_layers + 1 - layer_id)
            lr = args.learning_rate * lr_scale
        grouped.setdefault((lr, weight_decay), []).append(parameter)

    parameter_groups = [
        {"params": params, "lr": lr, "weight_decay": weight_decay}
        for (lr, weight_decay), params in sorted(grouped.items(), key=lambda item: item[0])
    ]
    return torch.optim.AdamW(parameter_groups, lr=args.learning_rate)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device: str,
    max_grad_norm: float,
    label_smoothing: float,
    rdrop_alpha: float,
    gradient_accumulation_steps: int,
    scaler,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        labels = batch["labels"]
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(**inputs)
            loss = classification_loss(outputs["logits"], labels, label_smoothing)
        if rdrop_alpha > 0:
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                outputs_b = model(**inputs)
                loss_b = classification_loss(outputs_b["logits"], labels, label_smoothing)
            loss = 0.5 * (loss + loss_b) + rdrop_alpha * _symmetric_kl(
                outputs["logits"],
                outputs_b["logits"],
            )
        scaled_loss = loss / gradient_accumulation_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        should_step = step % gradient_accumulation_steps == 0 or step == len(loader)
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= old_scale
            else:
                optimizer.step()
                optimizer_stepped = True
            if scheduler is not None and optimizer_stepped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs["logits"].argmax(dim=-1) == batch["labels"]).sum().item()
        total_examples += batch_size
    return total_loss / max(1, total_examples), total_correct / max(1, total_examples)


@torch.no_grad()
def evaluate(model, loader, device: str, amp_enabled: bool = False) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_chunks = []
    label_chunks = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(**batch)
        batch_size = batch["labels"].size(0)
        total_loss += outputs["loss"].item() * batch_size
        total_examples += batch_size
        logits_chunks.append(outputs["logits"].cpu())
        label_chunks.append(batch["labels"].cpu())
    logits = torch.cat(logits_chunks)
    labels = torch.cat(label_chunks)
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": accuracy(logits, labels),
        "macro_f1": macro_f1(logits, labels, num_labels=4),
    }


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def score_for_best(row: dict[str, float], best_metric: str) -> float:
    value = row[best_metric]
    return -value if best_metric == "valid_loss" else value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.mlm_checkpoint)
    train_dataset = AGNewsClassificationDataset(args.train_file, tokenizer, args.max_length, args.max_train_examples)
    valid_dataset = AGNewsClassificationDataset(args.valid_file, tokenizer, args.max_length, args.max_valid_examples)
    test_dataset = AGNewsClassificationDataset(args.test_file, tokenizer, args.max_length, args.max_test_examples)
    collator = DataCollatorWithPadding()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    if args.classifier_checkpoint:
        model = BertForSequenceClassification.from_pretrained(args.classifier_checkpoint).to(args.device)
        if args.dropout_prob is not None:
            model.config.hidden_dropout_prob = args.dropout_prob
            model.config.attention_probs_dropout_prob = args.dropout_prob
    else:
        model = build_classifier_from_mlm(args.mlm_checkpoint, args.dropout_prob).to(args.device)
    optimizer = build_optimizer(args, model)
    updates_per_epoch = (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    scheduler = build_scheduler(args, optimizer, updates_per_epoch * args.epochs)
    amp_enabled = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
    run_config = FineTuneConfig(
        mlm_checkpoint=args.mlm_checkpoint,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        best_metric=args.best_metric,
        label_smoothing=args.label_smoothing,
        rdrop_alpha=args.rdrop_alpha,
        layerwise_lr_decay=args.layerwise_lr_decay,
        classifier_lr_multiplier=args.classifier_lr_multiplier,
        dropout_prob=args.dropout_prob,
        early_stopping_patience=args.early_stopping_patience,
        amp=amp_enabled,
        seed=args.seed,
        device=args.device,
    )
    (output_dir / "finetune_config.json").write_text(json.dumps(asdict(run_config), indent=2), encoding="utf-8")
    save_tokenizer(tokenizer, output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    wandb_run = init_wandb(
        args,
        asdict(run_config),
        stage="classification_finetuning",
        output_dir=output_dir,
        extra_config={
            "train_examples": len(train_dataset),
            "valid_examples": len(valid_dataset),
            "test_examples": len(test_dataset),
            "parameters": parameter_count,
            "output_dir": str(output_dir),
        },
    )

    print(f"device: {args.device}")
    print(f"loaded MLM checkpoint: {args.mlm_checkpoint}")
    print(f"classification train/valid/test: {len(train_dataset):,}/{len(valid_dataset):,}/{len(test_dataset):,}")
    print(
        "sample limits: "
        f"train={args.max_train_examples if args.max_train_examples is not None else 'full'}, "
        f"valid={args.max_valid_examples if args.max_valid_examples is not None else 'full'}, "
        f"test={args.max_test_examples if args.max_test_examples is not None else 'full'}"
    )
    print(f"parameters: {parameter_count:,}")
    print(f"effective batch: {args.batch_size * args.gradient_accumulation_steps}; amp={amp_enabled}")

    history: list[dict[str, float]] = []
    best_score = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            args.device,
            args.max_grad_norm,
            args.label_smoothing,
            args.rdrop_alpha,
            args.gradient_accumulation_steps,
            scaler,
            amp_enabled,
        )
        valid_metrics = evaluate(model, valid_loader, args.device, amp_enabled)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "seconds": perf_counter() - start,
        }
        history.append(row)
        log_wandb(
            wandb_run,
            {
                "epoch": epoch,
                "finetune/train_loss": train_loss,
                "finetune/train_accuracy": train_acc,
                "finetune/valid_loss": valid_metrics["loss"],
                "finetune/valid_accuracy": valid_metrics["accuracy"],
                "finetune/valid_macro_f1": valid_metrics["macro_f1"],
                "finetune/seconds": row["seconds"],
                "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )
        print(
            f"epoch {epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_f1={valid_metrics['macro_f1']:.4f} time={row['seconds']:.1f}s"
        )
        score = score_for_best(row, args.best_metric)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(output_dir)
            if wandb_run is not None:
                wandb_run.summary[f"finetune/best_{args.best_metric}"] = row[args.best_metric]
                wandb_run.summary["finetune/best_epoch"] = best_epoch
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"early stopping after epoch {epoch:02d}; "
                    f"best_epoch={best_epoch:02d} best_{args.best_metric}={best_score:.6f}"
                )
                break

    write_history(output_dir / "finetune_history.csv", history)
    model = BertForSequenceClassification.from_pretrained(output_dir).to(args.device)
    test_metrics = evaluate(model, test_loader, args.device, amp_enabled)
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    best_row = max(history, key=lambda row: score_for_best(row, args.best_metric))
    record_experiment(
        stage="classification_finetuning",
        output_dir=output_dir,
        config=asdict(run_config),
        metrics={
            "best_metric": args.best_metric,
            "best_epoch": best_row["epoch"],
            f"best_{args.best_metric}": best_row[args.best_metric],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "epochs_completed": len(history),
        },
        history=history,
        notes="auto-recorded classifier run",
    )
    log_wandb(
        wandb_run,
        {
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "test/macro_f1": test_metrics["macro_f1"],
        },
        step=len(history),
    )
    if wandb_run is not None:
        wandb_run.summary["test/loss"] = test_metrics["loss"]
        wandb_run.summary["test/accuracy"] = test_metrics["accuracy"]
        wandb_run.summary["test/macro_f1"] = test_metrics["macro_f1"]
    finish_wandb(wandb_run)
    print(
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"saved best classifier checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
