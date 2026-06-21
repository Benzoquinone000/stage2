"""Continue-pretrain a BERT-style encoder on AG News with masked LM."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = Path(__file__).resolve().parents[1]
PACKAGE_SRC = ROOT / "mini_transformers" / "src"
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(PACKAGE_SRC))

from mini_transformers.configs import BertConfig
from mini_transformers.models import BertForMaskedLM
from mini_transformers.tokenization import WordPieceTokenizer, load_tokenizer, save_tokenizer
from task2_utils import (
    DataCollatorForMaskedLanguageModeling,
    add_wandb_args,
    finish_wandb,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    init_wandb,
    log_wandb,
    perplexity,
    record_experiment,
    set_seed,
)


@dataclass
class MLMConfig:
    init_checkpoint: str | None
    bert_config: str
    vocab_size: int
    max_length: int
    sequence_length: int
    hidden_size: int
    layers: int
    heads: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    decay_layernorm_bias: bool
    max_grad_norm: float
    scheduler: str
    warmup_steps: int
    mlm_probability: float
    amp: bool
    seed: int
    device: str


class AGNewsMLMDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: WordPieceTokenizer,
        max_length: int,
        max_examples: int | None = None,
    ) -> None:
        self.texts = read_texts(path, max_examples=max_examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        encoded = self.tokenizer.encode(self.texts[idx], max_length=self.max_length)
        return {"input_ids": encoded.input_ids, "attention_mask": encoded.attention_mask}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/processed/agnews_train.tsv")
    parser.add_argument("--valid-file", default="data/processed/agnews_valid.tsv")
    parser.add_argument("--output-dir", default="outputs/bert_mlm")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--bert-config", default="configs/bert_base.json")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--decay-layernorm-bias",
        action="store_true",
        help="Apply weight decay to LayerNorm and bias parameters. Disabled by default.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["linear", "cosine", "constant", "none"], default="linear")
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    add_wandb_args(parser)
    return parser.parse_args()


def load_bert_config(args: argparse.Namespace, vocab_size: int) -> BertConfig:
    config = BertConfig.from_json(args.bert_config)
    config.vocab_size = vocab_size
    config.num_labels = 4
    if args.max_length is not None:
        config.max_position_embeddings = args.max_length
    if args.hidden_size is not None:
        config.hidden_size = args.hidden_size
        config.intermediate_size = args.hidden_size * 4
    if args.layers is not None:
        config.num_hidden_layers = args.layers
    if args.heads is not None:
        config.num_attention_heads = args.heads
    return config


def read_texts(path: str | Path, max_examples: int | None = None) -> list[str]:
    texts: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            texts.append(row[1])
            if max_examples is not None and len(texts) >= max_examples:
                break
    return texts


def build_tokenizer(args: argparse.Namespace) -> WordPieceTokenizer:
    if args.init_checkpoint:
        return load_tokenizer(args.init_checkpoint)
    texts = read_texts(args.train_file, max_examples=args.max_train_examples)
    return WordPieceTokenizer.train(texts, vocab_size=args.vocab_size, min_freq=1)


def build_model(args: argparse.Namespace, config: BertConfig) -> BertForMaskedLM:
    if args.init_checkpoint:
        return BertForMaskedLM.from_pretrained(args.init_checkpoint)
    return BertForMaskedLM(config)


def build_scheduler(args: argparse.Namespace, optimizer, total_steps: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "constant":
        return get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
    if args.scheduler == "linear":
        return get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    return get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)


def build_optimizer(args: argparse.Namespace, model: BertForMaskedLM):
    if args.decay_layernorm_bias:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )

    no_decay_terms = ("bias", "layer_norm", "LayerNorm", "layernorm")
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    parameter_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device: str,
    max_grad_norm: float,
    gradient_accumulation_steps: int,
    scaler,
    amp_enabled: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(**batch)
            loss = outputs["loss"]
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
        batch_size = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(model, loader, device: str, amp_enabled: bool = False) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            outputs = model(**batch)
        batch_size = batch["input_ids"].size(0)
        total_loss += outputs["loss"].item() * batch_size
        total_examples += batch_size
    return total_loss / max(1, total_examples)


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args)
    if args.init_checkpoint:
        config = BertConfig.from_json(Path(args.init_checkpoint) / "config.json")
    else:
        config = load_bert_config(args, vocab_size=len(tokenizer.vocab))
    max_length = args.sequence_length or config.max_position_embeddings
    train_dataset = AGNewsMLMDataset(args.train_file, tokenizer, max_length, args.max_train_examples)
    valid_dataset = AGNewsMLMDataset(args.valid_file, tokenizer, max_length, args.max_valid_examples)
    mask_id = tokenizer.vocab.id_for_token("[MASK]")
    special_ids = tuple(tokenizer.vocab.id_for_token(token) for token in ["[PAD]", "[CLS]", "[SEP]", "[MASK]"])
    collator = DataCollatorForMaskedLanguageModeling(
        mask_token_id=mask_id,
        special_token_ids=special_ids,
        mlm_probability=args.mlm_probability,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = build_model(args, config).to(args.device)
    optimizer = build_optimizer(args, model)
    updates_per_epoch = (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    scheduler = build_scheduler(args, optimizer, updates_per_epoch * args.epochs)
    amp_enabled = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
    run_config = MLMConfig(
        init_checkpoint=args.init_checkpoint,
        bert_config=args.bert_config,
        vocab_size=len(tokenizer.vocab),
        max_length=config.max_position_embeddings,
        sequence_length=max_length,
        hidden_size=config.hidden_size,
        layers=config.num_hidden_layers,
        heads=config.num_attention_heads,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        decay_layernorm_bias=args.decay_layernorm_bias,
        max_grad_norm=args.max_grad_norm,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        mlm_probability=args.mlm_probability,
        amp=amp_enabled,
        seed=args.seed,
        device=args.device,
    )
    (output_dir / "mlm_config.json").write_text(json.dumps(asdict(run_config), indent=2), encoding="utf-8")
    save_tokenizer(tokenizer, output_dir)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    wandb_run = init_wandb(
        args,
        asdict(run_config),
        stage="mlm_pretraining",
        output_dir=output_dir,
        extra_config={
            "train_examples": len(train_dataset),
            "valid_examples": len(valid_dataset),
            "parameters": parameter_count,
            "output_dir": str(output_dir),
        },
    )

    print(f"device: {args.device}")
    print(f"mlm train/valid: {len(train_dataset):,}/{len(valid_dataset):,}")
    print(
        "sample limits: "
        f"train={args.max_train_examples if args.max_train_examples is not None else 'full'}, "
        f"valid={args.max_valid_examples if args.max_valid_examples is not None else 'full'}"
    )
    print(f"vocab size: {len(tokenizer.vocab):,}")
    print(f"parameters: {parameter_count:,}")
    print(f"sequence length: {max_length}; effective batch: {args.batch_size * args.gradient_accumulation_steps}; amp={amp_enabled}")

    history: list[dict[str, float]] = []
    best_valid_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            args.device,
            args.max_grad_norm,
            args.gradient_accumulation_steps,
            scaler,
            amp_enabled,
        )
        valid_loss = evaluate(model, valid_loader, args.device, amp_enabled)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_perplexity": perplexity(valid_loss),
            "seconds": perf_counter() - start,
        }
        history.append(row)
        log_wandb(
            wandb_run,
            {
                "epoch": epoch,
                "mlm/train_loss": train_loss,
                "mlm/valid_loss": valid_loss,
                "mlm/valid_perplexity": row["valid_perplexity"],
                "mlm/seconds": row["seconds"],
                "optimizer/learning_rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )
        print(
            f"epoch {epoch:02d} train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} valid_ppl={row['valid_perplexity']:.2f} "
            f"time={row['seconds']:.1f}s"
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save_pretrained(output_dir)
            if wandb_run is not None:
                wandb_run.summary["mlm/best_valid_loss"] = best_valid_loss

    write_history(output_dir / "mlm_history.csv", history)
    record_experiment(
        stage="mlm_pretraining",
        output_dir=output_dir,
        config=asdict(run_config),
        metrics={
            "best_valid_loss": best_valid_loss,
            "best_valid_perplexity": perplexity(best_valid_loss),
            "final_valid_loss": history[-1]["valid_loss"],
            "final_valid_perplexity": history[-1]["valid_perplexity"],
            "epochs_completed": len(history),
        },
        history=history,
        notes="auto-recorded MLM run",
    )
    finish_wandb(wandb_run)
    print(f"saved best MLM checkpoint: {output_dir}")


if __name__ == "__main__":
    main()
