"""Training and evaluation loops shared by CNN classifiers."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import move_batch
from .metrics import accuracy_from_logits, macro_f1, write_predictions


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def lr_multiplier(
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


def classification_or_distill_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    label_smoothing: float,
    distill_alpha: float,
    distill_temperature: float,
) -> torch.Tensor:
    hard_loss = F.cross_entropy(logits, batch["labels"], label_smoothing=label_smoothing)
    if distill_alpha <= 0:
        return hard_loss
    teacher_mask = batch["teacher_mask"]
    if not bool(teacher_mask.any().item()):
        return hard_loss
    temperature = max(distill_temperature, 1e-6)
    teacher_probs = batch["teacher_probs"][teacher_mask].float()
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if temperature != 1.0:
        teacher_probs = torch.softmax(torch.log(teacher_probs.clamp_min(1e-8)) / temperature, dim=-1)
    student_log_probs = F.log_softmax(logits[teacher_mask] / temperature, dim=-1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)
    return (1.0 - distill_alpha) * hard_loss + distill_alpha * soft_loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    max_grad_norm: float,
    total_steps: int,
    warmup_steps: int,
    global_step: int,
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
            loss = classification_or_distill_loss(logits, batch, label_smoothing, distill_alpha, distill_temperature)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        lr_scale = lr_multiplier(global_step, total_steps, warmup_steps, scheduler, min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_scale
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        global_step += 1
        labels = batch["labels"]
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size
    return total_loss / total_examples, total_correct / total_examples, global_step


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, amp_enabled: bool) -> dict[str, float | torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_chunks = []
    label_chunks = []
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch["labels"]
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(batch["input_ids"])
            loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_examples += labels.size(0)
        logits_chunks.append(logits.float().cpu())
        label_chunks.append(labels.cpu())
    logits = torch.cat(logits_chunks)
    labels = torch.cat(label_chunks)
    preds = logits.argmax(dim=-1)
    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy_from_logits(logits, labels),
        "macro_f1": macro_f1(preds, labels),
        "logits": logits,
        "labels": labels,
    }


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model: nn.Module, config, vocab: dict[str, int]) -> None:
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "vocab_size": len(vocab)}, path)


def save_test_outputs(output_dir: Path, metrics: dict[str, float | int], labels: torch.Tensor, logits: torch.Tensor) -> None:
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_predictions(output_dir / "test_probs.tsv", labels, torch.softmax(logits, dim=-1))

