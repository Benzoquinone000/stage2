"""A small, readable trainer for supervised PyTorch experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpointing import load_checkpoint, save_checkpoint
from .debugging import has_nan_or_inf
from .optimizers import build_adamw
from .schedulers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


@dataclass
class TrainingArguments:
    output_dir: str = "outputs/checkpoints/default"
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    scheduler_type: str = "linear"
    eval_steps: int | None = None
    save_steps: int | None = None
    resume_from_checkpoint: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        data_collator: Callable | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list | None = None,
    ) -> None:
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        self.global_step = 0
        self.optimizer = build_adamw(self.model, args.learning_rate, args.weight_decay)
        self.scheduler = None

    def train(self) -> None:
        self.model.to(self.args.device)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        total_steps = len(train_loader) * self.args.num_epochs
        self.scheduler = self.build_scheduler(total_steps)
        if self.args.resume_from_checkpoint is not None:
            payload = load_checkpoint(
                self.args.resume_from_checkpoint,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            self.global_step = int(payload.get("step") or 0)

        self.model.train()
        self.call_event("on_train_begin")
        for epoch in range(self.args.num_epochs):
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{self.args.num_epochs}")
            for batch in progress:
                loss = self.train_step(batch)
                progress.set_postfix(loss=f"{loss.item():.4f}")
                if self.should_evaluate():
                    metrics = self.evaluate()
                    print({f"eval_{key}": round(value, 4) for key, value in metrics.items()})
                    self.call_event("on_evaluate", metrics)
                if self.should_save_checkpoint():
                    self.save_training_checkpoint()
            if self.eval_dataset is not None:
                metrics = self.evaluate()
                self.call_event("on_evaluate", metrics)
                self.call_event("on_epoch_end", metrics)
        self.save_model()
        self.call_event("on_train_end")

    def train_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch = self.move_to_device(batch)
        outputs = self.model(**batch)
        loss = outputs["loss"]
        if has_nan_or_inf(loss):
            raise FloatingPointError("Loss contains NaN or Inf")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        self.call_event("on_step_end", loss.detach())
        return loss.detach()

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

        losses = []
        predictions = []
        labels = []
        for batch in eval_loader:
            batch = self.move_to_device(batch)
            outputs = self.model(**batch)
            if "loss" in outputs:
                losses.append(outputs["loss"].item())
            if "logits" in outputs and "labels" in batch:
                predictions.append(outputs["logits"].detach().cpu())
                labels.append(batch["labels"].detach().cpu())
        metrics = {"loss": sum(losses) / max(1, len(losses))}
        if self.compute_metrics is not None and predictions:
            predictions = self.pad_for_concat(predictions, pad_value=0)
            labels = self.pad_for_concat(labels, pad_value=-100)
            metrics.update(self.compute_metrics(torch.cat(predictions), torch.cat(labels)))
        self.model.train()
        return metrics

    def move_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.to(self.args.device) for name, tensor in batch.items()}

    def pad_for_concat(self, tensors: list[torch.Tensor], pad_value: int) -> list[torch.Tensor]:
        if not tensors or tensors[0].dim() < 2:
            return tensors
        max_len = max(tensor.size(1) for tensor in tensors)
        padded = []
        for tensor in tensors:
            if tensor.size(1) == max_len:
                padded.append(tensor)
                continue
            pad_shape = list(tensor.shape)
            pad_shape[1] = max_len - tensor.size(1)
            pad = tensor.new_full(pad_shape, pad_value)
            padded.append(torch.cat([tensor, pad], dim=1))
        return padded

    def build_scheduler(self, total_steps: int):
        if self.args.scheduler_type == "none":
            return None
        if self.args.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(self.optimizer, self.args.warmup_steps, total_steps)
        if self.args.scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(self.optimizer, self.args.warmup_steps, total_steps)
        raise ValueError(f"Unknown scheduler_type: {self.args.scheduler_type}")

    def should_evaluate(self) -> bool:
        if self.eval_dataset is None or self.args.eval_steps is None:
            return False
        return self.global_step > 0 and self.global_step % self.args.eval_steps == 0

    def should_save_checkpoint(self) -> bool:
        if self.args.save_steps is None:
            return False
        return self.global_step > 0 and self.global_step % self.args.save_steps == 0

    def save_training_checkpoint(self) -> None:
        path = Path(self.args.output_dir) / f"checkpoint_step_{self.global_step}.pt"
        save_checkpoint(path, self.model, self.optimizer, self.scheduler, self.global_step)

    def call_event(self, name: str, *args) -> None:
        for callback in self.callbacks:
            getattr(callback, name)(self, *args)

    def save_model(self) -> None:
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            torch.save(self.model.state_dict(), output_dir / "pytorch_model.bin")
