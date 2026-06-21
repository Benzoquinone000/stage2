"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    step: int | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, target)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None):
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    return payload
