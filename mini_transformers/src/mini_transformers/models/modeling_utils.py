"""Shared model utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class PreTrainedModel(nn.Module):
    config_class = None

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_json(output_dir / "config.json")
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, model_dir: str | Path):
        model_dir = Path(model_dir)
        if cls.config_class is None:
            raise ValueError("config_class must be set on the model class")
        config = cls.config_class.from_json(model_dir / "config.json")
        model = cls(config)
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        state_dict = cls._rename_old_keys(state_dict)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _rename_old_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        renamed = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("lm_head.") and not key.startswith("lm_head.proj."):
                new_key = key.replace("lm_head.", "lm_head.proj.", 1)
            if key.startswith("classifier.") and not key.startswith("classifier.out_proj."):
                new_key = key.replace("classifier.", "classifier.out_proj.", 1)
            renamed[new_key] = value
        return renamed
