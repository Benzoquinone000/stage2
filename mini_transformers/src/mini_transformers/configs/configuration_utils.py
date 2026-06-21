"""Base configuration utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, TypeVar

import yaml


ConfigT = TypeVar("ConfigT", bound="PretrainedConfig")


@dataclass
class PretrainedConfig:
    """Base config class with simple serialization helpers."""

    model_type: str = "base"
    pad_token_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
        allowed = cls.__dataclass_fields__.keys()
        filtered = {key: value for key, value in values.items() if key in allowed}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls: type[ConfigT], path: str | Path) -> ConfigT:
        with Path(path).open("r", encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
        values = values.get("model", values)
        return cls.from_dict(values)

    @classmethod
    def from_json(cls: type[ConfigT], path: str | Path) -> ConfigT:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
