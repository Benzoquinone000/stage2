"""I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
