"""Copy a checkpoint directory to a new location."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.checkpoint)
    dst = Path(args.output)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"exported {src} to {dst}")


if __name__ == "__main__":
    main()
