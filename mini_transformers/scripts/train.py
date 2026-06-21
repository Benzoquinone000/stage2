"""Dispatch to one of the task training scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


SCRIPTS = {
    "bert": "examples/language_modeling/train_bert_pretraining.py",
    "gpt2": "examples/language_modeling/train_gpt2_lm.py",
    "xlnet": "examples/language_modeling/train_xlnet_lm.py",
    "lm": "examples/language_modeling/train_gpt2_lm.py",
    "bert_pretraining": "examples/language_modeling/train_bert_pretraining.py",
    "xlnet_lm": "examples/language_modeling/train_xlnet_lm.py",
    "classification": "examples/sentiment_analysis/train_bert_classifier.py",
    "translation": "examples/machine_translation/train_transformer_mt.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(SCRIPTS))
    parser.add_argument("--config")
    args, rest = parser.parse_known_args()

    task = args.task
    config_args = []
    if args.config is not None:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        task = task or task_from_config(config)
        config_args = args_from_config(config)
    if task is None:
        parser.error("--task is required unless --config contains task.name")

    root = Path(__file__).resolve().parents[1]
    script = root / SCRIPTS[task]
    subprocess.run([sys.executable, str(script), *config_args, *rest], check=True)


def task_from_config(config: dict) -> str:
    name = config.get("task", {}).get("name")
    mapping = {
        "bert": "bert",
        "gpt2": "gpt2",
        "xlnet": "xlnet",
        "causal_language_modeling": "lm",
        "language_modeling": "lm",
        "bert_pretraining": "bert_pretraining",
        "xlnet_language_modeling": "xlnet_lm",
        "sentiment_analysis": "classification",
        "text_classification": "classification",
        "machine_translation": "translation",
    }
    if name not in mapping:
        raise ValueError(f"Cannot infer task from config task.name={name!r}")
    return mapping[name]


def args_from_config(config: dict) -> list[str]:
    output = []
    key_map = {
        "output_dir": "--output-dir",
        "batch_size": "--batch-size",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "num_epochs": "--epochs",
        "warmup_steps": "--warmup-steps",
        "eval_steps": "--eval-steps",
        "save_steps": "--save-steps",
        "scheduler_type": "--scheduler-type",
        "resume_from_checkpoint": "--resume-from-checkpoint",
    }
    model_key_map = {
        "hidden_size": "--hidden-size",
        "num_hidden_layers": "--layers",
        "num_attention_heads": "--heads",
    }
    data_key_map = {
        "text_file": "--text-file",
        "data_file": "--data-file",
        "valid_file": "--valid-file",
        "max_length": "--max-length",
        "block_size": "--block-size",
    }
    data_config = config.get("data", {})
    for source, mapping in [
        (config.get("training", {}), key_map),
        (config.get("model", {}), model_key_map),
        (data_config, data_key_map),
    ]:
        for key, flag in mapping.items():
            if key in source:
                output.extend([flag, str(source[key])])
    if "max_length" not in data_config:
        max_length = data_config.get("max_source_length", data_config.get("max_target_length"))
        if max_length is not None:
            output.extend(["--max-length", str(max_length)])
    return output


if __name__ == "__main__":
    main()
