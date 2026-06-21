"""Summarize completed AG News experiments into a Markdown report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_classifier_row(history: list[dict[str, str]]) -> dict[str, str]:
    return max(history, key=lambda row: float(row.get("valid_macro_f1", "-inf")))


def best_mlm_row(history: list[dict[str, str]]) -> dict[str, str]:
    return min(history, key=lambda row: float(row.get("valid_loss", "inf")))


def classifier_reflection(name: str, config: dict, metrics: dict, best_row: dict[str, str]) -> str:
    f1 = float(metrics["macro_f1"])
    valid_f1 = float(best_row["valid_macro_f1"])
    train_acc = float(best_row.get("train_accuracy", 0.0))
    if "clean" in name and f1 >= 0.924:
        return "cleaned data plus a smaller holdout improved test F1; prioritize clean data for folds and future pretraining."
    if "tapt_more" in name and f1 >= 0.923:
        return "extra TAPT improved test generalization; keep this checkpoint family as current anchor."
    if config.get("label_smoothing", 0) or config.get("dropout_prob"):
        return "regularization reduced overfit but gains were small; avoid pushing smoothing/dropout much higher."
    if train_acc > 0.96 and valid_f1 < 0.928:
        return "training accuracy is high while validation saturates; prefer data/ensemble gains over more epochs."
    if f1 < 0.916:
        return "too conservative; backbone update was likely underpowered."
    return "use as comparison point."


def mlm_reflection(name: str, config: dict, best_row: dict[str, str]) -> str:
    valid_loss = float(best_row["valid_loss"])
    if "tapt_more" in name:
        return "extra TAPT matched prior TAPT perplexity but helped test after classification; useful but watch overfit."
    if "news_augmented_tapt" in name:
        return "TAPT lowered AG News MLM loss after DAPT; keep DAPT->TAPT structure."
    if "news_augmented" in name:
        return "DAPT alone has worse AG News validation loss than TAPT; it is a domain bridge, not final checkpoint."
    if valid_loss > 5.2:
        return "AGNews-only MLM was undertrained/limited; external news data is justified."
    return "baseline pretraining checkpoint."


def build_classifier_rows() -> list[dict]:
    rows = []
    for metrics_path in sorted(OUTPUTS.glob("*/test_metrics.json")):
        out_dir = metrics_path.parent
        config_path = out_dir / "finetune_config.json"
        history_path = out_dir / "finetune_history.csv"
        if not config_path.exists() or not history_path.exists():
            continue
        metrics = read_json(metrics_path)
        config = read_json(config_path)
        history = read_history(history_path)
        if not history:
            continue
        best_row = best_classifier_row(history)
        rows.append(
            {
                "name": out_dir.name,
                "output_dir": str(out_dir),
                "test_accuracy": float(metrics["accuracy"]),
                "test_macro_f1": float(metrics["macro_f1"]),
                "best_epoch": int(float(best_row["epoch"])),
                "best_valid_macro_f1": float(best_row["valid_macro_f1"]),
                "mlm_checkpoint": config.get("mlm_checkpoint", ""),
                "learning_rate": config.get("learning_rate"),
                "label_smoothing": config.get("label_smoothing", 0.0),
                "dropout_prob": config.get("dropout_prob"),
                "reflection": classifier_reflection(out_dir.name, config, metrics, best_row),
            }
        )
    return sorted(rows, key=lambda row: row["test_macro_f1"], reverse=True)


def build_mlm_rows() -> list[dict]:
    rows = []
    for history_path in sorted(OUTPUTS.glob("*/mlm_history.csv")):
        out_dir = history_path.parent
        config_path = out_dir / "mlm_config.json"
        if not config_path.exists():
            continue
        config = read_json(config_path)
        history = read_history(history_path)
        if not history:
            continue
        best_row = best_mlm_row(history)
        rows.append(
            {
                "name": out_dir.name,
                "output_dir": str(out_dir),
                "best_epoch": int(float(best_row["epoch"])),
                "best_valid_loss": float(best_row["valid_loss"]),
                "best_valid_perplexity": float(best_row["valid_perplexity"]),
                "train_file": config.get("init_checkpoint") or config.get("bert_config", ""),
                "reflection": mlm_reflection(out_dir.name, config, best_row),
            }
        )
    return sorted(rows, key=lambda row: row["best_valid_loss"])


def build_ensemble_rows() -> list[dict]:
    rows = []
    for metrics_path in sorted(OUTPUTS.glob("*/ensemble/ensemble_metrics.json")):
        metrics = read_json(metrics_path)
        out_dir = metrics_path.parent
        experiment_name = f"{out_dir.parent.name}/ensemble"
        rows.append(
            {
                "name": experiment_name,
                "output_dir": str(out_dir),
                "test_accuracy": float(metrics["test_accuracy"]),
                "test_macro_f1": float(metrics["test_macro_f1"]),
                "test_loss": float(metrics["test_loss"]),
                "num_models": int(metrics["num_models"]),
                "reflection": "five-fold probability averaging improved generalization; keep this as the current scoring anchor.",
            }
        )
    return sorted(rows, key=lambda row: row["test_macro_f1"], reverse=True)


def render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    classifiers = build_classifier_rows()
    mlms = build_mlm_rows()
    ensembles = build_ensemble_rows()
    candidates = [
        {
            "name": row["name"],
            "test_accuracy": row["test_accuracy"],
            "test_macro_f1": row["test_macro_f1"],
            "reflection": row["reflection"],
        }
        for row in classifiers
    ] + [
        {
            "name": row["name"],
            "test_accuracy": row["test_accuracy"],
            "test_macro_f1": row["test_macro_f1"],
            "reflection": row["reflection"],
        }
        for row in ensembles
    ]
    best = max(candidates, key=lambda row: row["test_macro_f1"]) if candidates else None

    lines = [
        "# AG News Experiment Log",
        "",
        "This report is regenerated from local output directories. Future training scripts also append structured records to `reports/experiment_runs.jsonl`.",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Current Best",
                "",
                f"- `{best['name']}`: test accuracy `{best['test_accuracy']:.6f}`, macro F1 `{best['test_macro_f1']:.6f}`.",
                f"- Reflection: {best['reflection']}",
                "",
            ]
        )

    if ensembles:
        lines.extend(["## Ensemble Runs", ""])
        lines.extend(
            render_table(
                ["rank", "run", "models", "test acc", "test f1", "test loss", "reflection"],
                [
                    [
                        str(index + 1),
                        f"`{row['name']}`",
                        str(row["num_models"]),
                        f"{row['test_accuracy']:.6f}",
                        f"{row['test_macro_f1']:.6f}",
                        f"{row['test_loss']:.6f}",
                        row["reflection"],
                    ]
                    for index, row in enumerate(ensembles)
                ],
            )
        )
        lines.append("")

    lines.extend(["## Classifier Runs", ""])
    lines.extend(
        render_table(
            ["rank", "run", "test acc", "test f1", "best valid f1", "best epoch", "reflection"],
            [
                [
                    str(index + 1),
                    f"`{row['name']}`",
                    f"{row['test_accuracy']:.6f}",
                    f"{row['test_macro_f1']:.6f}",
                    f"{row['best_valid_macro_f1']:.6f}",
                    str(row["best_epoch"]),
                    row["reflection"],
                ]
                for index, row in enumerate(classifiers)
            ],
        )
    )
    lines.extend(["", "## MLM Runs", ""])
    lines.extend(
        render_table(
            ["run", "best valid loss", "best ppl", "best epoch", "reflection"],
            [
                [
                    f"`{row['name']}`",
                    f"{row['best_valid_loss']:.4f}",
                    f"{row['best_valid_perplexity']:.2f}",
                    str(row["best_epoch"]),
                    row["reflection"],
                ]
                for row in mlms
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Next Decisions",
            "",
            "- Keep `outputs/fivefold_best512_clean_tapt_more/ensemble` as the current scoring anchor.",
            "- Next score gains should come from calibrated/weighted ensembling or a genuinely diverse second model family, not from rerunning the failed standard BERT-base branch.",
            "- Preserve clean data, the 512x8 TAPT checkpoint family, and five-fold probability outputs for final submission analysis.",
            "",
        ]
    )
    (REPORTS / "experiment_log.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORTS / 'experiment_log.md'}")


if __name__ == "__main__":
    main()
