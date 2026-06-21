"""F1 metrics."""

from __future__ import annotations

import torch


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, num_labels: int | None = None) -> float:
    preds = logits.argmax(dim=-1)
    if num_labels is None:
        num_labels = int(max(preds.max(), labels.max()).item()) + 1
    scores = []
    for label_id in range(num_labels):
        tp = ((preds == label_id) & (labels == label_id)).sum().item()
        fp = ((preds == label_id) & (labels != label_id)).sum().item()
        fn = ((preds != label_id) & (labels == label_id)).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / max(1, len(scores))
