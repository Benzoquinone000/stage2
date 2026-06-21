"""Training callback hooks."""

from __future__ import annotations


class Callback:
    def on_train_begin(self, trainer) -> None:
        return None

    def on_step_end(self, trainer, loss) -> None:
        return None

    def on_evaluate(self, trainer, metrics: dict) -> None:
        return None

    def on_epoch_end(self, trainer, metrics: dict) -> None:
        return None

    def on_train_end(self, trainer) -> None:
        return None


class HistoryCallback(Callback):
    def __init__(self) -> None:
        self.losses: list[float] = []
        self.eval_metrics: list[dict] = []

    def on_step_end(self, trainer, loss) -> None:
        self.losses.append(float(loss.item()))

    def on_evaluate(self, trainer, metrics: dict) -> None:
        self.eval_metrics.append(dict(metrics))
