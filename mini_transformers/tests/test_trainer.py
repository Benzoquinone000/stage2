import torch
from torch.utils.data import Dataset

from mini_transformers.training import HistoryCallback, Trainer, TrainingArguments


def test_training_args_defaults():
    args = TrainingArguments()
    assert args.batch_size > 0


def test_training_args_scheduler_fields():
    args = TrainingArguments(warmup_steps=2, eval_steps=5)
    assert args.warmup_steps == 2
    assert args.eval_steps == 5


class TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {"x": torch.tensor([float(idx)]), "labels": torch.tensor(idx % 2)}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 2)

    def forward(self, x, labels=None):
        logits = self.proj(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}


def test_trainer_callbacks_and_checkpoint(tmp_path):
    history = HistoryCallback()
    trainer = Trainer(
        model=TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            batch_size=1,
            num_epochs=1,
            learning_rate=1e-3,
            warmup_steps=0,
            save_steps=1,
            scheduler_type="linear",
        ),
        train_dataset=TinyDataset(),
        eval_dataset=TinyDataset(),
        callbacks=[history],
    )
    trainer.train()
    assert history.losses
    assert history.eval_metrics
    checkpoints = sorted(tmp_path.glob("checkpoint_step_*.pt"))
    assert checkpoints

    resumed = Trainer(
        model=TinyModel(),
        args=TrainingArguments(
            output_dir=str(tmp_path / "resumed"),
            batch_size=1,
            num_epochs=1,
            learning_rate=1e-3,
            warmup_steps=0,
            resume_from_checkpoint=str(checkpoints[-1]),
        ),
        train_dataset=TinyDataset(),
    )
    resumed.train()
    assert resumed.global_step > 2
