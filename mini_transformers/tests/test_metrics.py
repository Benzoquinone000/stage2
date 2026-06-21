import torch

from mini_transformers.metrics import accuracy


def test_accuracy():
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    labels = torch.tensor([1, 0])
    assert accuracy(logits, labels) == 1.0
