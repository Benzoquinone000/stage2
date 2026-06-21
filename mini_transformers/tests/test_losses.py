import torch

from mini_transformers.objectives import causal_lm_loss, masked_lm_loss, next_sentence_prediction_loss


def test_causal_lm_loss():
    logits = torch.randn(2, 3, 5)
    labels = torch.ones(2, 3, dtype=torch.long)
    loss = causal_lm_loss(logits, labels)
    assert loss.item() > 0


def test_masked_lm_and_nsp_losses():
    logits = torch.randn(2, 3, 5)
    labels = torch.ones(2, 3, dtype=torch.long)
    assert masked_lm_loss(logits, labels).item() > 0

    nsp_logits = torch.randn(2, 2)
    nsp_labels = torch.tensor([0, 1])
    assert next_sentence_prediction_loss(nsp_logits, nsp_labels).item() > 0
