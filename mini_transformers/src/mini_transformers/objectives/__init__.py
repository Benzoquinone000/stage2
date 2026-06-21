"""Training objectives."""

from .classification import classification_loss, next_sentence_prediction_loss
from .language_modeling import causal_lm_loss, masked_lm_loss
from .translation import seq2seq_loss

__all__ = [
    "classification_loss",
    "next_sentence_prediction_loss",
    "causal_lm_loss",
    "masked_lm_loss",
    "seq2seq_loss",
]
