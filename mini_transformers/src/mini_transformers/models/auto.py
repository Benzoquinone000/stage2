"""Small auto-model registry."""

from __future__ import annotations


MODEL_REGISTRY: dict[str, type] = {}


def register_model(model_type: str):
    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator


def get_model_class(model_type: str):
    register_default_models()
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<empty>"
        raise KeyError(f"Unknown model type '{model_type}'. Available: {available}")
    return MODEL_REGISTRY[model_type]


def register_default_models() -> None:
    if MODEL_REGISTRY:
        return
    from .bert import BertForMaskedLM, BertForPreTraining, BertForSequenceClassification, BertModel
    from .gpt2 import GPT2ForCausalLM, GPT2Model
    from .transformer_mt import TransformerForMachineTranslation
    from .xlnet import XLNetLMHeadModel, XLNetModel

    MODEL_REGISTRY.update(
        {
            "bert": BertModel,
            "bert_for_sequence_classification": BertForSequenceClassification,
            "bert_for_masked_lm": BertForMaskedLM,
            "bert_for_pretraining": BertForPreTraining,
            "gpt2": GPT2Model,
            "gpt2_for_causal_lm": GPT2ForCausalLM,
            "xlnet": XLNetModel,
            "xlnet_lm": XLNetLMHeadModel,
            "transformer_mt": TransformerForMachineTranslation,
        }
    )
