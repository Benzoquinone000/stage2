from mini_transformers.data.dataset_registry import get_dataset_builder
from mini_transformers.models.auto import get_model_class


def test_model_registry_defaults():
    assert get_model_class("gpt2_for_causal_lm").__name__ == "GPT2ForCausalLM"


def test_dataset_registry_defaults():
    assert callable(get_dataset_builder("classification_tsv"))
