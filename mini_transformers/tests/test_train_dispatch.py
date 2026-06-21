from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_train_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = spec_from_file_location("train_script", script_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_model_task_aliases_exist():
    train_script = load_train_script()
    assert train_script.SCRIPTS["bert"].endswith("train_bert_pretraining.py")
    assert train_script.SCRIPTS["gpt2"].endswith("train_gpt2_lm.py")
    assert train_script.SCRIPTS["xlnet"].endswith("train_xlnet_lm.py")


def test_task_from_config_accepts_core_model_names():
    train_script = load_train_script()
    assert train_script.task_from_config({"task": {"name": "bert"}}) == "bert"
    assert train_script.task_from_config({"task": {"name": "gpt2"}}) == "gpt2"
    assert train_script.task_from_config({"task": {"name": "xlnet"}}) == "xlnet"
