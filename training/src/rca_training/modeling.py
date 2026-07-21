"""Model, tokenizer, and language-only LoRA construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ModelConfig


def language_lora_module_names(model: Any, targets: tuple[str, ...]) -> list[str]:
    """Resolve exact linear-module names while excluding every visual tower component."""
    excluded = ("vision", "visual", "image", "multimodal", "mm_projector")
    names: list[str] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        lowered = name.lower()
        if leaf in targets and not any(component in lowered for component in excluded):
            if module.__class__.__name__.lower().endswith("linear"):
                names.append(name)
    if not names:
        raise RuntimeError(f"No language-model LoRA modules found for targets {', '.join(targets)}")
    return names


def load_tokenizer(model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(config: ModelConfig, *, training: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    if config.dtype != "bfloat16":
        raise ValueError(f"Only bfloat16 is supported, got {config.dtype!r}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        attn_implementation=config.attention_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = not training
    if training and config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def create_sft_policy(base_model: Any, config: ModelConfig) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = language_lora_module_names(base_model, config.lora_targets)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(base_model, lora)


def load_adapter_policy(base_model: Any, adapter_path: Path, *, trainable: bool) -> Any:
    from peft import PeftModel

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    return PeftModel.from_pretrained(base_model, adapter_path, is_trainable=trainable)


def add_frozen_reference_adapter(model: Any, adapter_path: Path) -> None:
    """Add the immutable initial-SFT adapter beside the trainable default policy adapter."""
    model.load_adapter(adapter_path, adapter_name="reference", is_trainable=False)
    set_active_adapter(model, "default", trainable=True)


def set_active_adapter(model: Any, name: str, *, trainable: bool) -> None:
    model.set_adapter(name)
    for parameter_name, parameter in model.named_parameters():
        if "lora_" in parameter_name:
            parameter.requires_grad = trainable and f".{name}." in parameter_name


def save_adapter(model: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, selected_adapters=["default"])
