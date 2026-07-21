"""Shared conversation templates for training and evaluation."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a telecom root-cause analysis expert. Analyze the evidence carefully, "
    "then finish with exactly one selected label enclosed as \\boxed{C#}."
)


def prompt_messages(question: str) -> list[dict[str, str]]:
    """Return the identical system/user conversation used by SFT, GRPO, and evaluation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def completion_messages(response: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": response}]


def render_prompt(tokenizer: object, question: str) -> str:
    """Render a generation-ready prompt using the model's native chat template."""
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        raise TypeError("Tokenizer has no apply_chat_template method")
    return apply(prompt_messages(question), tokenize=False, add_generation_prompt=True)
