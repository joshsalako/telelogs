"""Exact boxed-answer parsing and GRPO reward utilities."""

from __future__ import annotations

import re
from collections.abc import Sequence

BOXED_PATTERN = re.compile(r"\\boxed\{(.*?)\}")


def extract_boxed_answer(text: str) -> str | None:
    """Return the final boxed contents exactly as emitted, or None."""
    matches = BOXED_PATTERN.findall(text)
    return matches[-1] if matches else None


def exact_reward(completion: str, target: str) -> float:
    return 1.0 if extract_boxed_answer(completion) == target else 0.0


def group_rewards(completions: Sequence[str], target: str, expected_size: int = 8) -> list[float]:
    if len(completions) != expected_size:
        raise ValueError(f"Expected {expected_size} completions, got {len(completions)}")
    return [exact_reward(completion, target) for completion in completions]


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Mean/std normalize one complete response group; constant groups map to zero."""
    if not rewards:
        raise ValueError("Cannot compute advantages for an empty group")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = variance**0.5
    if std == 0.0:
        return [0.0] * len(rewards)
    return [(reward - mean) / std for reward in rewards]
