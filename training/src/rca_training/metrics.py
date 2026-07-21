"""Validation metrics with strict handling of malformed predictions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .rewards import extract_boxed_answer

VALID_LABELS = frozenset(f"C{i}" for i in range(1, 9))


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    questions: int
    samples: int
    correct_samples: int
    correct_majorities: int

    @property
    def pass_at_1(self) -> float:
        return self.correct_samples / self.samples if self.samples else 0.0

    @property
    def maj_at_4(self) -> float:
        return self.correct_majorities / self.questions if self.questions else 0.0


def parsed_prediction(completion: str) -> str | None:
    prediction = extract_boxed_answer(completion)
    return prediction if prediction in VALID_LABELS else None


def unique_plurality(predictions: Sequence[str | None]) -> str | None:
    counts = Counter(value for value in predictions if value in VALID_LABELS)
    if not counts:
        return None
    highest = max(counts.values())
    winners = [label for label, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 else None


def compute_metrics(
    groups: Sequence[tuple[str, Sequence[str]]], expected_samples: int = 4
) -> EvaluationMetrics:
    correct_samples = 0
    correct_majorities = 0
    for target, completions in groups:
        if len(completions) != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, got {len(completions)}")
        predictions = [parsed_prediction(completion) for completion in completions]
        correct_samples += sum(prediction == target for prediction in predictions)
        correct_majorities += unique_plurality(predictions) == target
    return EvaluationMetrics(
        questions=len(groups),
        samples=len(groups) * expected_samples,
        correct_samples=correct_samples,
        correct_majorities=correct_majorities,
    )
