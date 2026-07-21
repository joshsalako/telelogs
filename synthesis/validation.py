"""Answer parsing, strict voting, and formatter-output validation."""

from __future__ import annotations

import re
from collections import Counter

from .client import MalformedModelOutput
from .models import Trajectory


BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*(R[1-8])\s*\}", re.IGNORECASE)
CANDIDATE_PATTERN = re.compile(r"\bR[1-8]\b", re.IGNORECASE)
FORMAT_HEADINGS = (
    "Task 1: Data analysis",
    "Task 2: Root cause analysis",
    "Task 3: Root cause identification",
    "Summary",
)


def parse_boxed_answer(text: str) -> str:
    """Return the last valid randomized identifier in a boxed expression."""

    matches = BOXED_PATTERN.findall(text)
    if not matches:
        raise MalformedModelOutput("No valid \\boxed{R1}-\\boxed{R8} answer found")
    return matches[-1].upper()


def validate_reasoning_output(text: str) -> None:
    parse_boxed_answer(text)


def strict_majority(
    trajectories: list[Trajectory], ground_truth: str, expected_count: int
) -> tuple[str | None, str]:
    """Accept only a >50% vote that also equals the randomized ground truth."""

    if len(trajectories) != expected_count:
        return None, "incomplete_agent_set"
    counts = Counter(trajectory.prediction for trajectory in trajectories)
    winner, votes = counts.most_common(1)[0]
    if votes <= expected_count // 2:
        return None, "no_strict_majority"
    if winner != ground_truth:
        return None, "incorrect_majority"
    return winner, "accepted"


def select_most_comprehensive(trajectories: list[Trajectory]) -> Trajectory:
    """Prefer candidate coverage, then non-whitespace trajectory length."""

    if not trajectories:
        raise ValueError("Cannot select from an empty trajectory list")

    def score(trajectory: Trajectory) -> tuple[int, int]:
        candidates = {
            item.upper() for item in CANDIDATE_PATTERN.findall(trajectory.text)
        }
        compact_length = len(re.sub(r"\s+", "", trajectory.text))
        return len(candidates), compact_length

    return max(trajectories, key=score)


def validate_formatted_output(text: str, expected_answer: str) -> None:
    if not text.lstrip().startswith(FORMAT_HEADINGS[0]):
        raise MalformedModelOutput(
            "Formatted output must begin with the Task 1 heading"
        )
    positions: list[int] = []
    for heading in FORMAT_HEADINGS:
        matches = list(re.finditer(rf"(?m)^{re.escape(heading)}\s*$", text))
        if len(matches) != 1:
            raise MalformedModelOutput(
                f"Expected exactly one heading {heading!r}, found {len(matches)}"
            )
        positions.append(matches[0].start())
    if positions != sorted(positions):
        raise MalformedModelOutput("Formatter headings are out of order")

    boxed_matches = BOXED_PATTERN.findall(text)
    if len(boxed_matches) != 1:
        raise MalformedModelOutput(
            "Formatted output must contain exactly one boxed answer"
        )
    if boxed_matches[0].upper() != expected_answer:
        raise MalformedModelOutput(
            f"Formatter changed the answer from {expected_answer}"
        )
    if not re.search(
        rf"\\boxed\s*\{{\s*{re.escape(expected_answer)}\s*\}}\s*$",
        text,
        re.IGNORECASE,
    ):
        raise MalformedModelOutput("The Summary must end with the boxed answer")
