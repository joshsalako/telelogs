"""Answer parsing, strict voting, and formatter-output validation."""

from __future__ import annotations

import re
from collections import Counter

from .client import MalformedModelOutput
from .models import Trajectory


BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*(R[1-8])\s*\}", re.IGNORECASE)
FINAL_BOXED_PATTERN = re.compile(r"\\boxed\s*\{\s*(R[1-8])\s*\}\s*$", re.IGNORECASE)
CANDIDATE_PATTERN = re.compile(r"\bR[1-8]\b", re.IGNORECASE)
RECONSIDERATION_PATTERN = re.compile(
    r"\b(?:wait|reconsider|check again|double-check|might have missed|let me revisit)\b",
    re.IGNORECASE,
)
FORMAT_HEADINGS = (
    "Task 1: Data analysis",
    "Task 2: Root cause analysis",
    "Task 3: Root cause identification",
    "Summary",
)
TABLE_COLUMNS = (
    "Candidate",
    "Required check",
    "Observed evidence/calculation",
    "Verdict",
)
VALID_VERDICTS = {"Supported", "Ruled out", "Present but secondary"}


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
    positions: list[int] = []
    for heading in FORMAT_HEADINGS:
        idx = text.find(heading)
        if idx == -1:
            raise MalformedModelOutput(f"Expected heading {heading!r} not found")
        positions.append(idx)
    if positions != sorted(positions):
        raise MalformedModelOutput("Formatter headings are out of order")

    task_2 = text[positions[1] + len(FORMAT_HEADINGS[1]) : positions[2]].strip()
    table_lines = [line.strip() for line in task_2.splitlines() if "|" in line]
    if len(table_lines) != 10:
        raise MalformedModelOutput(
            "Task 2 must contain one table header, one separator, and eight rows"
        )

    def table_cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    if tuple(table_cells(table_lines[0])) != TABLE_COLUMNS:
        raise MalformedModelOutput(
            f"Task 2 table columns must be exactly: {TABLE_COLUMNS}"
        )
    separator_cells = table_cells(table_lines[1])
    if len(separator_cells) != len(TABLE_COLUMNS) or not all(
        re.fullmatch(r":?-{3,}:?", cell) for cell in separator_cells
    ):
        raise MalformedModelOutput("Task 2 table has an invalid separator row")

    rows = [table_cells(line) for line in table_lines[2:]]
    if any(len(row) != len(TABLE_COLUMNS) for row in rows):
        raise MalformedModelOutput("Every Task 2 table row must have four columns")
    candidates = [row[0].upper() for row in rows]
    if set(candidates) != {f"R{i}" for i in range(1, 9)} or len(candidates) != 8:
        raise MalformedModelOutput(
            "Task 2 must contain each randomized candidate R1-R8 exactly once"
        )
    invalid_verdicts = {row[3] for row in rows} - VALID_VERDICTS
    if invalid_verdicts:
        raise MalformedModelOutput(
            f"Task 2 contains invalid verdicts: {sorted(invalid_verdicts)}"
        )
    supported = [row[0].upper() for row in rows if row[3] == "Supported"]
    if supported != [expected_answer]:
        raise MalformedModelOutput(
            "Task 2 must mark exactly the validated candidate as Supported"
        )

    if RECONSIDERATION_PATTERN.search(text):
        raise MalformedModelOutput(
            "Formatted output contains prohibited reconsideration language"
        )

    boxed_matches = BOXED_PATTERN.findall(text)
    if len(boxed_matches) != 1:
        raise MalformedModelOutput(
            "Formatted output must contain exactly one boxed answer"
        )
    if boxed_matches[0].upper() != expected_answer:
        raise MalformedModelOutput(
            f"Formatter changed the answer from {expected_answer}"
        )
    final_match = FINAL_BOXED_PATTERN.search(text)
    if final_match is None or final_match.group(1).upper() != expected_answer:
        raise MalformedModelOutput(
            "The Summary must end with the expected boxed answer"
        )
