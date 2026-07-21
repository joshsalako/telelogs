"""Strict, deterministic context randomization for the source questions."""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter

from .models import RandomizedExample


CAUSE_PATTERN = re.compile(r"(?m)^(C[1-8]):([^\n]*)$")
DRIVE_HEADING_PATTERN = re.compile(r"user plane drive test data", re.IGNORECASE)
ENGINEERING_HEADING_PATTERN = re.compile(
    r"(?:engineering|engeneering) parameters data", re.IGNORECASE
)
SEPARATOR_CELL_PATTERN = re.compile(r"^:?-{3,}:?$")


class SourceFormatError(ValueError):
    """Raised when a source record cannot be randomized without data loss."""


def _rng_for_item(
    global_seed: int, source_index: int, variant_index: int
) -> random.Random:
    material = f"{global_seed}:{source_index}:{variant_index}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(material).digest(), "big")
    return random.Random(derived_seed)


def _is_markdown_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(SEPARATOR_CELL_PATTERN.fullmatch(cell) for cell in cells)


def _find_heading(lines: list[str], pattern: re.Pattern[str], section: str) -> int:
    matches = [index for index, line in enumerate(lines) if pattern.search(line)]
    if len(matches) != 1:
        raise SourceFormatError(
            f"Expected exactly one {section} heading, found {len(matches)}"
        )
    return matches[0]


def _shuffle_table(
    lines: list[str], heading_index: int, rng: random.Random, section: str
) -> None:
    try:
        header_index = next(
            index
            for index in range(heading_index + 1, len(lines))
            if "|" in lines[index]
        )
    except StopIteration as exc:
        raise SourceFormatError(f"No pipe-delimited table follows {section}") from exc

    end_index = header_index + 1
    while end_index < len(lines) and "|" in lines[end_index]:
        end_index += 1

    separator_count = int(
        header_index + 1 < end_index and _is_markdown_separator(lines[header_index + 1])
    )
    data_start = header_index + 1 + separator_count
    data_rows = lines[data_start:end_index]
    if not data_rows:
        raise SourceFormatError(f"The {section} table has no data rows")

    column_count = len(lines[header_index].split("|"))
    if any(len(row.split("|")) != column_count for row in data_rows):
        raise SourceFormatError(f"Inconsistent column count in the {section} table")

    original_rows = list(data_rows)
    rng.shuffle(data_rows)
    if len(data_rows) > 1 and data_rows == original_rows:
        # A random shuffle may legally return the identity permutation. Avoid
        # leaving a superficial ordering cue unchanged when another ordering
        # is available.
        offset = rng.randrange(1, len(data_rows))
        data_rows = data_rows[offset:] + data_rows[:offset]
    if Counter(data_rows) != Counter(original_rows):  # Defensive invariant.
        raise AssertionError(f"Shuffling changed values in the {section} table")
    lines[data_start:end_index] = data_rows


def randomize_example(
    question: str,
    answer: str,
    source_index: int,
    global_seed: int,
    variant_index: int = 0,
) -> RandomizedExample:
    """Create one deterministic augmentation for a source/variant pair."""

    if not isinstance(question, str) or not isinstance(answer, str):
        raise SourceFormatError("question and answer must both be strings")
    if not re.fullmatch(r"C[1-8]", answer):
        raise SourceFormatError(f"Invalid ground-truth answer: {answer!r}")

    cause_matches = list(CAUSE_PATTERN.finditer(question))
    cause_ids = [match.group(1) for match in cause_matches]
    expected_ids = [f"C{number}" for number in range(1, 9)]
    if cause_ids != expected_ids:
        raise SourceFormatError(
            "Expected one ordered cause line for each identifier C1 through C8"
        )

    if variant_index < 0:
        raise ValueError("variant_index must be non-negative")

    rng = _rng_for_item(global_seed, source_index, variant_index)
    randomized_ids = [f"R{number}" for number in range(1, 9)]
    rng.shuffle(randomized_ids)
    if randomized_ids == [f"R{number}" for number in range(1, 9)]:
        offset = rng.randrange(1, len(randomized_ids))
        randomized_ids = randomized_ids[offset:] + randomized_ids[:offset]
    mapping = dict(zip(expected_ids, randomized_ids, strict=True))

    def replace_cause(match: re.Match[str]) -> str:
        return f"{mapping[match.group(1)]}:{match.group(2)}"

    randomized_question = CAUSE_PATTERN.sub(replace_cause, question)
    had_trailing_newline = randomized_question.endswith("\n")
    lines = randomized_question.splitlines()

    drive_heading = _find_heading(lines, DRIVE_HEADING_PATTERN, "drive-test")
    engineering_heading = _find_heading(
        lines, ENGINEERING_HEADING_PATTERN, "engineering"
    )
    if drive_heading >= engineering_heading:
        raise SourceFormatError("Drive-test table must precede engineering table")

    # Work from the later table first so index positions remain valid.
    _shuffle_table(lines, engineering_heading, rng, "engineering")
    _shuffle_table(lines, drive_heading, rng, "drive-test")
    randomized_question = "\n".join(lines) + ("\n" if had_trailing_newline else "")

    return RandomizedExample(
        source_index=source_index,
        question=randomized_question,
        answer=mapping[answer],
        identifier_mapping=mapping,
        variant_index=variant_index,
    )
