"""Strict loaders for synthetic SFT, original GRPO, and held-out validation data."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .metrics import VALID_LABELS
from .prompts import completion_messages, prompt_messages


class DataValidationError(ValueError):
    """A data file violates the alignment pipeline contract."""


@dataclass(frozen=True, slots=True)
class SFTRecord:
    question: str
    response: str


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    identifier: str
    question: str
    answer: str


TokenCounter = Callable[[str], int]
_TARGET_SUFFIX = re.compile(r"_(?P<variant>[1-4])$")


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{location} must be a non-empty string")
    return value


def _valid_label(value: Any, location: str) -> str:
    label = _nonempty_string(value, location)
    if label not in VALID_LABELS:
        raise DataValidationError(f"{location} must be one of C1-C8; got {label!r}")
    return label


def _check_length(
    text: str,
    location: str,
    token_counter: TokenCounter | None,
    max_tokens: int | None,
) -> None:
    if (token_counter is None) != (max_tokens is None):
        raise ValueError("token_counter and max_tokens must be supplied together")
    if token_counter is not None and max_tokens is not None:
        length = token_counter(text)
        if length > max_tokens:
            raise DataValidationError(f"{location} is {length} tokens; maximum is {max_tokens}")


def load_sft_records(
    path: Path,
    *,
    token_counter: TokenCounter | None = None,
    max_tokens: int | None = None,
) -> list[SFTRecord]:
    if not path.is_file():
        raise DataValidationError(
            f"SFT data not found: {path}. Run the synthesis pipeline before stage-one training."
        )
    records: list[SFTRecord] = []
    seen_questions: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise DataValidationError(f"{path}:{line_number} is empty")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataValidationError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(item, dict):
                raise DataValidationError(f"{path}:{line_number} must contain a JSON object")
            unknown = set(item) - {"question", "response", "answer"}
            if unknown:
                raise DataValidationError(
                    f"{path}:{line_number} has unexpected fields: {sorted(unknown)}"
                )
            question = _nonempty_string(item.get("question"), f"{path}:{line_number}.question")
            if "response" in item and "answer" in item:
                raise DataValidationError(
                    f"{path}:{line_number} must use response or legacy answer, not both"
                )
            response = _nonempty_string(
                item.get("response", item.get("answer")), f"{path}:{line_number}.response"
            )
            if question in seen_questions:
                raise DataValidationError(f"{path}:{line_number} duplicates an earlier question")
            _check_length(question + response, f"{path}:{line_number}", token_counter, max_tokens)
            seen_questions.add(question)
            records.append(SFTRecord(question=question, response=response))
    if not records:
        raise DataValidationError(f"{path} contains no SFT records")
    return records


def load_training_records(
    path: Path,
    *,
    token_counter: TokenCounter | None = None,
    max_tokens: int | None = None,
) -> list[TrainingRecord]:
    if not path.is_file():
        raise DataValidationError(f"Training data not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataValidationError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(payload, list) or not payload:
        raise DataValidationError(f"{path} must contain a non-empty JSON array")
    records: list[TrainingRecord] = []
    seen_questions: set[str] = set()
    for index, item in enumerate(payload):
        location = f"{path}[{index}]"
        if not isinstance(item, dict):
            raise DataValidationError(f"{location} must be an object")
        if set(item) != {"question", "answer"}:
            raise DataValidationError(
                f"{location} must contain only raw question and answer fields; got {sorted(item)}"
            )
        question = _nonempty_string(item["question"], f"{location}.question")
        answer = _valid_label(item["answer"], f"{location}.answer")
        if question in seen_questions:
            raise DataValidationError(f"{location} duplicates an earlier question")
        _check_length(question, location, token_counter, max_tokens)
        seen_questions.add(question)
        records.append(TrainingRecord(question=question, answer=answer))
    return records


def strip_validation_suffix(identifier: str) -> tuple[str, int]:
    match = _TARGET_SUFFIX.search(identifier)
    if match is None:
        raise DataValidationError(
            f"Validation target ID {identifier!r} must end in exactly _1, _2, _3, or _4"
        )
    return identifier[: match.start()], int(match.group("variant"))


def load_validation_records(question_path: Path, target_path: Path) -> list[ValidationRecord]:
    if not question_path.is_file():
        raise DataValidationError(f"Validation questions not found: {question_path}")
    if not target_path.is_file():
        raise DataValidationError(f"Validation targets not found: {target_path}")

    questions: dict[str, str] = {}
    with question_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "question"]:
            raise DataValidationError(
                f"{question_path} headers must be exactly ID,question; got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, 2):
            identifier = _nonempty_string(row.get("ID"), f"{question_path}:{row_number}.ID")
            question = _nonempty_string(
                row.get("question"), f"{question_path}:{row_number}.question"
            )
            if identifier in questions:
                raise DataValidationError(
                    f"{question_path}:{row_number} duplicates ID {identifier!r}"
                )
            questions[identifier] = question
    if not questions:
        raise DataValidationError(f"{question_path} contains no validation questions")

    targets: dict[str, dict[int, str]] = {}
    seen_target_ids: set[str] = set()
    with target_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ID", "Target"]:
            raise DataValidationError(
                f"{target_path} headers must be exactly ID,Target; got {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, 2):
            target_id = _nonempty_string(row.get("ID"), f"{target_path}:{row_number}.ID")
            if target_id in seen_target_ids:
                raise DataValidationError(f"{target_path}:{row_number} duplicates ID {target_id!r}")
            seen_target_ids.add(target_id)
            base_id, variant = strip_validation_suffix(target_id)
            label = _valid_label(row.get("Target"), f"{target_path}:{row_number}.Target")
            targets.setdefault(base_id, {})[variant] = label

    question_ids = set(questions)
    target_ids = set(targets)
    missing_targets = sorted(question_ids - target_ids)
    extra_targets = sorted(target_ids - question_ids)
    if missing_targets or extra_targets:
        raise DataValidationError(
            "Validation base-ID coverage mismatch: "
            f"missing targets={missing_targets[:5]}, targets without questions={extra_targets[:5]}"
        )

    records: list[ValidationRecord] = []
    for identifier, question in questions.items():
        variants = targets[identifier]
        if set(variants) != {1, 2, 3, 4}:
            raise DataValidationError(
                f"Validation ID {identifier!r} must have suffixes _1 through _4 exactly; "
                f"got {sorted(variants)}"
            )
        labels = set(variants.values())
        if len(labels) != 1:
            raise DataValidationError(
                f"Validation ID {identifier!r} has inconsistent targets: {sorted(labels)}"
            )
        records.append(
            ValidationRecord(identifier=identifier, question=question, answer=labels.pop())
        )
    return records


def format_sft_record(tokenizer: Any, record: SFTRecord, max_length: int) -> dict[str, list[int]]:
    """Tokenize a conversation and mask every prompt token from cross-entropy loss."""
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages(record.question), tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        prompt_messages(record.question) + completion_messages(record.response),
        tokenize=True,
        add_generation_prompt=False,
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if hasattr(full_ids, "tolist"):
        full_ids = full_ids.tolist()
    if len(full_ids) > max_length:
        raise DataValidationError(
            f"SFT example is {len(full_ids)} tokens; maximum combined length is {max_length}"
        )
    if len(full_ids) <= len(prompt_ids):
        raise DataValidationError("SFT chat template produced no assistant completion tokens")
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise DataValidationError(
            "Tokenizer chat template is not prefix-stable between prompt and completion"
        )
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :]),
    }
