"""Crash-tolerant JSONL output and sidecar checkpoint management."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TextIO

from .config import Settings


class ResumeStateError(RuntimeError):
    """Raised when existing output cannot safely be resumed."""


class OutputStore:
    """Serialize writes and track terminal source indices for safe resume."""

    def __init__(self, settings: Settings, input_digest: str) -> None:
        self._settings = settings
        self._metadata = {
            "kind": "metadata",
            "input_sha256": input_digest,
            "random_seed": settings.random_seed,
            "pipeline_version": settings.pipeline_version,
            "model_name": settings.model_name,
            "agents_per_item": settings.agents_per_item,
            "augmentations_per_item": settings.augmentations_per_item,
            "output_schema": "question_response_v1",
        }
        self._lock = asyncio.Lock()
        self.completed_variants: set[tuple[int, int]] = set()
        self.output_questions: set[str] = set()
        self._output_file: TextIO | None = None
        self._state_file: TextIO | None = None

    def open(self) -> None:
        self._settings.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_output()
        self._load_or_create_state()
        self._output_file = self._settings.output_path.open("a", encoding="utf-8")
        self._state_file = self._settings.state_path.open("a", encoding="utf-8")

    def close(self) -> None:
        for handle in (self._output_file, self._state_file):
            if handle is not None:
                handle.flush()
                handle.close()
        self._output_file = None
        self._state_file = None

    def _load_output(self) -> None:
        path = self._settings.output_path
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResumeStateError(
                        f"Malformed output JSON at {path}:{line_number}"
                    ) from exc
                if set(record) != {"question", "response"} or not all(
                    isinstance(record[key], str) for key in ("question", "response")
                ):
                    raise ResumeStateError(
                        f"Unexpected output schema at {path}:{line_number}"
                    )
                if record["question"] in self.output_questions:
                    raise ResumeStateError(
                        f"Duplicate question at {path}:{line_number}"
                    )
                self.output_questions.add(record["question"])

    def _load_or_create_state(self) -> None:
        path = self._settings.state_path
        if not path.exists():
            if self._settings.output_path.exists() and self.output_questions:
                raise ResumeStateError(
                    "Output exists without a checkpoint; refusing an unsafe resume"
                )
            with path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(self._metadata, ensure_ascii=False) + "\n")
                handle.flush()
            return

        metadata_seen = False
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResumeStateError(
                        f"Malformed checkpoint JSON at {path}:{line_number}"
                    ) from exc
                if not metadata_seen:
                    if event != self._metadata:
                        raise ResumeStateError(
                            "Checkpoint metadata differs from the current run configuration"
                        )
                    metadata_seen = True
                    continue
                if (
                    event.get("kind") != "completed"
                    or not isinstance(event.get("source_index"), int)
                    or not isinstance(event.get("variant_index"), int)
                ):
                    raise ResumeStateError(
                        f"Unexpected checkpoint event at {path}:{line_number}"
                    )
                self.completed_variants.add(
                    (event["source_index"], event["variant_index"])
                )
        if not metadata_seen:
            raise ResumeStateError("Checkpoint contains no metadata record")

    async def recover_accepted(self, source_index: int, variant_index: int) -> None:
        """Checkpoint an output line written just before a prior process crash."""

        async with self._lock:
            key = (source_index, variant_index)
            if key not in self.completed_variants:
                self._write_state(
                    source_index,
                    variant_index,
                    "accepted",
                    "recovered_from_output",
                )

    async def append_success(
        self,
        source_index: int,
        variant_index: int,
        question: str,
        response: str,
    ) -> None:
        async with self._lock:
            if self._output_file is None:
                raise RuntimeError("OutputStore is not open")
            if question in self.output_questions:
                raise ResumeStateError(
                    "Refusing to append a duplicate randomized question"
                )
            record = {"question": question, "response": response}
            self._output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._output_file.flush()
            self.output_questions.add(question)
            # Output is flushed first: a crash between these writes is recovered by
            # matching the deterministic randomized question on the next run.
            self._write_state(
                source_index, variant_index, "accepted", "strict_majority"
            )

    async def append_discard(
        self, source_index: int, variant_index: int, reason: str
    ) -> None:
        async with self._lock:
            self._write_state(source_index, variant_index, "discarded", reason)

    def _write_state(
        self,
        source_index: int,
        variant_index: int,
        status: str,
        reason: str,
    ) -> None:
        if self._state_file is None:
            raise RuntimeError("OutputStore is not open")
        key = (source_index, variant_index)
        if key in self.completed_variants:
            return
        event = {
            "kind": "completed",
            "source_index": source_index,
            "variant_index": variant_index,
            "status": status,
            "reason": reason,
        }
        self._state_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._state_file.flush()
        self.completed_variants.add(key)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
