"""Bounded asynchronous orchestration for the five-stage synthesis pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from .client import VLLMClient
from .config import Settings
from .models import RandomizedExample, Strategy, Trajectory
from .prompts import formatter_messages, reasoning_messages
from .randomization import SourceFormatError, randomize_example
from .store import OutputStore, sha256_file
from .validation import (
    parse_boxed_answer,
    select_most_comprehensive,
    strict_majority,
    validate_formatted_output,
    validate_reasoning_output,
)


LOGGER = logging.getLogger("synthesis")


@dataclass(slots=True)
class Progress:
    total: int
    counts: Counter[str] = field(default_factory=Counter)

    def record(self, status: str) -> int:
        self.counts[status] += 1
        self.counts["handled"] += 1
        return self.counts["handled"]


def _request_seed(
    global_seed: int, source_index: int, variant_index: int, request_name: str
) -> int:
    material = f"{global_seed}:{source_index}:{variant_index}:{request_name}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def _load_source(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        data: Any = json.load(handle)
    if not isinstance(data, list):
        raise SourceFormatError("train.json must contain a JSON array")
    for index, record in enumerate(data):
        if not isinstance(record, dict) or set(record) != {"question", "answer"}:
            raise SourceFormatError(
                f"Source record {index} must contain only question and answer keys"
            )
        if not all(isinstance(record[key], str) for key in ("question", "answer")):
            raise SourceFormatError(f"Source record {index} values must be strings")
    return data


async def _generate_trajectory(
    client: VLLMClient,
    settings: Settings,
    example: RandomizedExample,
    strategy: Strategy,
    agent_index: int,
) -> Trajectory:
    text = await client.chat(
        reasoning_messages(example.question, strategy),
        temperature=settings.reasoning_temperature,
        top_p=settings.reasoning_top_p,
        max_tokens=settings.reasoning_max_tokens,
        seed=_request_seed(
            settings.random_seed,
            example.source_index,
            example.variant_index,
            f"agent:{agent_index}",
        ),
        validator=validate_reasoning_output,
    )
    return Trajectory(
        strategy=strategy,
        text=text,
        prediction=parse_boxed_answer(text),
    )


async def _format_trajectory(
    client: VLLMClient,
    settings: Settings,
    example: RandomizedExample,
    trajectory: Trajectory,
) -> str:
    validator = partial(validate_formatted_output, expected_answer=example.answer)
    return await client.chat(
        formatter_messages(example.question, trajectory.text, example.answer),
        temperature=settings.formatting_temperature,
        top_p=settings.formatting_top_p,
        max_tokens=settings.formatting_max_tokens,
        seed=_request_seed(
            settings.random_seed,
            example.source_index,
            example.variant_index,
            "formatter",
        ),
        include_reasoning_content=False,
        validator=validator,
    )


async def _process_example(
    client: VLLMClient,
    store: OutputStore,
    settings: Settings,
    example: RandomizedExample,
) -> str:
    half = settings.agents_per_item // 2
    strategies = [Strategy.ELIMINATION] * half + [Strategy.CONTRADICTION] * half
    results = await asyncio.gather(
        *(
            _generate_trajectory(client, settings, example, strategy, agent_index)
            for agent_index, strategy in enumerate(strategies)
        ),
        return_exceptions=True,
    )
    failures = [result for result in results if isinstance(result, BaseException)]
    if failures:
        reason = f"agent_generation_failure:{type(failures[0]).__name__}"
        await store.append_discard(example.source_index, example.variant_index, reason)
        return reason

    trajectories = [result for result in results if isinstance(result, Trajectory)]
    winner, vote_status = strict_majority(
        trajectories, example.answer, settings.agents_per_item
    )
    if winner is None:
        await store.append_discard(
            example.source_index, example.variant_index, vote_status
        )
        return vote_status

    winning_group = [
        trajectory for trajectory in trajectories if trajectory.prediction == winner
    ]
    selected = select_most_comprehensive(winning_group)
    try:
        formatted = await _format_trajectory(client, settings, example, selected)
    except Exception as exc:  # Retries are exhausted inside VLLMClient.chat.
        reason = f"formatter_failure:{type(exc).__name__}"
        await store.append_discard(example.source_index, example.variant_index, reason)
        return reason

    await store.append_success(
        example.source_index, example.variant_index, example.question, formatted
    )
    return "accepted"


async def _worker(
    worker_id: int,
    queue: asyncio.Queue[tuple[int, int]],
    source: list[dict[str, str]],
    client: VLLMClient,
    store: OutputStore,
    settings: Settings,
    progress: Progress,
) -> None:
    while True:
        source_index, variant_index = await queue.get()
        try:
            record = source[source_index]
            try:
                example = randomize_example(
                    record["question"],
                    record["answer"],
                    source_index,
                    settings.random_seed,
                    variant_index,
                )
            except SourceFormatError as exc:
                reason = f"malformed_source:{type(exc).__name__}"
                await store.append_discard(source_index, variant_index, reason)
                status = reason
            else:
                if example.question in store.output_questions:
                    await store.recover_accepted(source_index, variant_index)
                    status = "resumed_accepted"
                else:
                    status = await _process_example(client, store, settings, example)

            handled = progress.record(status)
            if handled % settings.log_every_items == 0 or handled == progress.total:
                LOGGER.info(
                    "Handled %d/%d queued items; accepted=%d discarded=%d",
                    handled,
                    progress.total,
                    progress.counts["accepted"],
                    handled
                    - progress.counts["accepted"]
                    - progress.counts["resumed_accepted"],
                )
        except Exception as exc:
            # Keep a large unattended synthesis job moving, while making the
            # terminal failure visible and resumable in the checkpoint.
            LOGGER.exception(
                "Worker %d failed on source item %d variant %d",
                worker_id,
                source_index,
                variant_index,
            )
            reason = f"pipeline_failure:{type(exc).__name__}"
            await store.append_discard(source_index, variant_index, reason)
            progress.record(reason)
        finally:
            queue.task_done()


async def run_pipeline(settings: Settings) -> None:
    """Run the complete synthesis job without loading model responses into memory."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings.validate()
    source = _load_source(settings.input_path)
    store = OutputStore(settings, sha256_file(settings.input_path))
    store.open()

    all_variants = [
        (source_index, variant_index)
        for source_index in range(len(source))
        for variant_index in range(settings.augmentations_per_item)
    ]
    pending_variants = [
        key for key in all_variants if key not in store.completed_variants
    ]
    LOGGER.info(
        "Loaded %d source items (%d variants); %d checkpointed and %d pending",
        len(source),
        len(all_variants),
        len(all_variants) - len(pending_variants),
        len(pending_variants),
    )
    if not pending_variants:
        store.close()
        LOGGER.info("Nothing to do; the checkpoint is complete")
        return

    queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue()
    for key in pending_variants:
        queue.put_nowait(key)
    progress = Progress(total=len(pending_variants))

    try:
        async with VLLMClient(settings) as client:
            workers = [
                asyncio.create_task(
                    _worker(
                        worker_id,
                        queue,
                        source,
                        client,
                        store,
                        settings,
                        progress,
                    ),
                    name=f"synthesis-worker-{worker_id}",
                )
                for worker_id in range(settings.item_workers)
            ]
            await queue.join()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
    finally:
        store.close()

    LOGGER.info("Synthesis complete: %s", dict(progress.counts))
