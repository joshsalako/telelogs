# CoT SFT Data Synthesis

This package converts `train.json` into a randomized, validated Chain-of-Thought
SFT dataset using a local Qwen3.6-27B model served by an OpenAI-compatible vLLM
endpoint. It is asynchronous, bounded by configurable concurrency limits, and
safe to resume after interruption.

## Pipeline

For every source item, the pipeline creates three independent randomized
variants. For each variant, it:

1. Deterministically shuffles the data rows in both tables and maps `C1`-`C8`
   to a random permutation of `R1`-`R8`. Headers, descriptions, and every table
   value remain unchanged.
2. Runs two independent agents concurrently. One uses systematic elimination;
   one tests each candidate by contradiction.
3. parses each final `\boxed{R#}` prediction and continues only when at least
   a strict majority agrees with the randomized ground truth.
4. Selects the winning trajectory with the widest candidate coverage and sends
   it through a formatting pass that must produce exactly `Task 1`, `Task 2`,
   `Task 3`, and `Summary` sections. `Task 2` contains one eight-row evidence
   table, and `Summary` ends with the validated boxed answer.
5. Appends the randomized question and formatted response to JSONL.

Malformed responses, timeouts, rate limits, and transient server failures use
bounded exponential retries. An item is discarded after retries are exhausted,
when the vote is incorrect or inconclusive, or when final formatting fails.

## Configuration

Edit [`config.py`](config.py) before running. Important settings include:

- `CHAT_COMPLETIONS_URL`: defaults to
  `http://localhost:8000/v1/chat/completions`.
- `MODEL_NAME`: defaults to `Qwen3.6-27B` and must match the name exposed by
  the local vLLM deployment.
- `AGENTS_PER_ITEM`: defaults to two and must be an even number.
- `AUGMENTATIONS_PER_ITEM`: defaults to three. With 2,400 source items this
  schedules 7,200 independently randomized and validated generation attempts.
- `ITEM_WORKERS` and `MAX_IN_FLIGHT_REQUESTS`: bound item-level and HTTP
  concurrency respectively.
- `RANDOM_SEED`: makes each source item's identifier and row randomization
  reproducible across restarts.
- Token limits, sampling values, timeouts, retry attempts, and input/output
  paths are also centralized in the same file.

For a protected endpoint, export `VLLM_API_KEY`; otherwise the local default is
`EMPTY`.

## Install and run

From the repository root, use the existing uv-managed `.venv`:

```bash
uv sync --active
uv run --active python -m synthesis
```

The vLLM model server must already be running (e.g. deployed via Modal using `modal deploy modal_qwen_deploy.py`). The synthesis script will query the URL defined in your configuration.

## Outputs and resume behavior

Successful examples are written to `synthesis/sft_train_data.jsonl`, one object
per line:

```json
{"question": "<randomized prompt>", "response": "<four-section CoT>"}
```

Terminal variant statuses are appended to `synthesis/synthesis_state.jsonl`.
On a restart, accepted and discarded `(source_index, variant_index)` pairs are
skipped independently. Output is flushed before its accepted checkpoint event;
if interruption occurs between those two writes, the deterministic randomized
question is recognized and recovered on the next run without adding a
duplicate.

The pipeline refuses to resume if the checkpoint's input hash, random seed,
pipeline version, model name, agent count, augmentation count, or output schema
differs from the current configuration. Version 1 checkpoints from the
single-variant implementation are intentionally incompatible. Start a genuinely
new run with new output and state paths rather than mixing generations.

## Offline verification

The test suite never contacts vLLM:

```bash
uv run --active python -m unittest discover -s synthesis/tests -v
```

It covers deterministic multi-variant row-preserving randomization, identifier
remapping, boxed-answer parsing, strict majority voting, trajectory selection,
formatter validation, malformed-output retries, JSONL writing, and per-variant
checkpoint resume.
