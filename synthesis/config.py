"""Central configuration for the local vLLM synthesis job.

Edit the constants in this file before starting a run. The API key can also be
provided through ``VLLM_API_KEY`` so credentials never need to be committed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# Input/output paths.
INPUT_PATH = REPOSITORY_ROOT / "train.json"
OUTPUT_PATH = REPOSITORY_ROOT / "synthesis" / "sft_train_data.jsonl"
STATE_PATH = REPOSITORY_ROOT / "synthesis" / "synthesis_state.jsonl"

# Local OpenAI-compatible vLLM deployment.
CHAT_COMPLETIONS_URL = (
    "https://joshsalako--qwen-vllm-deployment-serve.modal.run/v1/chat/completions"
)
MODEL_NAME = "Qwen3.6-27B"
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

# Reproducibility and pipeline shape.
RANDOM_SEED = 20260721
AUGMENTATIONS_PER_ITEM = 3
AGENTS_PER_ITEM = 4  # Must be even so both reasoning strategies are balanced.
ITEM_WORKERS = 4
MAX_IN_FLIGHT_REQUESTS = 16
PIPELINE_VERSION = "2.0"

# Request/retry behavior.
REQUEST_TIMEOUT_SECONDS = 300.0
MAX_RETRY_ATTEMPTS = 4
RETRY_BACKOFF_MIN_SECONDS = 1.0
RETRY_BACKOFF_MAX_SECONDS = 30.0

# Generation settings. Reasoning uses diversity; formatting is conservative.
REASONING_TEMPERATURE = 0.7
REASONING_TOP_P = 0.95
REASONING_MAX_TOKENS = 4096
FORMATTING_TEMPERATURE = 0.2
FORMATTING_TOP_P = 0.9
FORMATTING_MAX_TOKENS = 2048

LOG_EVERY_ITEMS = 10


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings passed explicitly to pipeline components."""

    input_path: Path = INPUT_PATH
    output_path: Path = OUTPUT_PATH
    state_path: Path = STATE_PATH
    chat_completions_url: str = CHAT_COMPLETIONS_URL
    model_name: str = MODEL_NAME
    api_key: str = API_KEY
    random_seed: int = RANDOM_SEED
    augmentations_per_item: int = AUGMENTATIONS_PER_ITEM
    agents_per_item: int = AGENTS_PER_ITEM
    item_workers: int = ITEM_WORKERS
    max_in_flight_requests: int = MAX_IN_FLIGHT_REQUESTS
    pipeline_version: str = PIPELINE_VERSION
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS
    max_retry_attempts: int = MAX_RETRY_ATTEMPTS
    retry_backoff_min_seconds: float = RETRY_BACKOFF_MIN_SECONDS
    retry_backoff_max_seconds: float = RETRY_BACKOFF_MAX_SECONDS
    reasoning_temperature: float = REASONING_TEMPERATURE
    reasoning_top_p: float = REASONING_TOP_P
    reasoning_max_tokens: int = REASONING_MAX_TOKENS
    formatting_temperature: float = FORMATTING_TEMPERATURE
    formatting_top_p: float = FORMATTING_TOP_P
    formatting_max_tokens: int = FORMATTING_MAX_TOKENS
    log_every_items: int = LOG_EVERY_ITEMS

    def validate(self) -> None:
        if self.agents_per_item < 2 or self.agents_per_item % 2:
            raise ValueError("AGENTS_PER_ITEM must be an even integer of at least 2")
        if self.augmentations_per_item < 1:
            raise ValueError("AUGMENTATIONS_PER_ITEM must be a positive integer")
        if self.item_workers < 1 or self.max_in_flight_requests < 1:
            raise ValueError("Worker and request concurrency must be positive")
        if self.max_retry_attempts < 1:
            raise ValueError("MAX_RETRY_ATTEMPTS must be positive")


SETTINGS = Settings()
