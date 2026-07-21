"""Immutable, environment-driven deployment settings."""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DEPLOYMENT_ROOT = Path(__file__).resolve().parent.parent
ENV_PREFIX = "QWEN_VLLM_"


class SettingsError(ValueError):
    """Raised when deployment configuration is invalid."""


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(
        f"{name} must be a boolean (true/false, yes/no, on/off, or 1/0); got {value!r}"
    )


def parse_dotenv(path: Path) -> dict[str, str]:
    """Read a small, shell-compatible subset of dotenv syntax."""

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise SettingsError(f"{path}:{line_number}: expected NAME=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise SettingsError(f"{path}:{line_number}: invalid variable name {key!r}")
        try:
            parts = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise SettingsError(f"{path}:{line_number}: {exc}") from exc
        values[key] = " ".join(parts) if parts else ""
    return values


@dataclass(frozen=True, slots=True)
class DeploymentSettings:
    model: str = "Qwen/Qwen3.6-27B"
    served_model_name: str = "Qwen3.6-27B"
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    max_model_len: int = 16_384
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 8
    reasoning_parser: str = "qwen3"
    quantization: str = "bitsandbytes"
    api_key: str | None = None
    hf_token: str | None = None
    hf_home: str | None = None
    huggingface_hub_cache: str | None = None
    check_timeout_seconds: float = 30.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def validate(self) -> None:
        if not self.model.strip():
            raise SettingsError("QWEN_VLLM_MODEL must not be empty")
        if not self.served_model_name.strip():
            raise SettingsError("QWEN_VLLM_SERVED_MODEL_NAME must not be empty")
        if not self.host.strip():
            raise SettingsError("QWEN_VLLM_HOST must not be empty")
        if not 1 <= self.port <= 65_535:
            raise SettingsError("QWEN_VLLM_PORT must be between 1 and 65535")
        if self.tensor_parallel_size < 1:
            raise SettingsError("QWEN_VLLM_TENSOR_PARALLEL_SIZE must be at least 1")
        if self.max_model_len < 1:
            raise SettingsError("QWEN_VLLM_MAX_MODEL_LEN must be a positive integer")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise SettingsError(
                "QWEN_VLLM_GPU_MEMORY_UTILIZATION must be greater than 0 and at most 1"
            )
        if self.max_num_seqs < 1:
            raise SettingsError("QWEN_VLLM_MAX_NUM_SEQS must be a positive integer")
        if not self.reasoning_parser.strip():
            raise SettingsError("QWEN_VLLM_REASONING_PARSER must not be empty")
        if self.quantization != "bitsandbytes":
            raise SettingsError(
                "QWEN_VLLM_QUANTIZATION must be 'bitsandbytes' for this deployment"
            )
        if self.check_timeout_seconds <= 0:
            raise SettingsError(
                "QWEN_VLLM_CHECK_TIMEOUT_SECONDS must be greater than 0"
            )


def _convert(name: str, value: str, converter: type[int] | type[float]):
    try:
        return converter(value)
    except ValueError as exc:
        label = "an integer" if converter is int else "a number"
        raise SettingsError(f"{name} must be {label}; got {value!r}") from exc


def load_settings(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> DeploymentSettings:
    """Load `.env`, then overlay process environment variables."""

    source = os.environ if environ is None else environ
    merged = parse_dotenv(dotenv_path or DEPLOYMENT_ROOT / ".env")
    merged.update(source)

    def get(suffix: str, default: str) -> str:
        return merged.get(f"{ENV_PREFIX}{suffix}", default)

    def optional(name: str) -> str | None:
        return merged.get(name, "").strip() or None

    settings = DeploymentSettings(
        model=get("MODEL", "Qwen/Qwen3.6-27B"),
        served_model_name=get("SERVED_MODEL_NAME", "Qwen3.6-27B"),
        host=get("HOST", "127.0.0.1"),
        port=_convert("QWEN_VLLM_PORT", get("PORT", "8000"), int),
        tensor_parallel_size=_convert(
            "QWEN_VLLM_TENSOR_PARALLEL_SIZE", get("TENSOR_PARALLEL_SIZE", "1"), int
        ),
        max_model_len=_convert(
            "QWEN_VLLM_MAX_MODEL_LEN", get("MAX_MODEL_LEN", "16384"), int
        ),
        gpu_memory_utilization=_convert(
            "QWEN_VLLM_GPU_MEMORY_UTILIZATION",
            get("GPU_MEMORY_UTILIZATION", "0.90"),
            float,
        ),
        max_num_seqs=_convert("QWEN_VLLM_MAX_NUM_SEQS", get("MAX_NUM_SEQS", "8"), int),
        reasoning_parser=get("REASONING_PARSER", "qwen3"),
        quantization=get("QUANTIZATION", "bitsandbytes"),
        api_key=get("API_KEY", "").strip() or None,
        hf_token=optional("HF_TOKEN"),
        hf_home=optional("HF_HOME"),
        huggingface_hub_cache=optional("HUGGINGFACE_HUB_CACHE"),
        check_timeout_seconds=_convert(
            "QWEN_VLLM_CHECK_TIMEOUT_SECONDS",
            get("CHECK_TIMEOUT_SECONDS", "30"),
            float,
        ),
    )
    settings.validate()
    return settings
