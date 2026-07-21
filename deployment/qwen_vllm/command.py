"""Build the vLLM server command without importing vLLM or CUDA."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .config import DeploymentSettings


def build_serve_command(settings: DeploymentSettings) -> list[str]:
    settings.validate()
    command = [
        "vllm",
        "serve",
        settings.model,
        "--served-model-name",
        settings.served_model_name,
        "--host",
        settings.host,
        "--port",
        str(settings.port),
        "--tensor-parallel-size",
        str(settings.tensor_parallel_size),
        "--max-model-len",
        str(settings.max_model_len),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--max-num-seqs",
        str(settings.max_num_seqs),
        "--reasoning-parser",
        settings.reasoning_parser,
        "--quantization",
        settings.quantization,
    ]
    if settings.language_model_only:
        command.append("--language-model-only")
    if settings.api_key:
        command.extend(["--api-key", settings.api_key])
    return command


def build_serve_environment(
    settings: DeploymentSettings,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the child environment, including Hugging Face `.env` settings."""

    child = dict(os.environ if environ is None else environ)
    optional_values = {
        "HF_TOKEN": settings.hf_token,
        "HF_HOME": settings.hf_home,
        "HUGGINGFACE_HUB_CACHE": settings.huggingface_hub_cache,
    }
    for name, value in optional_values.items():
        if value is not None:
            child[name] = value
    return child
