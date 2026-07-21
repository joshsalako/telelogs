"""Small runtime helpers shared by training and evaluation commands."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any


def configure_logging(log_dir: Path, command: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rca_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"{command}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy

        state["numpy"] = numpy.random.get_state()
    except ImportError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    if "numpy" in state:
        import numpy

        numpy.random.set_state(state["numpy"])


def require_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Training dependencies are not installed. Run `uv sync --extra flash` in training/."
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("This command requires an NVIDIA GPU with CUDA available")
