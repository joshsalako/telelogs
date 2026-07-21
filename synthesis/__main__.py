"""Module entry point for ``uv run --active python -m synthesis``."""

import asyncio

from .config import SETTINGS
from .pipeline import run_pipeline


def main() -> None:
    asyncio.run(run_pipeline(SETTINGS))


if __name__ == "__main__":
    main()
