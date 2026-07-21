"""Command-line entry point for serving and checking Qwen3.6."""

from __future__ import annotations

import argparse
import os
import shlex
import sys

from .checks import CheckError, check_server
from .command import build_serve_command, build_serve_environment
from .config import SettingsError, load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m qwen_vllm")
    subparsers = parser.add_subparsers(dest="action", required=True)
    serve = subparsers.add_parser("serve", help="start the configured vLLM server")
    serve.add_argument(
        "--dry-run",
        action="store_true",
        help="print the command without importing vLLM or starting the server",
    )
    subparsers.add_parser("check", help="check health, model listing, and chat")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
        if args.action == "serve":
            command = build_serve_command(settings)
            if args.dry_run:
                print(shlex.join(command))
                return 0
            os.execvpe(command[0], command, build_serve_environment(settings))
            raise AssertionError("os.execvpe unexpectedly returned")
        for result in check_server(settings):
            print(result)
        return 0
    except (SettingsError, CheckError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
