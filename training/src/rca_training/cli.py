"""Command-line entry point for the alignment pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import RunConfig
from .data import (
    DataValidationError,
    load_sft_records,
    load_training_records,
    load_validation_records,
)
from .utils import configure_logging


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="base model name or path")
    parser.add_argument("--artifact-root", type=Path, help="override artifacts directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", help="enable W&B under this project name")
    parser.add_argument(
        "--attention-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rca_training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="validate all three data contracts")
    validate.add_argument(
        "--skip-sft", action="store_true", help="validate source and held-out data only"
    )
    validate.add_argument("--train-json", type=Path)
    validate.add_argument("--sft-jsonl", type=Path)
    validate.add_argument("--validation-questions", type=Path)
    validate.add_argument("--validation-targets", type=Path)

    sft = subparsers.add_parser("sft", help="run completion-only supervised fine-tuning")
    _add_shared(sft)
    sft.add_argument("--epochs", type=int, default=10)
    sft.add_argument("--learning-rate", type=float, default=1e-6)
    sft.add_argument("--gradient-accumulation-steps", type=int, default=128)
    sft.add_argument("--max-sequence-length", type=int, default=8192)
    sft.add_argument("--save-steps", type=int, default=100)
    sft.add_argument("--logging-steps", type=int, default=1)
    sft.add_argument("--sft-jsonl", type=Path)
    sft.add_argument("--lora-rank", type=int, default=32)
    sft.add_argument("--lora-alpha", type=int, default=64)
    sft.add_argument("--lora-dropout", type=float, default=0.05)
    sft.add_argument("--resume-from-checkpoint")

    grpo = subparsers.add_parser("grpo", help="run group-relative policy optimization")
    _add_shared(grpo)
    grpo.add_argument("--epochs", type=int, default=10)
    grpo.add_argument("--learning-rate", type=float, default=1e-6)
    grpo.add_argument("--num-generations", type=int, default=8)
    grpo.add_argument("--gradient-accumulation-steps", type=int, default=8)
    grpo.add_argument("--prompt-max-length", type=int, default=2048)
    grpo.add_argument("--completion-max-length", type=int, default=2048)
    grpo.add_argument("--temperature", type=float, default=0.7)
    grpo.add_argument("--top-p", type=float, default=0.95)
    grpo.add_argument("--epsilon", type=float, default=0.2)
    grpo.add_argument("--beta", type=float, default=0.001)
    grpo.add_argument("--save-steps", type=int, default=100)
    grpo.add_argument("--logging-steps", type=int, default=1)
    grpo.add_argument("--train-json", type=Path)
    grpo.add_argument("--resume-from-checkpoint")

    evaluate = subparsers.add_parser("evaluate", help="evaluate RL_Model on held-out validation")
    _add_shared(evaluate)
    evaluate.add_argument("--num-samples", type=int, default=4, choices=(4,))
    evaluate.add_argument("--prompt-max-length", type=int, default=2048)
    evaluate.add_argument("--completion-max-length", type=int, default=2048)
    evaluate.add_argument("--temperature", type=float, default=0.7)
    evaluate.add_argument("--top-p", type=float, default=0.95)
    evaluate.add_argument("--validation-questions", type=Path)
    evaluate.add_argument("--validation-targets", type=Path)
    return parser


def _resolved_config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig()
    if getattr(args, "artifact_root", None):
        config.paths.artifact_root = args.artifact_root.resolve()
    config.model.model_name = args.model
    config.model.seed = args.seed
    config.model.attention_implementation = args.attention_implementation
    config.wandb_project = args.wandb_project
    if args.command == "sft":
        config.sft.epochs = args.epochs
        config.sft.learning_rate = args.learning_rate
        config.sft.gradient_accumulation_steps = args.gradient_accumulation_steps
        config.sft.max_sequence_length = args.max_sequence_length
        config.sft.save_steps = args.save_steps
        config.sft.logging_steps = args.logging_steps
        config.model.lora_rank = args.lora_rank
        config.model.lora_alpha = args.lora_alpha
        config.model.lora_dropout = args.lora_dropout
        if args.sft_jsonl:
            config.paths.sft_jsonl = args.sft_jsonl.resolve()
        config.sft.resume_from_checkpoint = args.resume_from_checkpoint
    elif args.command == "grpo":
        config.grpo.epochs = args.epochs
        config.grpo.learning_rate = args.learning_rate
        config.grpo.num_generations = args.num_generations
        config.grpo.gradient_accumulation_steps = args.gradient_accumulation_steps
        config.grpo.prompt_max_length = args.prompt_max_length
        config.grpo.completion_max_length = args.completion_max_length
        config.grpo.temperature = args.temperature
        config.grpo.top_p = args.top_p
        config.grpo.epsilon = args.epsilon
        config.grpo.beta = args.beta
        config.grpo.save_steps = args.save_steps
        config.grpo.logging_steps = args.logging_steps
        if args.train_json:
            config.paths.train_json = args.train_json.resolve()
        config.grpo.resume_from_checkpoint = args.resume_from_checkpoint
    elif args.command == "evaluate":
        config.evaluation.num_samples = args.num_samples
        config.evaluation.prompt_max_length = args.prompt_max_length
        config.evaluation.completion_max_length = args.completion_max_length
        config.evaluation.temperature = args.temperature
        config.evaluation.top_p = args.top_p
        if args.validation_questions:
            config.paths.validation_questions = args.validation_questions.resolve()
        if args.validation_targets:
            config.paths.validation_targets = args.validation_targets.resolve()
    return config


def _validate_data(args: argparse.Namespace) -> None:
    config = RunConfig()
    if args.train_json:
        config.paths.train_json = args.train_json.resolve()
    if args.sft_jsonl:
        config.paths.sft_jsonl = args.sft_jsonl.resolve()
    if args.validation_questions:
        config.paths.validation_questions = args.validation_questions.resolve()
    if args.validation_targets:
        config.paths.validation_targets = args.validation_targets.resolve()
    logger = configure_logging(config.paths.artifact_root / "logs", "validate-data")
    train = load_training_records(config.paths.train_json)
    validation = load_validation_records(
        config.paths.validation_questions, config.paths.validation_targets
    )
    if len(train) != 2400:
        raise DataValidationError(f"Expected 2,400 original training records; got {len(train)}")
    if len(validation) != 864:
        raise DataValidationError(f"Expected 864 validation questions; got {len(validation)}")
    summary = f"train={len(train)}, validation={len(validation)}"
    if not args.skip_sft:
        sft = load_sft_records(config.paths.sft_jsonl)
        summary += f", sft={len(sft)}"
    else:
        summary += ", sft=skipped"
    logger.info("Data validation passed: %s", summary)
    print(f"Data validation passed: {summary}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-data":
            _validate_data(args)
        else:
            config = _resolved_config(args)
            if args.command == "sft":
                from .sft import run_sft

                run_sft(config)
            elif args.command == "grpo":
                from .grpo import run_grpo

                run_grpo(config)
            else:
                from .evaluate import run_evaluation

                run_evaluation(config)
    except (DataValidationError, FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
