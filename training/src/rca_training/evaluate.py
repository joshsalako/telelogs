"""Four-sample evaluation on the validation-only split."""

from __future__ import annotations

import json

from .config import RunConfig
from .data import load_validation_records
from .grpo import generate_group, validate_prompt_lengths
from .metrics import compute_metrics, parsed_prediction, unique_plurality
from .modeling import load_adapter_policy, load_base_model, load_tokenizer
from .utils import configure_logging, require_cuda, set_deterministic_seed, write_json


def run_evaluation(config: RunConfig) -> None:
    config.model.validate()
    config.evaluation.validate()
    require_cuda()
    set_deterministic_seed(config.model.seed)
    logger = configure_logging(config.paths.artifact_root / "logs", "evaluate")

    import torch

    records = load_validation_records(
        config.paths.validation_questions, config.paths.validation_targets
    )
    if len(records) != 864:
        raise ValueError(f"Official validation set must contain 864 questions; got {len(records)}")
    tokenizer = load_tokenizer(str(config.paths.rl_output))
    validate_prompt_lengths(tokenizer, records, config.evaluation.prompt_max_length, "evaluation")
    logger.info("Preflight-tokenized %d validation prompts", len(records))
    model = load_adapter_policy(
        load_base_model(config.model, training=False), config.paths.rl_output, trainable=False
    )
    model.cuda().eval()

    output_dir = config.paths.evaluation_output
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "samples.jsonl"
    question_path = output_dir / "questions.jsonl"
    metric_groups: list[tuple[str, list[str]]] = []
    with (
        sample_path.open("w", encoding="utf-8") as sample_file,
        question_path.open("w", encoding="utf-8") as question_file,
    ):
        for index, record in enumerate(records):
            _, _, completions = generate_group(
                model,
                tokenizer,
                record.question,
                num_generations=config.evaluation.num_samples,
                prompt_max_length=config.evaluation.prompt_max_length,
                completion_max_length=config.evaluation.completion_max_length,
                temperature=config.evaluation.temperature,
                top_p=config.evaluation.top_p,
                device=torch.device("cuda"),
            )
            predictions = [parsed_prediction(completion) for completion in completions]
            plurality = unique_plurality(predictions)
            metric_groups.append((record.answer, completions))
            for sample_index, (completion, prediction) in enumerate(
                zip(completions, predictions, strict=True), 1
            ):
                sample_file.write(
                    json.dumps(
                        {
                            "id": record.identifier,
                            "sample": sample_index,
                            "target": record.answer,
                            "prediction": prediction,
                            "correct": prediction == record.answer,
                            "completion": completion,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            question_file.write(
                json.dumps(
                    {
                        "id": record.identifier,
                        "target": record.answer,
                        "predictions": predictions,
                        "plurality": plurality,
                        "majority_correct": plurality == record.answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            logger.info("evaluated=%d/%d id=%s", index + 1, len(records), record.identifier)

    metrics = compute_metrics(metric_groups, config.evaluation.num_samples)
    aggregate = {
        "questions": metrics.questions,
        "samples_per_question": config.evaluation.num_samples,
        "correct_samples": metrics.correct_samples,
        "total_samples": metrics.samples,
        "correct_majorities": metrics.correct_majorities,
        "pass@1": metrics.pass_at_1,
        "maj@4": metrics.maj_at_4,
    }
    write_json(output_dir / "metrics.json", aggregate)
    print(f"pass@1: {metrics.pass_at_1:.2%} ({metrics.correct_samples}/{metrics.samples})")
    print(f"maj@4:  {metrics.maj_at_4:.2%} ({metrics.correct_majorities}/{metrics.questions})")
