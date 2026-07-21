"""Stage-two native-Transformers GRPO over the original training distribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RunConfig
from .data import TrainingRecord, load_training_records
from .modeling import (
    add_frozen_reference_adapter,
    load_adapter_policy,
    load_base_model,
    load_tokenizer,
    save_adapter,
    set_active_adapter,
)
from .prompts import render_prompt
from .rewards import group_advantages, group_rewards
from .utils import (
    capture_rng_state,
    configure_logging,
    require_cuda,
    restore_rng_state,
    set_deterministic_seed,
    write_json,
)


def trim_generated_tokens(token_ids: list[int], eos_token_id: int, pad_token_id: int) -> list[int]:
    """Keep the first EOS but remove batch padding generated after it."""
    if eos_token_id in token_ids:
        return token_ids[: token_ids.index(eos_token_id) + 1]
    if pad_token_id != eos_token_id:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()
    return token_ids


def generate_group(
    model: Any,
    tokenizer: Any,
    question: str,
    *,
    num_generations: int,
    prompt_max_length: int,
    completion_max_length: int,
    temperature: float,
    top_p: float,
    device: Any,
) -> tuple[list[int], list[list[int]], list[str]]:
    import torch

    rendered = render_prompt(tokenizer, question)
    encoded = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    prompt_ids = encoded["input_ids"][0].tolist()
    if len(prompt_ids) > prompt_max_length:
        raise ValueError(
            f"GRPO prompt is {len(prompt_ids)} tokens; maximum is {prompt_max_length}. "
            "Increase --prompt-max-length explicitly or correct the record."
        )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    prior_cache = model.config.use_cache
    model.config.use_cache = True
    model.eval()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=num_generations,
            max_new_tokens=completion_max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    model.config.use_cache = prior_cache
    completion_ids = [
        trim_generated_tokens(
            row[len(prompt_ids) :].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id
        )
        for row in output
    ]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return prompt_ids, completion_ids, completions


def validate_prompt_lengths(
    tokenizer: Any, records: list[Any], prompt_max_length: int, stage: str
) -> None:
    for index, record in enumerate(records):
        rendered = render_prompt(tokenizer, record.question)
        token_count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if token_count > prompt_max_length:
            identifier = getattr(record, "identifier", index)
            raise ValueError(
                f"{stage} record {identifier!r} is {token_count} prompt tokens; "
                f"maximum is {prompt_max_length}"
            )


def _completion_log_probs(
    model: Any, prompt_ids: list[int], completion_ids: list[int], device: Any
) -> Any:
    import torch

    if not completion_ids:
        return torch.empty(0, device=device)
    sequence = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(sequence)
    logits = model(input_ids=sequence, attention_mask=attention).logits
    token_logits = logits[:, len(prompt_ids) - 1 : -1, :]
    targets = sequence[:, len(prompt_ids) :]
    return (
        torch.log_softmax(token_logits.float(), dim=-1)
        .gather(dim=-1, index=targets.unsqueeze(-1))
        .squeeze(0)
        .squeeze(-1)
    )


def _load_resume_state(path: Path) -> tuple[int, int, int]:
    state_file = path / "training_state.json"
    if not state_file.is_file():
        raise FileNotFoundError(f"Checkpoint has no training_state.json: {path}")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    return int(state["epoch"]), int(state["record_in_epoch"]), int(state["optimizer_step"])


def run_grpo(config: RunConfig) -> None:
    config.model.validate()
    config.grpo.validate()
    require_cuda()
    set_deterministic_seed(config.model.seed)
    logger = configure_logging(config.paths.artifact_root / "logs", "grpo")

    import torch
    from accelerate import Accelerator

    accelerator = Accelerator(
        mixed_precision="bf16", log_with="wandb" if config.wandb_project else None
    )
    if config.wandb_project:
        accelerator.init_trackers(config.wandb_project, config=config.to_dict())

    tokenizer = load_tokenizer(str(config.paths.sft_output))
    records = load_training_records(config.paths.train_json)
    validate_prompt_lengths(tokenizer, records, config.grpo.prompt_max_length, "GRPO")
    logger.info("Preflight-tokenized %d original GRPO prompts", len(records))
    base = load_base_model(config.model, training=True)
    policy_path = (
        Path(config.grpo.resume_from_checkpoint)
        if config.grpo.resume_from_checkpoint
        else config.paths.sft_output
    )
    model = load_adapter_policy(base, policy_path, trainable=True)
    add_frozen_reference_adapter(model, config.paths.sft_output)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.grpo.learning_rate,
    )
    model, optimizer = accelerator.prepare(model, optimizer)

    start_epoch = start_record = optimizer_step = 0
    if config.grpo.resume_from_checkpoint:
        checkpoint = Path(config.grpo.resume_from_checkpoint)
        start_epoch, start_record, optimizer_step = _load_resume_state(checkpoint)
        saved_state = torch.load(
            checkpoint / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(saved_state["optimizer"])
        restore_rng_state(saved_state["rng"])
        logger.info(
            "Resumed checkpoint %s at epoch %d record %d", checkpoint, start_epoch, start_record
        )

    for epoch in range(start_epoch, config.grpo.epochs):
        order_generator = torch.Generator().manual_seed(config.model.seed + epoch)
        order = torch.randperm(len(records), generator=order_generator).tolist()
        for position, record_index in enumerate(order):
            if epoch == start_epoch and position < start_record:
                continue
            record: TrainingRecord = records[record_index]
            unwrapped = accelerator.unwrap_model(model)
            set_active_adapter(unwrapped, "default", trainable=True)
            prompt_ids, completion_ids, completions = generate_group(
                model,
                tokenizer,
                record.question,
                num_generations=config.grpo.num_generations,
                prompt_max_length=config.grpo.prompt_max_length,
                completion_max_length=config.grpo.completion_max_length,
                temperature=config.grpo.temperature,
                top_p=config.grpo.top_p,
                device=accelerator.device,
            )
            rewards = group_rewards(completions, record.answer, config.grpo.num_generations)
            advantages = group_advantages(rewards)

            old_log_probs: list[Any] = []
            reference_log_probs: list[Any] = []
            with torch.no_grad():
                set_active_adapter(unwrapped, "default", trainable=False)
                for ids in completion_ids:
                    old_log_probs.append(
                        _completion_log_probs(model, prompt_ids, ids, accelerator.device).detach()
                    )
                set_active_adapter(unwrapped, "reference", trainable=False)
                for ids in completion_ids:
                    reference_log_probs.append(
                        _completion_log_probs(model, prompt_ids, ids, accelerator.device).detach()
                    )
            set_active_adapter(unwrapped, "default", trainable=True)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for ids, old_logp, reference_logp, advantage in zip(
                completion_ids, old_log_probs, reference_log_probs, advantages, strict=True
            ):
                current_logp = _completion_log_probs(model, prompt_ids, ids, accelerator.device)
                if current_logp.numel() == 0:
                    continue
                ratio = torch.exp(current_logp - old_logp)
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - config.grpo.epsilon, 1.0 + config.grpo.epsilon
                )
                advantage_tensor = torch.tensor(advantage, device=accelerator.device)
                policy_objective = torch.minimum(
                    ratio * advantage_tensor, clipped_ratio * advantage_tensor
                )
                log_ratio = reference_logp - current_logp
                kl = torch.exp(log_ratio) - log_ratio - 1.0
                loss = (-policy_objective + config.grpo.beta * kl).mean()
                accelerator.backward(loss / config.grpo.gradient_accumulation_steps)
                group_loss += float(loss.detach()) / config.grpo.num_generations
            optimizer.step()
            optimizer_step += 1

            if optimizer_step % config.grpo.logging_steps == 0:
                accuracy = sum(rewards) / len(rewards)
                logger.info(
                    "epoch=%d step=%d reward=%.3f loss=%.6f",
                    epoch + 1,
                    optimizer_step,
                    accuracy,
                    group_loss,
                )
                accelerator.log(
                    {"grpo/reward": accuracy, "grpo/loss": group_loss, "grpo/epoch": epoch + 1},
                    step=optimizer_step,
                )
            if optimizer_step % config.grpo.save_steps == 0 and accelerator.is_main_process:
                checkpoint = config.paths.rl_output / f"checkpoint-{optimizer_step}"
                save_adapter(unwrapped, checkpoint)
                torch.save(
                    {"optimizer": optimizer.state_dict(), "rng": capture_rng_state()},
                    checkpoint / "optimizer.pt",
                )
                write_json(
                    checkpoint / "training_state.json",
                    {
                        "epoch": epoch,
                        "record_in_epoch": position + 1,
                        "optimizer_step": optimizer_step,
                    },
                )
        start_record = 0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        set_active_adapter(unwrapped, "default", trainable=False)
        save_adapter(unwrapped, config.paths.rl_output)
        tokenizer.save_pretrained(config.paths.rl_output)
        write_json(config.paths.rl_output / "resolved_config.json", config.to_dict())
        write_json(
            config.paths.rl_output / "training_state.json",
            {"epoch": config.grpo.epochs, "record_in_epoch": 0, "optimizer_step": optimizer_step},
        )
        logger.info("Saved cumulative SFT+GRPO adapter to %s", config.paths.rl_output)
    accelerator.end_training()
