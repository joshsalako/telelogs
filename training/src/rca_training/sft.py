"""Stage-one completion-only supervised fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RunConfig
from .data import SFTRecord, format_sft_record, load_sft_records
from .modeling import (
    create_sft_policy,
    load_adapter_policy,
    load_base_model,
    load_tokenizer,
    save_adapter,
)
from .utils import (
    capture_rng_state,
    configure_logging,
    require_cuda,
    restore_rng_state,
    set_deterministic_seed,
    write_json,
)


class TokenizedSFTDataset:
    def __init__(self, records: list[SFTRecord], tokenizer: Any, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return format_sft_record(self.tokenizer, self.records[index], self.max_length)


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            result["input_ids"].append([self.pad_token_id] * padding + feature["input_ids"])
            result["attention_mask"].append([0] * padding + feature["attention_mask"])
            result["labels"].append([-100] * padding + feature["labels"])
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


class EpochRandomSampler:
    """A restart-stable random order derived only from seed and epoch."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        import torch

        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def _checkpoint_state(path: Path) -> tuple[int, int, int]:
    state_path = path / "training_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Checkpoint has no training_state.json: {path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return int(state["epoch"]), int(state["batch_in_epoch"]), int(state["optimizer_step"])


def run_sft(config: RunConfig) -> None:
    config.model.validate()
    config.sft.validate()
    require_cuda()
    set_deterministic_seed(config.model.seed)
    logger = configure_logging(config.paths.artifact_root / "logs", "sft")

    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    accelerator = Accelerator(
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb" if config.wandb_project else None,
    )
    if config.wandb_project:
        accelerator.init_trackers(config.wandb_project, config=config.to_dict())

    tokenizer = load_tokenizer(config.model.model_name)
    records = load_sft_records(config.paths.sft_jsonl)
    for record in records:
        format_sft_record(tokenizer, record, config.sft.max_sequence_length)
    logger.info("Preflight-tokenized %d SFT records", len(records))
    dataset = TokenizedSFTDataset(records, tokenizer, config.sft.max_sequence_length)
    sampler = EpochRandomSampler(len(dataset), config.model.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.sft.micro_batch_size,
        sampler=sampler,
        collate_fn=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    base = load_base_model(config.model, training=True)
    model = (
        load_adapter_policy(base, Path(config.sft.resume_from_checkpoint), trainable=True)
        if config.sft.resume_from_checkpoint
        else create_sft_policy(base, config.model)
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.sft.learning_rate,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    start_epoch = start_batch = optimizer_step = 0
    if config.sft.resume_from_checkpoint:
        checkpoint = Path(config.sft.resume_from_checkpoint)
        start_epoch, start_batch, optimizer_step = _checkpoint_state(checkpoint)
        saved_state = torch.load(
            checkpoint / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(saved_state["optimizer"])
        restore_rng_state(saved_state["rng"])
        logger.info(
            "Resumed checkpoint %s at epoch %d batch %d", checkpoint, start_epoch, start_batch
        )

    model.train()
    for epoch in range(start_epoch, config.sft.epochs):
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                optimizer_step += 1
                if optimizer_step % config.sft.logging_steps == 0:
                    value = accelerator.gather(loss.detach()).mean().item()
                    logger.info("epoch=%d step=%d loss=%.6f", epoch + 1, optimizer_step, value)
                    accelerator.log(
                        {"sft/loss": value, "sft/epoch": epoch + 1}, step=optimizer_step
                    )
                if optimizer_step % config.sft.save_steps == 0 and accelerator.is_main_process:
                    checkpoint = config.paths.sft_output / f"checkpoint-{optimizer_step}"
                    save_adapter(accelerator.unwrap_model(model), checkpoint)
                    torch.save(
                        {"optimizer": optimizer.state_dict(), "rng": capture_rng_state()},
                        checkpoint / "optimizer.pt",
                    )
                    write_json(
                        checkpoint / "training_state.json",
                        {
                            "epoch": epoch,
                            "batch_in_epoch": batch_index + 1,
                            "optimizer_step": optimizer_step,
                        },
                    )
        start_batch = 0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_adapter(unwrapped, config.paths.sft_output)
        tokenizer.save_pretrained(config.paths.sft_output)
        write_json(config.paths.sft_output / "resolved_config.json", config.to_dict())
        write_json(
            config.paths.sft_output / "training_state.json",
            {"epoch": config.sft.epochs, "batch_in_epoch": 0, "optimizer_step": optimizer_step},
        )
        logger.info("Saved cumulative SFT adapter to %s", config.paths.sft_output)
    accelerator.end_training()
