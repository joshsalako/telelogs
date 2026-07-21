"""Typed, serializable configuration for both alignment stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINING_ROOT.parent


@dataclass(slots=True)
class Paths:
    train_json: Path = REPO_ROOT / "train.json"
    sft_jsonl: Path = REPO_ROOT / "synthesis" / "sft_train_data.jsonl"
    validation_questions: Path = REPO_ROOT / "validation_questions.csv"
    validation_targets: Path = REPO_ROOT / "validation_target.csv"
    artifact_root: Path = TRAINING_ROOT / "artifacts"

    @property
    def sft_output(self) -> Path:
        return self.artifact_root / "SFT_Model"

    @property
    def rl_output(self) -> Path:
        return self.artifact_root / "RL_Model"

    @property
    def evaluation_output(self) -> Path:
        return self.artifact_root / "evaluation"


@dataclass(slots=True)
class ModelConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    dtype: str = "bfloat16"
    attention_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True
    seed: int = 42
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def validate(self) -> None:
        if self.dtype != "bfloat16":
            raise ValueError("The single-GPU pipeline requires bfloat16")
        if self.lora_rank <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")


@dataclass(slots=True)
class SFTConfig:
    epochs: int = 10
    learning_rate: float = 1e-6
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 128
    max_sequence_length: int = 8192
    save_steps: int = 100
    logging_steps: int = 1
    resume_from_checkpoint: str | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("SFT epochs and learning rate must be positive")
        if self.micro_batch_size != 1:
            raise ValueError("SFT micro-batch size must remain 1 on the target GPU")
        if self.gradient_accumulation_steps <= 0 or self.max_sequence_length <= 0:
            raise ValueError("SFT accumulation steps and sequence length must be positive")
        if self.save_steps <= 0 or self.logging_steps <= 0:
            raise ValueError("SFT save and logging intervals must be positive")


@dataclass(slots=True)
class GRPOConfig:
    epochs: int = 10
    learning_rate: float = 1e-6
    micro_batch_size: int = 1
    num_generations: int = 8
    gradient_accumulation_steps: int = 8
    prompt_max_length: int = 2048
    completion_max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    epsilon: float = 0.2
    beta: float = 0.001
    scale_rewards: str = "group"
    save_steps: int = 100
    logging_steps: int = 1
    resume_from_checkpoint: str | None = None

    @property
    def effective_response_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("GRPO epochs and learning rate must be positive")
        if self.micro_batch_size != 1 or self.num_generations <= 1:
            raise ValueError("GRPO requires micro-batch 1 and at least two generations")
        if self.num_generations != self.gradient_accumulation_steps:
            raise ValueError(
                "GRPO requires gradient_accumulation_steps == num_generations so each "
                "optimizer update contains one complete response group"
            )
        if self.scale_rewards != "group":
            raise ValueError("Only group reward scaling is supported")
        if self.prompt_max_length <= 0 or self.completion_max_length <= 0:
            raise ValueError("GRPO prompt and completion limits must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("GRPO requires temperature > 0 and top-p in (0, 1]")
        if not 0 < self.epsilon < 1 or self.beta < 0:
            raise ValueError("GRPO epsilon must be in (0, 1) and beta cannot be negative")
        if self.save_steps <= 0 or self.logging_steps <= 0:
            raise ValueError("GRPO save and logging intervals must be positive")


@dataclass(slots=True)
class EvaluationConfig:
    num_samples: int = 4
    prompt_max_length: int = 2048
    completion_max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95

    def validate(self) -> None:
        if self.num_samples != 4:
            raise ValueError("Evaluation requires exactly four samples per question")
        if self.prompt_max_length <= 0 or self.completion_max_length <= 0:
            raise ValueError("Evaluation prompt and completion limits must be positive")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Evaluation requires temperature > 0 and top-p in (0, 1]")


@dataclass(slots=True)
class RunConfig:
    paths: Paths = field(default_factory=Paths)
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb_project: str | None = None

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))
