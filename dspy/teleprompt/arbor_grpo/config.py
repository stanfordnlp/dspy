from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ArborTrainConfig:
    # Core rollout control for GRPO
    num_generations: int | None = None
    temperature: float = 1.0
    beta: float = 0.04
    num_iterations: int = 1
    per_device_train_batch_size: int = 8
    learning_rate: float = 1e-6
    gradient_accumulation_steps: int = 1
    # This is false by default in TRL, but we prefer True here
    gradient_checkpointing: bool = True
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_steps: int = 10
    max_prompt_length: int | None = None
    max_completion_length: int | None = None
    gradient_checkpointing_kwargs: dict[str, Any] | None = None
    bf16: bool = False
    scale_rewards: bool = True
    max_grad_norm: float = 1.0
    report_to: str = "none"
    log_completions: bool = True
    logging_steps: int = 10
    # By default, None means model's max context length
    max_context_length: int | None = None
    lora: bool = False
    max_steps: int = 500
    # Controls how many prompts are generated per batch during async rollouts
    generation_batch_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        # Convert dataclass fields into a plain dict for downstream usage
        return asdict(self)


