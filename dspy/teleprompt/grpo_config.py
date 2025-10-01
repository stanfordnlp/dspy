from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dspy.clients.lm import LM
from collections import defaultdict

@dataclass
class GRPOConfig:
    """
    Configuration class for GRPO training.
    
    This class defines all the parameters needed for GRPO training in DSPy's Arbor implementation,
    matching exactly what the ArborReinforceJob.initialize() method expects
    """
    # Required GRPO parameter
    num_generations: int = field(
        metadata={"help": "Number of generations per prompt for GRPO training. Must be at least 2 for advantage calculation."}
    )
    # Core GRPO parameters
    temperature: float = field(
        default=0.9,
        metadata={"help": "Temperature for sampling during generation. Higher values make generation more random."}
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient for regularization. Controls the trade-off between reward optimization and staying close to the reference policy."}
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in GRPO algorithm)."}
    )
    # Training optimization parameters
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device during training."}
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate for the optimizer."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps before performing an optimizer step."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory at the expense of speed."}
    )
    lr_scheduler_type: str = field(
        default="constant_with_warmup",
        metadata={"help": "Type of learning rate scheduler to use. Options include 'constant_with_warmup', 'linear', 'cosine', etc."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Maximum gradient norm for gradient clipping."}
    )
    # Model and generation parameters
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of input prompts. If None, uses model's default."}
    )
    max_completion_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum length of generated completions. If None, uses model's default."}
    )
    max_context_length: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum context length for the model. If None, uses model's maximum."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bfloat16 mixed precision training."}
    )
    # GRPO-specific training parameters
    scale_rewards: bool = field(
        default=True,
        metadata={"help": "Whether to scale rewards during training for better stability."}
    )
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Additional keyword arguments for gradient checkpointing configuration."}
    )
    # LoRA parameters
    lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning."}
    )
    # Logging and monitoring parameters
    report_to: str = field(
        default="none",
        metadata={"help": "Where to report training metrics. Options: 'none', 'wandb', 'tensorboard', etc."}
    )
    log_completions: bool = field(
        default=True,
        metadata={"help": "Whether to log sample completions during training for monitoring."}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Number of training steps between logging metrics."}
    )
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if not hasattr(self, 'num_generations') or self.num_generations is None:
            raise ValueError("num_generations must be set in the GRPO configuration")
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive. Got {self.temperature}.")
        if self.beta < 0:
            raise ValueError(f"Beta (KL coefficient) must be non-negative. Got {self.beta}.")
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive. Got {self.learning_rate}.")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format for compatibility with existing train_kwargs.
        eg
        "temperature" : .79 
        .... 
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    @classmethod
    def convert_to_lm_dict(cls, arg) -> Dict[LM, "GRPOConfig"]:
        """Convert to LM dict, matching the existing DSPy pattern."""
        non_empty_dict = arg and isinstance(arg, dict)
        if non_empty_dict and all(isinstance(k, LM) for k in arg.keys()):
            if not all(isinstance(v, GRPOConfig) for v in arg.values()):
                raise ValueError("All values in the LM dict must be GRPOConfig instances.")
            return arg
        if not isinstance(arg, GRPOConfig):
            raise ValueError("arg must be a GRPOConfig instance or a dict with LM keys and GRPOConfig values.")
        return defaultdict(lambda: arg)
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GRPOConfig":
        """Create config from dictionary."""
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)