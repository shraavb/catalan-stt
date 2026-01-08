"""Training configuration for Whisper fine-tuning."""

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for Whisper fine-tuning."""

    # Model
    base_model: str = "openai/whisper-small"
    language: str = "es"
    task: str = "transcribe"

    # Training hyperparameters
    output_dir: str = "./models/catalan-whisper"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = -1
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

    # Optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data
    max_duration_seconds: float = 30.0
    sample_rate: int = 16000

    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # Evaluation
    eval_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False

    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Flatten nested config
        flat_config = {}
        for section in ["model", "training", "data", "evaluation"]:
            if section in data:
                flat_config.update(data[section])

        return cls(**{k: v for k, v in flat_config.items() if hasattr(cls, k)})

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def get_training_args(self):
        """Get HuggingFace TrainingArguments-compatible dict."""
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "evaluation_strategy": self.eval_strategy,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "seed": self.seed,
            "dataloader_num_workers": self.dataloader_num_workers,
            "remove_unused_columns": self.remove_unused_columns,
            "push_to_hub": self.push_to_hub,
        }
