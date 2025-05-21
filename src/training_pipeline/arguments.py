from dataclasses import dataclass, field
from transformers import TrainingArguments
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from typing import Optional
from omegaconf import DictConfig


@dataclass
class MLMTrainingArguments(TrainingArguments):
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "The probability of masking tokens in the input."},
    )
    span_masking: bool = field(
        default=False,
        metadata={
            "help": "If True, mask contiguous spans of tokens instead of individual tokens."
        },
    )
    span_lambda: float = field(
        default=3.0,
        metadata={
            "help": "Average span length for span masking when span_masking=True."
        },
    )

    optimizer_type: str = field(
        default="adamw",
        metadata={"help": "The type of optimizer to use. adamw | adafactor"},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={
            "help": "The type of learning rate scheduler to use. linear | cosine | infinite"
        },
    )
    warmup_ratio: float = field(
        default=0.01,
        metadata={"help": "The ratio of total steps to use for warmup."},
    )

    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory."},
    )
    torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile to speed up kernel execution."},
    )

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if not (0.0 < self.mlm_probability <= 1.0):
            raise ValueError(
                f"mlm_probability must be between 0.0 and 1.0, got {self.mlm_probability}."
            )
        if self.span_masking and self.span_lambda <= 0:
            raise ValueError(
                f"span_lambda must be greater than 0 when span_masking is True, got {self.span_lambda}."
            )
        allowed = {"adamw", "adafactor"}
        if self.optimizer_type not in allowed:
            raise ValueError(
                f"optimizer_type must be one of {allowed}, got {self.optimizer_type}."
            )
        if self.torch_compile and not hasattr(torch, "compile"):
            raise ValueError(
                "torch.compile is not available in this version of PyTorch."
            )


@dataclass
class ModelArguments:
    config_name: str = field(
        default="google-bert/bert-base-uncased",
        metadata={"help": "Path to the model config file."},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Path to the tokenizer config file."}
    )
    pretrainer_model_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pre-trained model."}
    )

    def __post_init__(self):
        if not self.config_name and not self.pretrainer_model_path:
            raise ValueError(
                "Specify either `config_name` or `pretrainer_model_path`."
            )


def parse_dataclasses(cfg):
    return (
        ModelArguments(**OmegaConf.to_container(cfg.model, resolve=True)),
        MLMTrainingArguments(
            **OmegaConf.to_container(
                cfg.training_arguments.trainer_args, resolve=True
            )
        ),
    )


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):

    model_args, train_args = parse_dataclasses(cfg)
    print(model_args)
    print(train_args)


if __name__ == "__main__":
    main()
