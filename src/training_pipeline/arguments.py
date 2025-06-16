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

    use_duplicate_weights: bool = field(
        default=True,
        metadata={
            "help": "If True, use duplicate group weights during training."
        }
    )

    use_class_weights: bool = field(
        default=False,
        metadata={
            "help": "If True, use class weights during training."
        }
    )

    train_fraction: float = field(
        default=1.0,
        metadata={
            "help": "Fraction of the training dataset to use. 1.0 means use the entire dataset."
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
        if (0.0 < self.train_fraction <= 1.0) is False:
            raise ValueError(
                f"train_fraction must be between 0.0 and 1.0, got {self.train_fraction}."
            )


@dataclass
class ModelArguments:
    config_name: str = field(
        default="google-bert/bert-base-uncased",
        metadata={"help": "Path to the model config file."},
    )
    num_labels: Optional[int] = field(
        default=None,
        metadata={"help": "Number of labels for the classification task."},
    )
    local_files_only: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only use local files. If True, will not download from the internet."
        },
    )
    pretrained_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the pre-trained model."}
    )

    def __post_init__(self):
        if not self.config_name and not self.pretrained_model_name_or_path:
            raise ValueError(
                "Specify either `config_name` or `pretrained_model_name_or_path`."
            )
        if self.num_labels is not None and self.num_labels <= 0:
            raise ValueError(
                f"num_labels must be greater than 0, got {self.num_labels}."
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
