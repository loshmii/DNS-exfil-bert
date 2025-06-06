import functools
import transformers
import torch
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    BertForSequenceClassification,
)
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from pathlib import Path
from training_pipeline.arguments import parse_dataclasses
import logging
from training_pipeline.masker import MaskSampler
from training_pipeline.data_collator import DnsDataCollatorForCLC
from omegaconf import OmegaConf
from training_pipeline.builders import CLSDatasetBuilder
from transformers.integrations import TensorBoardCallback
from omegaconf import DictConfig, OmegaConf
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)
import torchmetrics
import time
from typing import Dict, Sequence, Any, Optional, Literal
from training_pipeline.trainer import (
    add_num_resolver,
    add_parent_resolver,
    FileLoggingCallback,
)
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, average_precision_score
from matplotlib import pyplot as plt
from scipy.special import softmax, expit
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import (
    CharTokenizer,
    CharTokConfig,
)
from training_pipeline.cls_trainer import (
    CLSTrainer,
    ROCCurveCallback,
)

BASE = Path(__file__).parent.parent.parent


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="cls_train_char",
    version_base="1.3",
)
@add_parent_resolver
@add_num_resolver
def main(cfg: DictConfig):

    model_args, train_args = parse_dataclasses(cfg)

    output_dir = Path(train_args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in root.handlers:
        root.removeHandler(h)

    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.FileHandler(str(output_dir / "train.log"), mode="w")
    ch.setFormatter(formatter)
    root.addHandler(ch)

    tokenizer = CharTokenizer(
        CharTokConfig(
            BASE / "configs" / "tokenizer" / "char.yaml",
        )
    )

    model = BertForSequenceClassification._from_config(
        BertConfig(
            vocab_size=tokenizer.vocab_size,
            **OmegaConf.to_container(cfg.model_config, resolve=True),
        )
    )

    builder = CLSDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.CLS_builder_args, resolve=True),
    )
    ds = builder.build()

    model.config._dup_weight_map = builder.get_dup_weight_map()
    weights = builder.get_class_weights()

    train_ds = ds["train"]
    eval_ds = ds["validation"]
    test_ds = ds["test"]

    data_collator = DnsDataCollatorForCLC(
        tokenizer=tokenizer,
        **OmegaConf.to_container(
            cfg.training_arguments.CLS_collator_args, resolve=True
        ),
    )

    writer = SummaryWriter(log_dir=str(train_args.logging_dir))

    print("Starting training with the following configuration:")
    trainer = CLSTrainer(
        model=model,
        args=train_args,
        class_weights=weights,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            FileLoggingCallback(),
        ],
        compute_metrics=None,
    )

    roc_cb = ROCCurveCallback(
        writer=writer,
        trainer=trainer,
    )
    trainer.add_callback(roc_cb)

    trainer.train()

    loss = trainer.evaluate(
        eval_dataset=test_ds,
        metric_key_prefix="test",
    )

    trainer.save_model(
        output_dir=str(output_dir / "model"),
    )
    trainer.save_state()


if __name__ == "__main__":
    main()
