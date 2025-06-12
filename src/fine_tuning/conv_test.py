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
from training_pipeline.utils import stratified_subsets, EvalSubsetCallback
from sklearn.utils.class_weight import compute_class_weight

BASE = Path(__file__).parent.parent.parent


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="cls_conv_test",
    version_base="1.3",
)
@add_parent_resolver
@add_num_resolver
def main(cfg: DictConfig):

    model_args, train_args = parse_dataclasses(cfg)

    output_dir = Path(train_args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.FileHandler(
        str(output_dir / "training.log"), mode="w"
    )
    ch.setLevel(logging.INFO)
    root.addHandler(ch)

    tokenizer = CharTokenizer(
        CharTokConfig(
            BASE / "configs" / "tokenizer" / "char.yaml",
        )
    )

    model_cfg =BertConfig(
        **OmegaConf.to_container(cfg.model_config, resolve=True),
        vocab_size=tokenizer.vocab_size,
    )

    model = BertForSequenceClassification._from_config(
        model_cfg
    )

    builder = CLSDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.CLS_builder_args, resolve=True),
    )
    ds = builder.build()

    if train_args.use_duplicate_weights:
        model.config._dup_weight_map = builder.get_dup_weight_map()
    
    weights = builder.get_class_weights() if train_args.use_class_weights else None

    pos_idx = np.where(ds['train']['label'] == 1)[0][:256]
    neg_idx = np.where(ds['train']['label'] == 0)[0][:256]
    toy_idx = np.concatenate([pos_idx, neg_idx])
    rng = np.random.default_rng(42)
    rng.shuffle(toy_idx)

    toy_train = ds['train'].select(toy_idx)
    toy_eval = toy_train

    train_args.learning_rate = 1e-4
    train_args.per_device_train_batch_size = 32
    train_args.label_smoothing_factor = 0.0
    train_args.lr_scheduler_type = "linear"
    train_args.warmup_steps = 10
    train_args.max_steps = 200
    train_args.logging_steps = 10
    train_args.eval_steps = 50

    data_collator = DnsDataCollatorForCLC(
        tokenizer=tokenizer,
        **OmegaConf.to_container(
            cfg.training_arguments.CLS_collator_args, resolve=True
        ),
    )

    print(train_args)
    print(OmegaConf.to_container(cfg, resolve=True))

    trainer = CLSTrainer(
        model=model,
        args=train_args,
        train_dataset=toy_train,
        eval_dataset=toy_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[
            FileLoggingCallback()
        ],
        compute_metrics=None,
    )

    roc_cb = ROCCurveCallback(
        writer=SummaryWriter(
            str(train_args.logging_dir)
        ),
        trainer=trainer,
    )
    trainer.add_callback(roc_cb)

    trainer.train()

if __name__ == "__main__":
    main()