import functools
import transformers
import torch
from transformers import (
    AutoModelForMaskedLM,
    BertConfig,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from training_pipeline.arguments import parse_dataclasses
import logging
from training_pipeline.masker import MaskSampler
from training_pipeline.data_collator import DnsDataCollatorForMLM
from omegaconf import OmegaConf
from training_pipeline.builders import MLMDatasetBuilder
from transformers.integrations import TensorBoardCallback
from omegaconf import DictConfig, OmegaConf
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import (
    CharTokenizer,
    CharTokConfig,
)
import torchmetrics
import time
from typing import Dict, Sequence, Any

BASE = Path(__file__).parent.parent.parent


def argmax_logits(logits, labels):
    return logits.argmax(dim=-1)


def _flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


class HPParamstersCallback(TrainerCallback):
    def __init__(
        self,
        cfg: DictConfig,
        metrics: Sequence[str],
    ):
        self.cfg = cfg
        self.metric_keys = metrics
        self.log_dir = str(Path(cfg.paths.tensorboard).expanduser().resolve())

    def on_train_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        super().on_evaluate(args, state, control, **kwargs)
        nested = OmegaConf.to_object(self.cfg)
        flat_cfg = _flatten_dict(nested)

        raw_overrides = HydraConfig.get().overrides.task
        override_keys = {ov.split("=", 1)[0] for ov in raw_overrides}

        hparams = {k: flat_cfg[k] for k in override_keys if k in flat_cfg}
        logging_metrics = dict()
        if metrics is not None:
            for key in self.metric_keys:
                if key in metrics:
                    logging_metrics[key] = metrics[key]

        if logging_metrics:
            self.writer.add_hparams(hparams, logging_metrics)
            self.writer.close()

            score_path = Path(self.cfg.paths.score).expanduser().resolve()
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.touch(exist_ok=True)
            with open(score_path, "w") as f:
                if "test_masked_accuracy" in logging_metrics:
                    f.write("%g" % logging_metrics["test_masked_accuracy"])
                elif "eval_masked_accuracy" in logging_metrics:
                    f.write("%g" % logging_metrics["eval_masked_accuracy"])
                elif "masked_accuracy" in metrics:
                    f.write("%g" % metrics["masked_accuracy"])


class PerplexityCallback(TrainerCallback):
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if logs is None:
            return control
        if "loss" in logs:
            logs["perplexity"] = torch.exp(torch.tensor(logs["loss"])).item()
        if "eval_loss" in logs:
            logs["eval_perplexity"] = torch.exp(
                torch.tensor(logs["eval_loss"])
            ).item()
        return control


class MaskingCallback(TrainerCallback):
    def __init__(self, collator):
        self.mask_sampler: MaskSampler = collator.mask_sampler

    def on_epoch_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.mask_sampler.set_epoch(state.epoch)
        return control


class FileLoggingCallback(TrainerCallback):
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        logs = logs or {}
        logger = logging.getLogger("transformer.trainer")
        logger.info({k: float(v) for k, v in logs.items()})
        return control


class MLMTrainer(Trainer):
    def __init__(self, *args, ignore_index=-100, **kwargs):
        kwargs["preprocess_logits_for_metrics"] = argmax_logits
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index
        self.acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.model.config.vocab_size,
            average="micro",
            ignore_index=self.ignore_index,
        ).to(self.model.device)

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None
    ):
        result = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        _, logits, labels = result[:3]

        if (
            logits is not None
            and labels is not None
            and not prediction_loss_only
        ):
            preds = logits
            mask = labels.ne(self.ignore_index)
            self.acc.update(preds[mask], labels[mask])

        return result

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        **gen_kwargs,
    ):
        eval_dataset = (
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        start_time = time.time()
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=self.args.prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **gen_kwargs,
        )

        total_batch_size = self.args.eval_batch_size * max(
            1, self.args.world_size
        )
        output.metrics[f"{metric_key_prefix}_runtime"] = (
            time.time() - start_time
        )
        output.metrics[f"{metric_key_prefix}_samples_per_second"] = (
            total_batch_size / output.metrics[f"{metric_key_prefix}_runtime"]
        )
        output.metrics[f"{metric_key_prefix}_steps_per_second"] = (
            total_batch_size
            / output.metrics[f"{metric_key_prefix}_samples_per_second"]
        )
        if not ignore_keys or "masked_accuracy" not in ignore_keys:
            output.metrics[f"{metric_key_prefix}_masked_accuracy"] = (
                self.acc.compute().item()
            )
            self.acc.reset()

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self.log(output.metrics)

        return output.metrics

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        mlm_loss = outputs.loss
        if return_outputs:
            return mlm_loss, outputs
        return mlm_loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if self.args.optimizer_type == "adamw":
            OptimCls = torch.optim.AdamW
            optim_kwargs = {
                "params": self.model.parameters(),
                "lr": self.args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": self.args.weight_decay,
            }
        elif self.args.optimizer_type == "adafactor":
            OptimCls = transformers.optimization.Adafactor
            optim_kwargs = {
                "params": self.model.parameters(),
                "lr": self.args.learning_rate,
                "eps": (1e-30, 1e-3),
                "clip_threshold": 1.0,
                "decay_rate": -0.8,
                "beta1": None,
                "weight_decay": self.args.weight_decay,
                "relative_step": False,
                "scale_parameter": True,
                "warmup_init": False,
            }
        else:
            raise ValueError(
                f"Unknown optimizer type: {self.args.optimizer_type}"
            )

        self.optimizer = OptimCls(**optim_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            warmup_steps = int(self.args.warmup_ratio * num_training_steps)
            if self.args.lr_scheduler_type == "cosine":
                self.lr_scheduler = (
                    transformers.optimization.get_cosine_schedule_with_warmup(
                        optimizer or self.optimizer,
                        warmup_steps,
                        num_training_steps,
                    )
                )
            elif self.args.lr_scheduler_type == "infinite":
                self.lr_scheduler = transformers.optimization.get_constant_schedule_with_warmup(
                    optimizer or self.optimizer
                )
            else:
                self.lr_scheduler = (
                    transformers.optimization.get_linear_schedule_with_warmup(
                        optimizer or self.optimizer,
                        warmup_steps,
                        num_training_steps,
                    )
                )
        return self.lr_scheduler


def add_parent_resolver(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        hc = HydraConfig.get()
        is_multirun = hc.mode == RunMode.MULTIRUN

        def parent(path: str, n: int = 0) -> str:
            path = Path(path)
            idx = n - 1 if not is_multirun else n
            return (
                str(path.parents[idx] if idx < len(path.parents) else path)
                if idx >= 0
                else str(path)
            )

        OmegaConf.register_new_resolver(
            "parent",
            parent,
            replace=True,
        )
        return fn(*args, **kwargs)

    return wrapper


def add_num_resolver(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def num(job_num: str) -> str:
            hc = HydraConfig.get()

            if hc.mode == RunMode.MULTIRUN:
                return "/" + str(hc.job.num)
            else:
                return ""

        OmegaConf.register_new_resolver(
            "num",
            num,
            replace=True,
        )
        return fn(*args, **kwargs)

    return wrapper


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
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
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    ch = logging.FileHandler(str(output_dir / "training.log"), mode="w")
    ch.setFormatter(fmt)
    root.addHandler(ch)

    tokenizer = CharTokenizer(
        CharTokConfig(
            BASE / "configs" / "tokenizer" / "char.yaml",
        )
    )

    model_cfg = BertConfig(
        **OmegaConf.to_container(cfg.model_config, resolve=True),
        vocab_size=tokenizer.vocab_size,
    )

    model = AutoModelForMaskedLM.from_config(model_cfg)

    builder = MLMDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.MLM_builder_args, resolve=True),
    )
    ds = builder.build()

    train_ds = ds["train"].select(range(1))
    eval_ds = train_ds
    test_ds = train_ds

    mask_sampler = MaskSampler(
        **OmegaConf.to_container(
            cfg.training_arguments.mask_args, resolve=True
        )
    )

    data_collator = DnsDataCollatorForMLM(
        tokenizer=tokenizer,
        mask_sampler=mask_sampler,
        **OmegaConf.to_container(
            cfg.training_arguments.MLM_collator_args, resolve=True
        ),
    )

    train_args.remove_unused_columns = False

    trainer = MLMTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            MaskingCallback(data_collator),
            FileLoggingCallback(),
            PerplexityCallback(),
            TensorBoardCallback(),
        ],
        compute_metrics=None,
    )

    trainer.train()

    loss = trainer.evaluate(
        eval_dataset=eval_ds,
        metric_key_prefix="test",
    )


if __name__ == "__main__":
    main()
