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
from training_pipeline.sampler import GroupWeightedRandomSampler
from collections import defaultdict

BASE = Path(__file__).parent.parent.parent.resolve()


def argmax_logits(logits, labels):
    return logits.argmax(dim=-1)


class ROCCurveCallback(TrainerCallback):
    def __init__(self, writer: SummaryWriter, trainer):
        self.writer = writer
        self.trainer = trainer

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        logits, labels = (
            self.trainer._last_eval_logits,
            self.trainer._last_eval_labels,
        )
        if logits is None or labels is None:
            logging.warning(
                "No logits or labels found for ROC curve. Skipping ROC curve plotting."
            )
            return control

        if logits.ndim == 2 and logits.shape[-1] > 1:
            probs = softmax(logits, axis=-1)[:, 1]
        else:
            probs = 1 / (1 + np.exp(-logits.reshape(-1)))

        fpr, tpr, _ = roc_curve(labels, probs)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="ROC curve")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        self.writer.add_figure(
            "ROC Curve",
            fig,
            global_step=self.trainer.state.global_step,
        )
        plt.close(fig)


class CLSTrainer(Trainer):
    def __init__(
        self,
        *args,
        num_labels: int = 2,
        problem_type: Literal[
            "single_label_classification",
            "multi_label_classification",
        ] = "single_label_classification",
        class_weights: Optional[torch.Tensor] = None,
        pos_weight: Optional[float] = None,
        metrics_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.num_labels = num_labels
        self.problem_type = problem_type
        self.threshold = metrics_threshold

        self.class_weights = (
            torch.tensor(class_weights, device=self.args.device)
            if class_weights is not None
            else None
        )
        self.pos_weight = (
            torch.tensor(pos_weight, device=self.args.device)
            if pos_weight is not None
            else None
        )

        self.acc = torchmetrics.Accuracy(
            task="binary" if num_labels == 2 else "multilabel",
            num_classes=num_labels,
        ).to(self.args.device)
        self.prec = torchmetrics.Precision(
            task="binary" if num_labels == 2 else "multilabel",
            num_classes=num_labels,
        ).to(self.args.device)
        self.rec = torchmetrics.Recall(
            task="binary" if num_labels == 2 else "multilabel",
            num_classes=num_labels,
        ).to(self.args.device)
        self.f1 = torchmetrics.F1Score(
            task="binary" if num_labels == 2 else "multilabel",
            num_classes=num_labels,
        ).to(self.args.device)
        self.auroc = torchmetrics.AUROC(
            task="binary" if num_labels == 2 else "multilabel",
            num_classes=num_labels,
            average=None if num_labels > 2 else "macro",
        ).to(self.args.device)

        self._last_eval_logits = None
        self._last_eval_labels = None

    def get_dup_weight_map(self) -> Dict[int, float]:
        if self._dup_weight_map is None:
            raise RuntimeError("Call .build() before accessing this method.")
        return self._dup_weight_map

    def _update_metrics(
        self,
        preds,
        probs,
        labels,
    ):
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds).long().to(self.args.device)
        else:
            preds = preds.to(self.args.device)
        if isinstance(probs, np.ndarray):
            probs = torch.from_numpy(probs.astype(np.float32)).to(
                self.args.device
            )
        else:
            probs = probs.to(self.args.device)
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long().to(self.args.device)
        else:
            labels = labels.to(self.args.device)

        self.acc.update(preds, labels)
        self.prec.update(preds, labels)
        self.rec.update(preds, labels)
        self.f1.update(preds, labels)
        self.auroc.update(probs, labels)

    def get_train_dataloader(self):
        if "dup_gid" in self.train_dataset.column_names:
            dup_gids = [int(x) for x in self.train_dataset["dup_gid"]]
            gid_to_idx = defaultdict(list)
            for idx, gid in enumerate(dup_gids):
                gid_to_idx[gid].append(idx)

            dup_weights_map = self.model.config._dup_weight_map
            gids_sorted = sorted(dup_weights_map)
            weights = [dup_weights_map[gid] for gid in gids_sorted]
            print(gid_to_idx)
            print(len(self.train_dataset))
            exit()

            sampler = GroupWeightedRandomSampler(
                dup_gids=gids_sorted,
                dup_gid_to_indices=gid_to_idx,
                group_weights=weights,
                num_samples=len(self.train_dataset),
            )
        else:
            sampler = None

        if self.args.remove_unused_columns:
            self.train_dataset = self._remove_unused_columns(
                self.train_dataset, description="Training"
            )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[Sequence[str]] = None,
    ):
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

        if (
            not prediction_loss_only
            and logits is not None
            and labels is not None
        ):
            if self.problem_type == "single_label_classification":
                probs = logits.softmax(dim=-1)[..., 1]
                preds = (probs > 0.5).long()
            else:
                probs = logits.sigmoid()
                preds = (probs > self.threshold).long()

            self._update_metrics(preds, probs, labels)

        return loss, logits, labels

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
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

        if not self.args.prediction_loss_only:
            raw_logits = output.predictions
            labels = output.label_ids

            self._last_eval_logits, self._last_eval_labels = raw_logits, labels

            if raw_logits.ndim == 2 and raw_logits.shape[-1] > 1:
                probs_pos = softmax(raw_logits, axis=-1)[:, 1]
                preds = (probs_pos > 0.5).astype(np.int64)
            else:
                probs_pos = expit(raw_logits.ravel())
                preds = (probs_pos > self.threshold).astype(np.int64)

            fpr, tpr, _ = roc_curve(labels, probs_pos)
            recall01 = (
                float(tpr[fpr <= 0.01].max()) if any(fpr <= 0.01) else 0.0
            )

            pr_auc = float(average_precision_score(labels, probs_pos))

            output.metrics[f"{metric_key_prefix}_recall@1%FPR"] = recall01
            output.metrics[f"{metric_key_prefix}_pr_auc"] = pr_auc

        end_time = time.time()

        total_batch_size = self.args.eval_batch_size * max(
            1, self.args.world_size
        )
        output.metrics[f"{metric_key_prefix}_runtime"] = end_time - start_time
        output.metrics[f"{metric_key_prefix}_samples_per_second"] = (
            total_batch_size / output.metrics[f"{metric_key_prefix}_runtime"]
        )
        output.metrics[f"{metric_key_prefix}_steps_per_second"] = (
            total_batch_size
            / output.metrics[f"{metric_key_prefix}_samples_per_second"]
        )

        if not ignore_keys or "accuracy" not in ignore_keys:
            output.metrics[f"{metric_key_prefix}_accuracy"] = (
                self.acc.compute().item()
            )
        if not ignore_keys or "precision" not in ignore_keys:
            output.metrics[f"{metric_key_prefix}_precision"] = (
                self.prec.compute().item()
            )
        if not ignore_keys or "recall" not in ignore_keys:
            output.metrics[f"{metric_key_prefix}_recall"] = (
                self.rec.compute().item()
            )
        if not ignore_keys or "f1" not in ignore_keys:
            output.metrics[f"{metric_key_prefix}_f1"] = (
                self.f1.compute().item()
            )
        if not ignore_keys or "auroc" not in ignore_keys:
            au = self.auroc.compute()
            if isinstance(au, torch.Tensor):
                au = au.mean()
            output.metrics[f"{metric_key_prefix}_auroc"] = au.item()

        for m in (self.acc, self.prec, self.rec, self.f1, self.auroc):
            m.reset()

        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics=output.metrics,
        )
        self.log(output.metrics)

        return output.metrics

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.problem_type == "single_label_classification":
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.problem_type == "multi_label_classification":
            labels = labels.float()
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

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
                f"Unsupported optimizer type: {self.args.optimizer_type}"
            )

        self.optimizer = OptimCls(**optim_kwargs)
        return self.optimizer

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.lr_scheduler is None:
            warmup_steps = (
                self.args.warmup_steps
                if self.args.warmup_steps
                else int(self.args.warmup_ratio * num_training_steps)
            )
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
                    optimizer or self.optimizer,
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
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    ch = logging.FileHandler(str(output_dir / "train.log"), mode="w")
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

    model = AutoModelForSequenceClassification.from_config(
        model_cfg,
    )

    builder = CLSDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.CLS_builder_args, resolve=True),
    )
    ds = builder.build()

    model.config._dup_weight_map = builder.get_dup_weight_map()
    weights = builder.get_class_weights()

    train_ds = ds["train"].select(range(5))
    eval_ds = ds["validation"].select(range(5))
    test_ds = ds["test"].select(range(2))

    data_collator = DnsDataCollatorForCLC(
        tokenizer=tokenizer,
        **OmegaConf.to_container(
            cfg.training_arguments.CLS_collator_args, resolve=True
        ),
    )

    writer = SummaryWriter(log_dir=str(train_args.logging_dir))
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


if __name__ == "__main__":
    main()
