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
from pathlib import Path
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from training_pipeline.arguments import parse_dataclasses
import logging
from training_pipeline.masker import MaskSampler
from training_pipeline.data_collator import DnsDataCollatorForMLM
from omegaconf import OmegaConf


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


if __name__ == "__main__":
    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="trainer_test",
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tokenizer=bpe8k_pretrained", "model=bert_uncased", "model_config=bert_config_from_pretrained"],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)

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

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    model_cfg = BertConfig(**OmegaConf.to_container(cfg.model_config, resolve=True), 
        vocab_size=tokenizer.vocab_size)
    model = AutoModelForMaskedLM.from_config(model_cfg)

    data_files = OmegaConf.to_container(cfg.dataset.files, resolve=True)

    ds = load_dataset(
        path="src/training_pipeline/dataset_builder.py",
        name="default",
        data_files=data_files,
        streaming=False,
        trust_remote_code=True,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    train_ds = ds["train"]
    eval_ds = ds["validation"].select(range(5))

    mask_sampler = MaskSampler(
        mlm_probability=train_args.mlm_probability,
        strategy="token",
    )

    data_collator = DnsDataCollatorForMLM(
        tokenizer=tokenizer,
        mask_sampler=mask_sampler,
    )

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
        ],
    )

    trainer.train()
