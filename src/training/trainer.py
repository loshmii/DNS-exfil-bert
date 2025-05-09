import transformers
import torch
from transformers import (
    AutoModelForMaskedLM,
    BertConfig,
    Trainer,
    DataCollatorForLanguageModeling,
)
import hydra
from pathlib import Path
from data_pipeline.dataset import load_dns_dataset
from hydra.core.hydra_config import HydraConfig
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)
from arguments import parse_dataclasses
from transformers import logging as hf_logging
import logging


class MLMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        print("Outputs are computed")
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
            overrides=["tokenizer=bpe_from_pretrained", "model=bert_for_mlm"],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)

    model_args, data_args, train_args = parse_dataclasses(cfg)

    output_dir = Path(train_args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    log_path = output_dir / "training.log"
    fh = logging.FileHandler(str(log_path), mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    hf_logging.enable_default_handler()
    hf_logging.set_verbosity_info()

    tokenizer = BpeTokenizer.from_pretrained(cfg.tokenizer.load_dir)
    model_cfg = BertConfig.from_pretrained(
        cfg.model.config_name, vocab_size=tokenizer.vocab_size
    )
    model = AutoModelForMaskedLM.from_config(model_cfg)
    ds = load_dns_dataset(data_args, tokenizer)
    train_ds = ds["train"].select(range(64))
    eval_ds = ds["val"].select(range(32))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=train_args.mlm_probability,
    )

    trainer = MLMTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
