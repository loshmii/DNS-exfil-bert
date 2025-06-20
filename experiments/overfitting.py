from training_pipeline.arguments import parse_dataclasses
from training_pipeline.trainer import (
    MLMTrainer,
    MaskingCallback,
    FileLoggingCallback,
)
from training_pipeline.masker import MaskSampler
from training_pipeline.data_collator import DnsDataCollatorForMLM
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from datasets import load_dataset
import logging
from transformers import (
    BertConfig,
    AutoModelForMaskedLM,
)
from omegaconf import OmegaConf
from data_pipeline.dns_tokenizers.char_dns.v0_1.config import CharTokConfig
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import CharTokenizer

if __name__ == "__main__":

    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="overfitting_test",
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "tokenizer=bpe8k_pretrained",
                "training_arguments=overfitting",
                "training_arguments.optimizer_type=adamw",
            ],
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

    ch = logging.FileHandler(str(output_dir / "overfitting.log"), mode="w")
    ch.setFormatter(fmt)
    root.addHandler(ch)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    model_cfg = BertConfig(
        **OmegaConf.to_container(cfg.model_config, resolve=True),
        vocab_size=tokenizer.vocab_size
    )
    model = AutoModelForMaskedLM.from_config(model_cfg)

    data_files = {
        "train": [str(f) for f in cfg.dataset.files.train],
        "validation": [str(f) for f in cfg.dataset.files.validation],
        "test": [str(f) for f in cfg.dataset.files.test],
    }

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

    train_ds = ds["train"].select(range(1))
    eval_ds = train_ds.select([0])

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
