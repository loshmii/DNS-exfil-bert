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
from datasets import load_dataset, concatenate_datasets
import logging
from transformers import (
    BertConfig,
    AutoModelForMaskedLM,
)
from omegaconf import OmegaConf

if __name__ == "__main__":
    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="conv_test_bert_bpe8k",
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "tokenizer=bpe8k_pretrained",
                "training_arguments=args_for_conv_test",
                "dataset=dataset_for_mlm",
                "model_config=bert_config_from_pretrained",
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
    ch = logging.FileHandler(str(output_dir / "training.log"), mode="w")
    ch.setFormatter(fmt)
    root.addHandler(ch)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    model_cfg = BertConfig(
        **OmegaConf.to_container(cfg.model_config, resolve=True),
        vocab_size=tokenizer.vocab_size,
    )
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

    ds = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    splits = ds.train_test_split(test_size=0.1, seed=42)
    train_eval_split = splits["train"]
    test_ds = splits["test"]
    train_eval_split = train_eval_split.train_test_split(
        test_size=0.11111, seed=42
    )
    train_ds = train_eval_split["train"]
    val_ds = train_eval_split["test"].select(
        range(int(0.05 * train_ds.shape[0]))
    )

    mask_sampler = MaskSampler(
        mlm_probability=train_args.mlm_probability,
        strategy="token",
    )

    data_collator = DnsDataCollatorForMLM(
        tokenizer=tokenizer,
        mask_sampler=mask_sampler,
        pad_to_multiple_of=8,
    )

    trainer = MLMTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            MaskingCallback(data_collator),
            FileLoggingCallback(),
        ],
    )

    trainer.train()
    trainer.save_model(output_dir)
