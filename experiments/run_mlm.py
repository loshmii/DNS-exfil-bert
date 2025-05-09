from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import torch
from data_pipeline.dataset import get_dataset_from_processed
from hydra.core.hydra_config import HydraConfig


def main():
    with initialize(
        config_path="../configs", job_name="run_mlm", version_base="1.3"
    ):
        cfg = compose(
            config_name="config",
            overrides=["tokenizer=bpe_from_pretrained", "model=bert_for_mlm"],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)
        tokenizer = instantiate(cfg.tokenizer)
        bert_cfg = instantiate(
            cfg.model.config, vocab_size=tokenizer.vocab_size
        )
        model = AutoModelForMaskedLM.from_config(bert_cfg)
        dataset = get_dataset_from_processed()

        def tokenize_funct(examples):
            return tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
                return_special_tokens_mask=True,
            )

        tokenized_dataset = dataset["train"].map(
            tokenize_funct,
            batched=True,
            remove_columns=["text"],
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

        training_args = TrainingArguments()


if __name__ == "__main__":
    main()
