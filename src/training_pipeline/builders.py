from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast, AutoTokenizer
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pandas as pd


@dataclass
class DnsDatasetBuilder(ABC):
    raw_files: Dict[str, List[str]]
    tokenizer: PreTrainedTokenizerFast
    max_length: int = 256
    streaming: bool = False
    cache_dir: Optional[str] = None

    def build(self) -> DatasetDict:
        if (
            self.cache_dir
            and Path(self.cache_dir).exists()
            and any(Path(self.cache_dir).iterdir())
        ):
            return load_from_disk(self.cache_dir)

        ds = self._load_raw()
        ds = self._dedupe(ds)
        ds = ds.map(self._tokenize, batched=True, remove_columns=["text"])
        ds = self._postprocess(ds)
        ds.set_format("torch")

        if self.cache_dir:
            p = Path(self.cache_dir)
            if not p.exists() or not any(p.iterdir()):
                ds.save_to_disk(str(p))
        return ds

    def _load_raw(self) -> DatasetDict:

        return load_dataset(
            "csv",
            data_files=self.raw_files,
        )

    def _dedupe(self, ds: DatasetDict) -> DatasetDict:
        return ds

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

    @abstractmethod
    def _postprocess(self, ds: DatasetDict) -> DatasetDict: ...


class MLMDatasetBuilder(DnsDatasetBuilder):

    def _postprocess(self, ds):
        return ds


class CLSDatasetBuilder(DnsDatasetBuilder):
    label2id: Optional[Dict[str, int]] = None
    label2id = {
        "0": 0,
        "1": 1,
    }

    def _postprocess(self, ds):
        def add_label(example):
            example["label"] = self.label2id[example["label"]]
            return example

        return ds.map(add_label, batched=True)


if __name__ == "__main__":
    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="trainer_test",
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "tokenizer=bpe8k_pretrained",
                "dataset=dataset_for_labeled",
            ],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    data_files = OmegaConf.to_container(cfg.dataset.files, resolve=True)

    builder = CLSDatasetBuilder(
        raw_files=data_files,
        tokenizer=tokenizer,
        max_length=tokenizer.model_max_length,
        streaming=True,
        cache_dir=str(
            Path.cwd() / "experiments" / "cache"
        ),  # TODO: make this configurable
    )

    ds = builder.build()
    print(ds)
