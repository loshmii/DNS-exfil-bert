from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from datasets import (
    DatasetDict,
    Dataset,
    load_dataset,
    load_from_disk,
    Features,
    Value,
    concatenate_datasets,
)
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import shutil
import json
import hashlib
import os


@dataclass
class DnsDatasetBuilder(ABC):
    raw_files: Dict[str, List[str]]
    tokenizer: PreTrainedTokenizerFast
    streaming: bool = False
    max_length: Optional[int] = None
    proportion: Optional[
        Union[DictConfig, Dict, Tuple[float, float, float]]
    ] = None
    cache_dir: Optional[str] = None
    force_rebuild: Optional[bool] = False
    seed: Optional[int] = 42

    def __post_init__(self):
        if not self.proportion:
            self.proportion = (0.8, 0.1, 0.1)
        if isinstance(self.proportion, DictConfig):
            self.proportion = OmegaConf.to_container(
                self.proportion, resolve=True
            )
        if isinstance(self.proportion, Dict):
            p_train = self.proportion.get("train", 0.8)
            p_val = self.proportion.get("validation", 0.1)
            p_test = self.proportion.get("test", 0.1)
            self.proportion = (p_train, p_val, p_test)
        if not isinstance(self.proportion, tuple):
            raise ValueError(
                "proportions must be a tuple of three floats or a dict with keys 'train', 'validation', and 'test'"
            )
        added = 0
        for prop in self.proportion:
            if prop < 0 or prop > 1:
                raise ValueError("Proportions must be between 0 and 1")
            added += prop
        if added != 1:
            raise ValueError("Proportions must sum to 1")
        if not self.max_length:
            self.max_length = self.tokenizer.model_max_length

        if (
            self.max_length > self.tokenizer.model_max_length
            or self.max_length < 0
        ):
            raise ValueError(
                f"max_length must be between 0 and {self.tokenizer.model_max_length}"
            )

    def build(self) -> DatasetDict:
        if self.streaming:
            return self._build_streaming()
        full = self._prepare()
        return self._split(full)

    def _build_streaming(self) -> DatasetDict:

        ds = self._load_raw()
        ds = self._preprocess(ds)
        ds = ds.map(
            self._tokenize,
            batched=True,
            num_proc=16 if os.cpu_count() >= 16 else 4,
            remove_columns=["text"],
        )
        ds = self._postprocess(ds)

        return ds

    def _prepare(self) -> Dataset:
        cache_path = Path(self.cache_dir) if self.cache_dir else None
        meta_path = cache_path / "metadata.json" if cache_path else None

        if self.force_rebuild and cache_path and cache_path.exists():
            shutil.rmtree(cache_path)

        if cache_path and cache_path.exists() and meta_path.exists():
            try:
                saved = json.loads(meta_path.read_text())
                if self._is_cache_valid(saved):
                    ds = load_from_disk(str(cache_path))
                    ds.set_format("torch")
                    return ds
                else:
                    raise ValueError("Cache mismatch")
            except Exception:
                shutil.rmtree(cache_path)

        ds_dict = self._load_raw()
        ds_dict = self._preprocess(ds_dict)
        ds_dict = ds_dict.map(
            self._tokenize,
            batched=True,
            num_proc=16 if os.cpu_count() >= 16 else 4,
            remove_columns=["text"],
        )
        ds_dict = self._postprocess(ds_dict)
        ds_dict.set_format("torch")

        full = concatenate_datasets(
            [
                ds_dict["train"],
                ds_dict.get("validation", ds_dict["train"].select([])),
                ds_dict.get("test", ds_dict["train"].select([])),
            ]
        )

        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)
            full.save_to_disk(str(cache_path))
            meta = self._build_metadata()
            tmp = cache_path / "metadata.json.tmp"
            tmp.write_text(json.dumps(meta, indent=2))
            tmp.replace(meta_path)

        return full

    def _split(self, full: Dataset) -> DatasetDict:
        p_train, p_val, p_test = self.proportion

        tr_vs_rest = full.train_test_split(
            train_size=p_train,
            shuffle=True,
            seed=self.seed,
        )
        train_ds = tr_vs_rest["train"]

        frac_val_of_rest = p_val / (p_val + p_test)

        val_ds, test_ds = (
            tr_vs_rest["test"]
            .train_test_split(
                train_size=frac_val_of_rest, shuffle=True, seed=self.seed
            )
            .values()
        )

        return DatasetDict(
            {
                "train": train_ds,
                "validation": val_ds,
                "test": test_ds,
            }
        )

    def _build_metadata(self) -> Dict[str, Any]:
        files_meta = {}
        for split, paths in self.raw_files.items():
            files_meta[str(split)] = {}
            for path in paths:
                path = Path(path)
                files_meta[str(split)][str(path)] = self._file_checksum(path)

        return {
            "builder_class": self.__class__.__name__,
            "max_length": self.max_length,
            "streaming": self.streaming,
            "raw_files": files_meta,
        }

    def _is_cache_valid(self, saved: Dict[str, Union[str, int]]) -> bool:
        if saved.get("builder_class") != self.__class__.__name__:
            return False
        for key in ("max_length", "streaming"):
            if saved.get(key) != getattr(self, key):
                return False
        for split, paths in self.raw_files.items():
            if split not in saved["raw_files"]:
                return False
            for path in paths:
                path = Path(path)
                old_checksum = saved["raw_files"][str(split)].get(str(path))
                if (
                    old_checksum is None
                    or old_checksum != self._file_checksum(path)
                ):
                    return False

        return True

    @staticmethod
    def _file_checksum(path: Path, chunk_size: int = 8192) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _reproport(self, ds: DatasetDict) -> DatasetDict:
        mixed_ds = concatenate_datasets(
            [ds["train"], ds["validation"], ds["test"]]
        )
        train_size = int(self.proportion[0] * mixed_ds.shape[0])
        val_size = int(self.proportion[1] * mixed_ds.shape[0])
        test_size = int(self.proportion[2] * mixed_ds.shape[0])
        train_val_split = mixed_ds.train_test_split(
            train_size=train_size,
            test_size=test_size + val_size,
            shuffle=True,
            seed=42,
        )
        train_ds = train_val_split["train"]
        val_test_split = train_val_split["test"].train_test_split(
            train_size=val_size, test_size=test_size, shuffle=True, seed=42
        )
        val_ds = val_test_split["train"]
        test_ds = val_test_split["test"]
        return DatasetDict(
            {
                "train": train_ds,
                "validation": val_ds,
                "test": test_ds,
            }
        )

    def _load_raw(self) -> DatasetDict:
        features = Features(
            {
                "text": Value(dtype="string"),
                "label": Value(dtype="int32"),
            }
        )

        return load_dataset(
            "csv",
            data_files=self.raw_files,
            streaming=self.streaming,
            features=features,
        )

    def _dedupe(self, ds: DatasetDict) -> DatasetDict:
        return ds

    def _tokenize(self, examples):
        return self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            padding="max_length",
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

    @abstractmethod
    def _postprocess(self, ds: DatasetDict) -> DatasetDict: ...

    @abstractmethod
    def _preprocess(self, ds: DatasetDict) -> DatasetDict: ...


class MLMDatasetBuilder(DnsDatasetBuilder):

    def _preprocess(self, ds):
        ds = ds.remove_columns(["label"])
        ds = self._dedupe(ds)
        return ds

    def _postprocess(self, ds):
        return ds


class CLSDatasetBuilder(DnsDatasetBuilder):
    label2id: Optional[Dict[str, int]] = None
    label2id = {
        "0": 0,
        "1": 1,
    }

    def _preprocess(self, ds):
        return ds

    def _postprocess(self, ds):
        return ds


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):

    tokenizer = BpeTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True),
    )

    mlm_builder = MLMDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.builder_args, resolve=True),
    )

    ds = mlm_builder.build()
    print(ds["train"][0])

    cls_builder = CLSDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.builder_args, resolve=True),
    )
    ds = cls_builder.build()
    print(ds["train"].shape[0], ds["validation"].shape[0], ds["test"].shape[0])


if __name__ == "__main__":
    main()
