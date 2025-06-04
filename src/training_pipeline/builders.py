from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from datasets import (
    DatasetDict,
    Dataset,
    IterableDatasetDict,
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
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import (
    CharTokenizer,
    CharTokConfig,
)
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import shutil
import json
import hashlib
import os
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import math

BASE = Path(__file__).parent.parent.parent


def _arrow_sqrt_freq(
    table: pa.Table,
    col: str = "dup_gid",
):
    vc = pc.value_counts(table[col])
    values = vc.field(0)
    counts = vc.field(1)
    dup_ids = values.to_pylist()
    counts = counts.to_pylist()
    return {int(g): math.sqrt(c) for g, c in zip(dup_ids, counts)}


@dataclass  # TODO: unify loading for MLM and CLS no need to reject MLM in CLS case
class DnsDatasetBuilder(ABC):
    raw_files: Dict[str, List[str]]
    tokenizer: PreTrainedTokenizerFast
    streaming: bool = False
    max_length: Optional[int] = None
    cache_dir: Optional[str] = None
    force_rebuild: Optional[bool] = False
    seed: Optional[int] = 42

    def __post_init__(self):
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
        else:
            return self._prepare()

    def _build_streaming(self) -> IterableDatasetDict:
        ds_stream = load_dataset(
            "csv",
            data_files=self.raw_files,
            streaming=True,
        )

        ds_stream = self._preprocess_streaming(ds_stream)

        ds_stream = ds_stream.map(
            self._tokenize,
            batched=True,
            num_proc=1,
            remove_columns=["text"],
        )

        ds_stream = self._postprocess_streaming(ds_stream)
        return ds_stream

    @abstractmethod
    def _preprocess_streaming(
        self,
        ds: IterableDatasetDict,
    ) -> IterableDatasetDict:
        raise NotImplementedError

    @abstractmethod
    def _postprocess(
        self,
        ds: IterableDatasetDict,
    ) -> IterableDatasetDict:
        raise NotImplementedError

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

        ds_dict = load_dataset(
            "csv",
            data_files=self.raw_files,
        )
        ds_dict = self._preprocess(ds_dict)
        ds_dict = ds_dict.map(
            self._tokenize,
            batched=True,
            num_proc=16 if os.cpu_count() >= 16 else 4,
            remove_columns=["text"],
        )
        ds_dict = self._postprocess(ds_dict)
        ds_dict.set_format("torch")

        if cache_path:
            cache_path.mkdir(parents=True, exist_ok=True)

            ds_dict.save_to_disk(str(cache_path))

            meta = self._build_metadata()
            tmp = cache_path / "metadata.json.tmp"
            tmp.write_text(json.dumps(meta, indent=2))
            tmp.replace(meta_path)

        return ds_dict

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
    def _postprocess(self, ds: DatasetDict) -> DatasetDict:
        raise NotImplementedError

    @abstractmethod
    def _preprocess(self, ds: DatasetDict) -> DatasetDict:
        raise NotImplementedError


class MLMDatasetBuilder(DnsDatasetBuilder):

    def _preprocess(self, ds: DatasetDict) -> DatasetDict:
        return ds

    def _preprocess_streaming(
        self, ds: IterableDatasetDict
    ) -> IterableDatasetDict:
        return ds

    def _postprocess(self, ds: DatasetDict) -> DatasetDict:
        return ds

    def _postprocess_streaming(
        self, ds: IterableDatasetDict
    ) -> IterableDatasetDict:
        return ds


class CLSDatasetBuilder(DnsDatasetBuilder):
    label2id: Optional[Dict[str, int]] = None
    label2id = {
        "0": 0,
        "1": 1,
    }
    _dup_weight_map: Optional[Dict[int, float]] = None

    def _build_weight_map(self, ds: DatasetDict, cache_dir: Union[str, Path]):
        weight_file = Path(cache_dir) / "dup_gid_sqrt_freq.pt"
        weight_file.parent.mkdir(parents=True, exist_ok=True)

        if weight_file.exists():
            self._dup_weight_map = torch.load(weight_file)
            return

        arrow_tbl = ds["train"].data
        self._dup_weight_map = _arrow_sqrt_freq(arrow_tbl, col="dup_gid")
        torch.save(self._dup_weight_map, weight_file)

    def _preprocess(self, ds: DatasetDict) -> DatasetDict:
        ds = ds.remove_columns(["ok", "reason"])
        return ds

    def _preprocess_streaming(
        self, ds: IterableDatasetDict
    ) -> IterableDatasetDict:
        raise NotImplementedError(
            "Streaming preprocessing is not currently implemented for CLS task"
        )
        columns_to_drop = ["ok", "reason", "dup_gid"]
        return ds.remove_columns(columns_to_drop)

    def _postprocess(self, ds: DatasetDict) -> DatasetDict:
        cache_path = (
            Path(self.cache_dir) if self.cache_dir else Path(".tmp_weights")
        )
        self._build_weight_map(ds, cache_path)
        if self._dup_weight_map is not None:

            def add_weight(ex):
                return {
                    "sample_weight": self._dup_weight_map[int(ex["dup_gid"])]
                }

            ds["train"] = ds["train"].map(
                add_weight,
                num_proc=16 if os.cpu_count() >= 16 else 4,
                batched=False,
            )
        ds = DatasetDict(
            {
                "train": ds["train"],
                "validation": ds["validation"],
                "test": ds["test"],
            }
        )
        return ds.remove_columns(["special_tokens_mask"])

    def _postprocess_streaming(
        self, ds: IterableDatasetDict
    ) -> IterableDatasetDict:
        raise NotImplementedError(
            "Streaming postprocessing is not currently implemented for CLS task"
        )
        return ds.remove_columns(["special_tokens_mask"])

    def get_class_weights(
        self, ds: Optional[DatasetDict] = None
    ) -> torch.Tensor:
        if ds is None:
            ds = self.build()

        labels = np.asarray(ds["train"]["label"])

        classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=labels,
        )
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        return weights_tensor


@hydra.main(
    config_path=str(BASE / "configs"),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):

    tokenizer = CharTokenizer(
        CharTokConfig(
            BASE / "configs" / "tokenizer" / "char.yaml",
        )
    )

    """mlm_builder = MLMDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.MLM_builder_args, resolve=True),
    )

    ds = mlm_builder.build()
    print(ds["train"][0])"""

    cls_builder = CLSDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.CLS_builder_args, resolve=True),
    )
    ds = cls_builder.build()
    print(ds["train"][0])


if __name__ == "__main__":
    main()
