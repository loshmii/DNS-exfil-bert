from pathlib import Path
import tempfile
from datasets import load_from_disk
import numpy as np
import torch
import pytest

from training_pipeline.builders import CLSDatasetBuilder
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import (
    CharTokConfig,
    CharTokenizer
)

BASE = Path(__file__).parent.parent.parent
RAW_CSV_ROOT = BASE / "data" / "processed" / "subset"
TOKENIZER_CFG = BASE / "configs" / "tokenizer" / "char.yaml"

def _build_ds():
    tok = CharTokenizer(
        CharTokConfig(TOKENIZER_CFG)
    )

    builder = CLSDatasetBuilder(
        tokenizer=tok,
        raw_files={
            "train": str(RAW_CSV_ROOT / "train_100k.csv"),
            "validation": str(RAW_CSV_ROOT / "val_100k.csv"),
            "test": str(RAW_CSV_ROOT / "test_100k.csv")
        },
        max_length=128,
        cache_dir=None,
        force_rebuild=True,
    )

    ds = builder.build()

    ds["train"]      = ds["train"].select(range(min(len(ds["train"]),      1_000)))
    ds["validation"] = ds["validation"].select(range(min(len(ds["validation"]), 1_000)))
    ds["test"]       = ds["test"].select(range(min(len(ds["test"]),       1_000)))
    return ds

@pytest.mark.parametrize("roundtrip", [False, True])
def test_dup_gid_integrity(roundtrip, tmp_path):
    ds = _build_ds()

    if roundtrip:
        save_dir = tmp_path / "hf_ds"
        ds.save_to_disk(save_dir)
        ds = load_from_disk(save_dir)

    col_iter_set = {int(x) for x in ds["train"]["dup_gid"]}
    unique_set = {int(x) for x in ds["train"].unique("dup_gid")}

    assert col_iter_set, "dup_gid column is empty - test bug?"
    assert col_iter_set == unique_set, (
        "Mismatch between iterated and unique dup_gid values\n"
        f"|iterated|={len(col_iter_set)} | unique|={len(unique_set)}"
    )