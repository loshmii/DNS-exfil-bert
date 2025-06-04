import os
import csv
import pytest
import torch
import numpy as np

from training_pipeline.builders import (
    MLMDatasetBuilder,
    CLSDatasetBuilder,
    DnsDatasetBuilder,
)


class DummyTokenizer:
    def __init__(self, max_seq_len):
        self.model_max_length = max_seq_len

    def __call__(self, texts, max_length=None, **kwargs):
        seqs = []
        for text in texts:
            ml = (
                max_length
                if (
                    max_length is not None
                    and max_length < self.model_max_length
                )
                else self.model_max_length
            )
            ids = [ord(c) for c in text][:ml]
            seqs.append(ids)

        return {
            "input_ids": seqs,
            "attention_mask": [[1] * len(seq) for seq in seqs],
            "special_tokens_mask": [[0] * len(seq) for seq in seqs],
        }


@pytest.fixture
def dummy_csv(tmp_path):

    texts = ["a", "bb", "ccc"]
    labels = [0, 1, 0]
    splits = ["train", "validation", "test"]
    data_files = {}
    for split, idx in zip(splits, range(3)):
        path = tmp_path / f"{split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            writer.writerow([texts[idx], labels[idx]])
        data_files[split] = [str(path)]
    return data_files


@pytest.fixture
def dummy_csv_cls(tmp_path):

    train_texts = ["alpha", "beta"]
    train_labels = [0, 1]
    train_ok = [True, True]
    train_reason = [0, 0]
    train_dup = [1, 1]
    val_texts = ["gamma"]
    val_labels = [1]
    val_ok = [False]
    val_reason = [1]
    val_dup = [2]
    test_texts = ["delta"]
    test_labels = [0]
    test_ok = [True]
    test_reason = [0]
    test_dup = [3]

    data_files = {}
    train_path = tmp_path / "train.csv"
    with open(train_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "ok", "reason", "dup_gid"])
        for t, l, o, r, d in zip(
            train_texts, train_labels, train_ok, train_reason, train_dup
        ):
            writer.writerow([t, l, o, r, d])
    data_files["train"] = [str(train_path)]

    val_path = tmp_path / "validation.csv"
    with open(val_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "ok", "reason", "dup_gid"])
        for t, l, o, r, d in zip(
            val_texts, val_labels, val_ok, val_reason, val_dup
        ):
            writer.writerow([t, l, o, r, d])
    data_files["validation"] = [str(val_path)]

    test_path = tmp_path / "test.csv"
    with open(test_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label", "ok", "reason", "dup_gid"])
        for t, l, o, r, d in zip(
            test_texts, test_labels, test_ok, test_reason, test_dup
        ):
            writer.writerow([t, l, o, r, d])
    data_files["test"] = [str(test_path)]

    return data_files


def test_dnsdatasetbuilder_max_length_validation():
    tokenizer = DummyTokenizer(max_seq_len=10)
    raw_files = {"train": [], "validation": [], "test": []}

    with pytest.raises(ValueError):
        MLMDatasetBuilder(
            raw_files=raw_files,
            tokenizer=tokenizer,
            max_length=20,
        )

    with pytest.raises(ValueError):
        CLSDatasetBuilder(
            raw_files=raw_files,
            tokenizer=tokenizer,
            max_length=-5,
        )


def test_mlm_dataset_builder_splits_and_tokenization(dummy_csv):
    data_files = dummy_csv
    max_len = 5
    tokenizer = DummyTokenizer(max_seq_len=max_len)
    builder = MLMDatasetBuilder(
        raw_files=data_files,
        tokenizer=tokenizer,
        max_length=max_len,
        streaming=False,
    )

    ds = builder.build()

    assert set(ds.keys()) == {"train", "validation", "test"}

    for split in ["train", "validation", "test"]:
        assert len(ds[split]) == 1
        ex = ds[split][0]
        input_ids = ex["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        assert 1 <= input_ids.numel() <= max_len


def test_clsdatasetbuilder_preprocess_and_postprocess(dummy_csv_cls, tmp_path):
    data_files = dummy_csv_cls
    max_len = 6
    tokenizer = DummyTokenizer(max_seq_len=max_len)
    cache_dir = tmp_path / "cls_cache"
    builder = CLSDatasetBuilder(
        raw_files=data_files,
        tokenizer=tokenizer,
        streaming=False,
        max_length=max_len,
        cache_dir=str(cache_dir),
    )

    ds = builder.build()

    assert set(ds.keys()) == {"train", "validation", "test"}

    for split in ["train", "validation", "test"]:
        with pytest.raises(KeyError):
            _ = ds[split]["ok"]
        with pytest.raises(KeyError):
            _ = ds[split]["reason"]

    gid_train = ds["train"]["dup_gid"]
    assert gid_train is not None
    expected_weight = torch.sqrt(torch.tensor([2.0])).item()
    sample_weights = ds["train"]["sample_weight"]
    for w in sample_weights:
        assert abs(w.item() - expected_weight) < 1e-6

    with pytest.raises(KeyError):
        _ = ds["validation"]["sample_weight"]
    with pytest.raises(KeyError):
        _ = ds["test"]["sample_weight"]

    class_weights = builder.get_class_weights()
    assert isinstance(class_weights, torch.Tensor)
    assert torch.allclose(
        class_weights, torch.tensor([1.0, 1.0], dtype=torch.float)
    )


def test_clasdatasetbuilder_streaming_not_implemented(dummy_csv_cls):
    data_files = dummy_csv_cls
    tokenizer = DummyTokenizer(max_seq_len=4)
    builder = CLSDatasetBuilder(
        raw_files=data_files, tokenizer=tokenizer, streaming=True, max_length=4
    )
    with pytest.raises(NotImplementedError):
        builder.build()


def test_dnsdatasetbuilder_file_checksum(tmp_path):
    file_path = tmp_path / "test.txt"
    content = b"hello world"
    file_path.write_bytes(content)

    checksum = DnsDatasetBuilder._file_checksum(file_path)
    checksum2 = DnsDatasetBuilder._file_checksum(file_path)
    assert checksum == checksum2

    file_path.write_bytes(b"different content")
    checksum3 = DnsDatasetBuilder._file_checksum(file_path)
    assert checksum != checksum3
