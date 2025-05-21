import os
import csv
import pytest
import torch

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
                if max_length and max_length < self.model_max_length
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


def test_mlm_dataset_builder_splits_and_tokenization(dummy_csv):
    data_files = dummy_csv
    max_len = 5
    tokenizer = DummyTokenizer(max_seq_len=max_len)
    builder = MLMDatasetBuilder(
        raw_files=data_files,
        tokenizer=tokenizer,
        streaming=False,
        max_length=max_len,
        proportion=(1 / 3, 1 / 3, 1 / 3),
    )

    ds = builder.build()

    assert set(ds.keys()) == {"train", "validation", "test"}

    for split in ds:
        assert len(ds[split]) == 1
        ex = ds[split][0]
        input_ids = ex["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        assert 1 <= len(input_ids) <= max_len


def test_cls_dataset_builder_splits_and_labels(dummy_csv):
    data_files = dummy_csv
    max_len = 4
    tokenizer = DummyTokenizer(max_seq_len=max_len)
    builder = CLSDatasetBuilder(
        raw_files=data_files,
        tokenizer=tokenizer,
        streaming=False,
        max_length=max_len,
        proportion=(1 / 3, 1 / 3, 1 / 3),
    )

    ds = builder.build()

    assert set(ds.keys()) == {"train", "validation", "test"}
    for split in ds:
        assert len(ds[split]) == 1
        ex = ds[split][0]
        input_ids = ex["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        assert 1 <= len(input_ids) <= max_len

        label = ex.get("label")
        assert isinstance(label, torch.Tensor)
        assert label.item() in {0, 1}
