import pytest
from datasets import DatasetDict
import tempfile
from pathlib import Path
from data_pipeline.dataset import DataArguments, load_dns_dataset

def test_load_processed_no_tokenizer():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "data" / "processed"
        root.mkdir(parents=True, exist_ok=True)
        lines = ["a.com", "b.net", "c.org"]
        for split in ["train", "val", "test"]:
            (root / f"{split}.txt").write_text("\n".join(lines))
        args = DataArguments(
            root=str(root.parent.parent),
            layout="processed",
            block_size=5,
        )
        ds: DatasetDict = load_dns_dataset(args, tokenizer=None)
        assert set(ds) == {"train", "val", "test"}
        assert ds["train"].column_names == ["text"]
        assert ds["train"]["text"] == lines

def test_load_raw_no_tokenizer():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "data" / "raw"
        root.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            d = root / split
            d.mkdir(parents=True, exist_ok=True)
            (d / "positive.csv").write_text("Subdomain,Exfiltration\nA,1")
            (d / "negative.csv").write_text("Subdomain,Exfiltration\nB,0")
        args = DataArguments(
            root = str(root.parent.parent),
            layout = "raw",
            block_size = 10,
        )
        ds: DatasetDict = load_dns_dataset(args, tokenizer=None)
        assert "text" in ds["train"].column_names
        assert "Exfiltration" not in ds["train"].column_names
        assert set(ds["train"]["text"]) == {"A", "B"}

def test_load_with_stub_tokenizer():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "data" / "processed"
        root.mkdir(parents=True, exist_ok=True)
        (root / "train.txt").write_text("nike\nadidas")
        (root / "val.txt").write_text("puma\nreebok")
        (root / "test.txt").write_text("underarmour\nnewbalance")
        args = DataArguments(
            root=str(root.parent.parent),
            layout="processed",
            block_size=4,
        )
        def stub_tokenizer(batch, **kwargs):
            batch_size = len(batch)
            return {
                "input_ids": [[1,2,3,0] for _ in batch],
                "attention_mask": [[1,1,1,0] for _ in batch],
                "special_tokens_mask": [[0,0,0,1] for _ in batch],
            }
        
        ds: DatasetDict = load_dns_dataset(args, tokenizer=stub_tokenizer)
        assert "text" not in ds["train"].column_names
        assert set(ds["train"].column_names) == {"input_ids", "attention_mask", "special_tokens_mask"}
        for ids, mask, sp in zip(ds["train"]["input_ids"], ds["train"]["attention_mask"], ds["train"]["special_tokens_mask"]):
            assert len(ids) == 4
            assert len(mask) == 4
            assert len(sp) == 4