from training_pipeline.dataset_builder import DnsDatasetBuilder
from datasets import load_dataset
from pathlib import Path
import pytest


@pytest.fixture
def sample_data(tmp_path):
    root = Path(tmp_path)
    lines = ["a.com", "b.net", "c.org"]
    files = {}
    for split in ("train", "validation", "test"):
        f = root / f"{split}.txt"
        f.write_text("\n".join(lines), encoding="utf-8")
        files[split] = str(f)
    return files

@pytest.fixture
def sample_multiple_fies(tmp_path):
    root = Path(tmp_path)
    lines = ["a.com", "b.net", "c.org"]
    files = {}
    for split in ("train", "validation", "test"):
        f_1 = root / f"{split}_1.txt"
        f_2 = root / f"{split}_2.txt"
        f_1.write_text("\n".join(lines), encoding="utf-8")
        f_2.write_text("\n".join(lines), encoding="utf-8")
        files[split] = [str(f_1), str(f_2)]
    return files

def test_dataset_splits_and_contents(sample_data):
    files = sample_data
    ds = load_dataset(
        path="src/training_pipeline/dataset_builder.py",
        name="default",
        data_files=files,
        streaming=False,
        trust_remote_code=True,
    )

    assert set(ds.keys()) == {"train", "validation", "test"}
    for name in ("train", "validation", "test"):
        expected = Path(files[name]).read_text(encoding="utf-8").splitlines()
        texts = [ex["text"] for ex in ds[name]]
        assert texts == expected

def test_dataset_multiple_files(sample_multiple_fies):
    files = sample_multiple_fies
    ds = load_dataset(
        path="src/training_pipeline/dataset_builder.py",
        name="default",
        data_files=files,
        streaming=False,
        trust_remote_code=True,
    )
    assert set(ds.keys()) == {"train", "validation", "test"}
    for name in ("train", "validation", "test"):
        expected = []
        for file in files[name]:
            expected.extend(Path(file).read_text(encoding="utf-8").splitlines())
        texts = [ex["text"] for ex in ds[name]]
        assert texts == expected

def test_dataset_info():
    builder = DnsDatasetBuilder()
    info = builder._info()
    assert "text" in info.features
    assert info.features["text"].dtype == "string"

def test_streaming_mode(sample_data):
    files = sample_data
    ds = load_dataset(
        path="src/training_pipeline/dataset_builder.py",
        name="default",
        data_files=files,
        streaming=True,
        trust_remote_code=True,
    )

    first = next(iter(ds["train"]))
    assert isinstance(first["text"], str)