from data_pipeline.dataset_builder import DnsDatasetBuilder
from datasets import load_dataset
import tempfile
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

def test_dataset_splits_and_contents(sample_data):
    files = sample_data
    ds = load_dataset(
        path="src/data_pipeline/dataset_builder.py",
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

def test_dataset_info():
    builder = DnsDatasetBuilder()
    info = builder._info()
    assert "text" in info.features
    assert info.features["text"].dtype == "string"