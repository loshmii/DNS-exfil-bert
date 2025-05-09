import pytest
import tempfile
from data_pipeline.dataset import _build_file_map
from pathlib import Path


def test_build_file_map_processed():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        proc = root / "data" / "processed"
        proc.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (proc / f"{split}.txt").write_text(f"{split}_line")
        fm = _build_file_map(root, layout="processed")
        assert list(fm.keys()) == ["train", "val", "test"]
        assert all(
            Path(path_list[0]).exists()
            and path_list[0].endswith(f"{split}.txt")
            for split, path_list in fm.items()
        )


def test_build_file_map_raw():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        raw = root / "data" / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            d = raw / split
            d.mkdir(parents=True, exist_ok=True)
            (d / "positive.csv").write_text("Subdomain,Exfiltration\nA,1")
            (d / "negative.csv").write_text("Subdomain,Exfiltration\nB,0")
        fm = _build_file_map(root, layout="raw")
        assert set(fm.keys()) == {"train", "val", "test"}
        for split, paths in fm.items():
            assert len(paths) == 2
            assert all(Path(p).exists() for p in paths)
