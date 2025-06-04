import pytest
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

import data_pipeline.io as io


def test_list_files_flat_and_nested(tmp_path):
    raw = tmp_path / "raw"

    flat_train = raw / "train"
    flat_train.mkdir(parents=True, exist_ok=True)
    (flat_train / "positive.csv").write_text("a\nb\n")
    loader = io.RawSplitLoader(raw)
    files = loader.list_files("train", "positive")
    assert len(files) == 1
    assert files[0] == flat_train / "positive.csv"

    nested = flat_train / "negative"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "part1.csv").write_text("c\nd\n")
    (nested / "part2.csv").write_text("e\nf\n")
    files = loader.list_files("train", "negative")
    assert {f.name for f in files} == {"part1.csv", "part2.csv"}


def test_csv_domain_streamer_header(tmp_path):
    path = tmp_path / "test.csv"
    lines = ["Domain", "nike.com", "adidas.com", "puma.com"]
    path.write_text("\r\n".join(lines))
    streamer = io.CsvDomainStreamer(path, label=1)
    assert list(streamer) == [
        ("nike.com", 1),
        ("adidas.com", 1),
        ("puma.com", 1),
    ]


@pytest.mark.parametrize(
    "raw, keep_punycode, expected_ascii, expected_punycode",
    [
        ("WWW.Example.COM", False, "www.example.com", ""),
        ("a..b.com", False, "a.b.com", ""),
        ("xn--caf-dma.com", True, "xn--caf-dma.com", "café.com"),
    ],
)
def test_domain_normalizer(
    raw, keep_punycode, expected_ascii, expected_punycode
):
    ascii_part, unicode_part = io.DomainNormalizer.normalize(
        raw=raw, keep_puncycode=keep_punycode
    )
    assert ascii_part == expected_ascii
    assert unicode_part == expected_punycode


@pytest.mark.parametrize(
    "domain, valid_flag",
    [
        ("example.com", True),
        ("-invalid.com", False),
        ("idegascina" * 40 + ".com", False),
        ("too..much..dots.com", False),
        ("val-id.com", True),
    ],
)
def test_domain_validator(domain, valid_flag):
    is_valid, _ = io.DomainValidator.is_valid(domain)
    assert is_valid == valid_flag


def test_count_and_iterate(tmp_path):
    raw = tmp_path / "raw"
    val_dir = raw / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "negative.csv").write_text("hdr\nx\nx\n")
    (val_dir / "positive.csv").write_text("hdr\ny\ny\n")
    loader = io.RawSplitLoader(raw)
    iterator = io.RawDataIterator(loader)
    assert iterator.count_dmn_in_splt("val") == 4

    items = list(iterator.iterate(("val",)))
    negatives = [d for (split, d, lbl) in items if lbl == 0]
    positives = [d for (split, d, lbl) in items if lbl == 1]
    assert len(negatives) == 2
    assert len(positives) == 2
    assert all(split == "val" for (split, _, _) in items)


def test_raw_to_normalized_csv(tmp_path):
    base = tmp_path / "raw"
    base.mkdir(parents=True, exist_ok=True)
    train = base / "train"
    train.mkdir(parents=True, exist_ok=True)
    pos = train / "positive.csv"
    neg = train / "negative.csv"
    pos.write_text("domain,label\nnike.com,1\nadidas.com,1\nfa ke,1\n")
    neg.write_text(
        "domain,label\nfacebook.com,0\nfacebook.com,0\nbad..domain,0\n"
    )
    out = tmp_path / "processed"
    io.raw_to_normalized_csv(base, out, splits=("train",))
    proc_csv = out / "train.csv"
    assert proc_csv.exists()
    df = pd.read_csv(proc_csv)
    assert list(df.columns) == ["text", "label", "ok", "reason", "dup_gid"]
    assert len(df) == 6
    row_nike = df[df["text"] == "nike.com"].iloc[0]
    assert bool(row_nike["ok"]) is True
    row_fake = df[df["text"] == "fa ke"].iloc[0]
    assert bool(row_fake["ok"]) is False
    row_bad = df[df["text"] == "bad.domain"].iloc[0]
    assert bool(row_bad["ok"]) is True


def test_build_mlm_csvs(tmp_path):
    processed = tmp_path / "processed"
    original = processed / "original"
    original.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "text": ["a.com", "b.com", "c.com", "d.com", "e.com"],
            "label": [0, 1, 0, 1, 0],
            "ok": [True, True, True, False, True],
            "reason": [0, 0, 0, 4, 0],
            "dup_gid": [1, 2, 2, 3, 1],
        }
    )

    for split in ("train", "val", "test"):
        df.to_csv(original / f"{split}.csv", index=False)

    io.build_mlm_csvs(original, processed, splits=("train", "val", "test"))
    mlm_dir = processed / "mlm"
    for split in ("train", "val", "test"):
        path = mlm_dir / f"{split}.csv"
        assert path.exists()
        subdf = pd.read_csv(path)
        assert set(subdf["text"].tolist()).issubset({"a.com", "b.com"})


def test_build_cls_csvs(tmp_path):
    processed = tmp_path / "processed"
    original = processed / "original"
    original.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "text": ["w.com", "x.com", "y.com", "z.com"],
            "label": [0, 0, 1, 1],
            "ok": [True, True, True, True],
            "reason": [0, 0, 0, 0],
            "dup_gid": [1, 1, 2, 2],
        }
    )
    for split in ("train", "val", "test"):
        df.to_csv(original / f"{split}.csv", index=False)

    io.build_cls_csvs(original, processed, splits=("train", "val", "test"))
    cls_dir = processed / "cls"
    out_files = {f.name for f in cls_dir.iterdir() if f.is_file()}
    splits_gids = {}
    for split in ("train", "val", "test"):
        subdf = pd.read_csv(cls_dir / f"{split}.csv")
        gids = set(subdf["dup_gid"].tolist())
        splits_gids[split] = gids

    all_seen = splits_gids["train"] | splits_gids["val"] | splits_gids["test"]
    assert all_seen == {1, 2}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_raw_to_normalized_csv(Path(tmpdir))
