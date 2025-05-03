import pytest
import csv
import sqlite3
import data_pipeline.io as io
import tempfile
from pathlib import Path


def test_small_dedup():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        txt = tmp_path / "demo.txt"
        lbl = tmp_path / "demo.labels"
        txt.write_text("a.com\na.com\nb.net\n")
        lbl.write_text("1\n1\n0\n")
        uniq_t, uniq_l = io.remove_duplicates(txt, lbl)
        assert uniq_t.read_text().splitlines() == ["a.com", "b.net"]
        assert uniq_l.read_text().splitlines() == ["1", "0"]


def test_list_files_flat_and_nested():
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
        flat_train = tmp_path / "raw" / "train"
        flat_train.mkdir(parents=True)
        (flat_train / "positive.csv").write_text("a\nb\n")
        loader = io.RawSplitLoader(tmp_path / "raw")
        files = loader.list_files("train", "positive")
        assert len(files) == 1
        assert files[0] == flat_train / "positive.csv"

        nested = tmp_path / "raw" / "train" / "negative"
        nested.mkdir(parents=True)
        (nested / "part1.csv").write_text("x\ny\n")
        (nested / "part2.csv").write_text("z\nw\n")
        files = loader.list_files("train", "negative")
        assert {f.name for f in files} == {"part1.csv", "part2.csv"}


def test_csv_domain_streamer_header():
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
        path = tmp_path / "test.csv"
        lines = ["Domain", "nike.com", "adidas.com", "buzz.com"]
        path.write_text("\r\n".join(lines))
        streamer = io.CsvDomainStreamer(path, label=1)
        out = list(streamer)
        assert out == [
            ("nike.com", 1),
            ("adidas.com", 1),
            ("buzz.com", 1),
        ]


@pytest.mark.parametrize(
    "raw, uni, expected",
    [
        ("WWW.Example.COM", False, "example.com"),
        ("a..b.com", False, "a.b.com"),
        ("xn--caf-dma.com", True, "café.com"),
    ],
)
def test_domain_normalizer(raw, uni, expected):
    assert io.DomainNormalizer.normalize(raw, to_unicode=uni) == expected


@pytest.mark.parametrize(
    "domain, valid",
    [
        ("example.com", True),
        ("-invalid.com", False),
        ("idegascina" * 40 + ".com", False),
        ("too..much..dots.com", False),
        ("val-id.com", True),
    ],
)
def test_domain_validator(domain, valid):
    assert io.DomainValidator.is_valid(domain) == valid


def test_count():
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
        path = tmp_path / "raw"
        for label in ("negative", "positive"):
            p = path / "val"
            p.mkdir(parents=True, exist_ok=True)
            f = p / f"{label}.csv"
            f.write_text("header\nx\nx\n")
        loader = io.RawSplitLoader(path)
        iterator = io.RawDataIterator(loader)
        assert iterator.count_dmn_in_splt("val") == 4


def test_raw_to_norm():
    with tempfile.TemporaryDirectory() as tmp_path:
        base = Path(tmp_path) / "raw"
        base.mkdir()
        train = base / "train"
        train.mkdir()
        pos = train / "positive.csv"
        neg = train / "negative.csv"
        pos.write_text("hdr\nnike.com\nadidas.com\nfa ke\n")
        neg.write_text("hdr\nfacebook.com\nfacebook.com\n")
        out = Path(tmp_path) / "norm"
        io.raw_to_normalized(base, out, ("train",), to_unicode=False)

        txt = (out / "train.txt").read_text().splitlines()
        lbl = (out / "train.labels").read_text().splitlines()
        assert set(txt) == {"nike.com", "adidas.com", "facebook.com"}
        assert set(lbl) == {"1", "0"}


def test_remove_duplicates():
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_path = Path(tmp_path)
        txt = tmp_path / "demo.txt"
        lbl = tmp_path / "demo.labels"
        txt.write_text("a\nb\n")
        lbl.write_text("1\n0\n")
        uniq_t, uniq_l = io.remove_duplicates(txt, lbl)
        assert not (tmp_path / "demo.db").exists()
