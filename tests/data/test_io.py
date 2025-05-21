import pytest
import data_pipeline.io as io
from pathlib import Path
import csv

def test_list_files_flat_and_nested(tmp_path):
    raw = tmp_path / "raw"

    flat_train = raw / "train"
    flat_train.mkdir(parents=True)
    (flat_train / "positive.csv").write_text("a\nb\n")
    loader = io.RawSplitLoader(raw)
    files = loader.list_files("train", "positive")
    assert len(files) == 1
    assert files[0] == flat_train / "positive.csv"

    nested = flat_train / "negative"
    nested.mkdir()
    (nested / "part1.csv").write_text("x\ny\n")
    (nested / "part2.csv").write_text("z\nw\n")
    files = loader.list_files("train", "negative")
    assert {f.name for f in files} == {"part1.csv", "part2.csv"}


def test_csv_domain_streamer_header(tmp_path):
    path = tmp_path / "test.csv"
    lines = ["Domain", "nike.com", "adidas.com", "buzz.com"]
    path.write_text("\r\n".join(lines))
    streamer = io.CsvDomainStreamer(path, label=1)
    assert list(streamer) == [
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


def test_count(tmp_path):
    raw = tmp_path / "raw"
    for label in ("negative", "positive"):
        p = raw / "val"
        p.mkdir(parents=True, exist_ok=True)
        f = p / f"{label}.csv"
        f.write_text("header\nx\nx\n")
    loader = io.RawSplitLoader(raw)
    iterator = io.RawDataIterator(loader)
    assert iterator.count_dmn_in_splt("val") == 4


def test_raw_to_norm(tmp_path):
    base = tmp_path / "raw"
    base.mkdir()
    train = base / "train"
    train.mkdir()
    pos = train / "positive.csv"
    neg = train / "negative.csv"
    pos.write_text("hdr\nnike.com\nadidas.com\nfa ke\n")
    neg.write_text("hdr\nfacebook.com\nfacebook.com\n")
    out = tmp_path / "norm"
    io.raw_to_normalized(base, out, ("train",), to_unicode=False)

    txt = (out / "train.txt").read_text().splitlines()
    lbl = (out / "train.labels").read_text().splitlines()
    assert set(txt) == {"nike.com", "adidas.com", "facebook.com"}
    assert set(lbl) == {"1", "0"}


def test_remove_duplicates_small(tmp_path):
    input_csv = tmp_path / "demo.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerow(["a.com", "1"])
        writer.writerow(["a.com", "1"])
        writer.writerow(["b.net", "0"])
    uniq_csv = io.remove_duplicates(input_csv, out_dir=tmp_path)
    lines = uniq_csv.read_text(encoding="utf-8").splitlines()
    assert lines == ["text,label", "a.com,1", "b.net,0"]


def test_remove_duplicates_db_cleanup(tmp_path):
    input_csv = tmp_path / "demo.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerow(["x.com", "0"])
    _ = io.remove_duplicates(input_csv, out_dir=tmp_path)
    db_path = tmp_path / f"{input_csv.stem}.db"
    assert not db_path.exists()