from pathlib import Path
from typing import List, IO, Iterator, Tuple, Sequence, ClassVar, Union
import time
from tqdm import tqdm
import csv
import re
import idna
import sqlite3


class RawSplitLoader:
    def __init__(self, base_path: Path):
        if not base_path.exists():
            raise FileNotFoundError(f"Base path {base_path} does not exist.")
        self.base_path = base_path

    def list_files(self, split: str, label: str) -> List[Path]:
        split_dir = self.base_path / split
        if not split_dir.is_dir():
            return []
        div_variant = split_dir / label
        if div_variant.is_dir():
            return sorted(p for p in div_variant.rglob("*.csv") if p.is_file())
        files_variant = split_dir / f"{label}.csv"
        if files_variant.is_file():
            return [files_variant]
        return sorted(
            p
            for p in split_dir.rglob("*.csv")
            if label in p.stem and p.is_file()
        )


class CsvDomainStreamer:
    def __init__(self, file_path: Path, label: int):
        self.file_path = file_path
        self.label = label

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8", newline="") as f:
            yield from self._read_csv(f)

    def _read_csv(self, handle: IO[str]) -> Iterator[Tuple[str, int]]:
        reader = csv.reader(handle, strict=True)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            domain = row[0].strip()
            yield domain, self.label


class RawDataIterator:
    def __init__(self, loader: RawSplitLoader):
        self.loader = loader

    def iterate(self, splits: Sequence[str] = ("train", "val", "test")):
        for split in splits:
            for label_str, label_int in (("negative", 0), ("positive", 1)):
                for file_path in self.loader.list_files(split, label_str):
                    streamer = CsvDomainStreamer(file_path, label_int)
                    for domain, lbl in streamer:
                        yield split, domain, lbl

    def count_dmn_in_splt(self, split: str):
        cnt = 0
        for label in ("negative", "positive"):
            for fp in self.loader.list_files(split, label):
                with open(fp, "r", encoding="utf-8") as f:
                    cnt += sum(1 for _ in f) - 1
        return cnt


class DomainValidator:
    LABEL_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$"
    )
    MAX_DOMAIN_LENGTH: ClassVar[int] = 253

    @classmethod
    def is_valid(cls, domain: str) -> bool:
        if not (1 <= len(domain) <= cls.MAX_DOMAIN_LENGTH):
            return False

        if ".." in domain:
            return False

        blacklisted_labels = set(
            [
                "null",
                "none",
                "NaN",
            ]
        )

        labels = domain.split(".")
        for label in labels:
            if (
                not cls.LABEL_PATTERN.match(label)
                or label in blacklisted_labels
            ):
                return False

        return True


class CSVQueryValidator:
    LINE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"^(?P<domain>[^,]+),(?P<label>[01])$"
    )

    @classmethod
    def is_valid(cls, line: str) -> bool:
        line = line.rstrip("\n\r")
        m = cls.LINE_PATTERN.match(line)
        if not m:
            return False

        domain = m.group("domain")
        label = m.group("label")

        if not DomainValidator.is_valid(DomainNormalizer.normalize(domain)):
            return False

        return True


class DomainNormalizer:

    DOT_COLLAPSE_PATTERN: ClassVar[re.Pattern] = re.compile(r"\.{2,}")

    @staticmethod
    def normalize(raw: str, to_unicode: bool = True) -> str:
        dom = raw.strip().lower()
        if dom.endswith("."):
            dom = dom[:-1]
        dom = DomainNormalizer.DOT_COLLAPSE_PATTERN.sub(".", dom)
        if dom.startswith("www."):
            dom = dom[4:]
        if to_unicode:
            try:
                dom = idna.decode(dom)
            except idna.IDNAError:
                pass
        return dom


def raw_to_normalized(
    raw_base: Union[str, Path],
    out_base: Union[str, Path],
    splits: Tuple[str, ...] = ("train", "val", "test"),
    to_unicode: bool = False,
):
    path = Path(raw_base) if isinstance(raw_base, str) else raw_base
    out_path = Path(out_base) if isinstance(out_base, str) else out_base
    out_path.mkdir(parents=True, exist_ok=True)

    loader = RawSplitLoader(path)
    iterator = RawDataIterator(loader)

    txt_files = {
        split: open(out_path / f"{split}.txt", "w", encoding="utf-8")
        for split in splits
    }
    lbl_files = {
        split: open(out_path / f"{split}.labels", "w", encoding="utf-8")
        for split in splits
    }

    try:
        for split in splits:
            cnt = iterator.count_dmn_in_splt(split)
            seen = valid = 0
            start = time.perf_counter()

            for _, raw_dom, label in tqdm(
                iterator.iterate((split,)),
                total=cnt,
                desc=f"Processing {split}",
                unit="doms",
                colour="green",
                dynamic_ncols=True,
            ):
                seen += 1

                dom = DomainNormalizer.normalize(raw_dom, to_unicode)
                if not DomainValidator.is_valid(dom):
                    continue

                txt_files[split].write(f"{dom}\n")
                lbl_files[split].write(f"{label}\n")
                valid += 1

            elapsed = time.perf_counter() - start
            rate = seen / elapsed if elapsed > 0 else 0.0
            print(
                f"[{split:5s}] kept {valid}/{seen}"
                f"in {elapsed: .1f}s ({rate :.1f} dom/s)"
            )

    finally:
        for f in (*txt_files.values(), *lbl_files.values()):
            f.close()


def raw_to_normalized_csv(
    raw_base: Union[str, Path],
    out_base: Union[str, Path],
    splits: Tuple[str, ...] = ("train", "val", "test"),
    to_unicode: bool = False,
):

    raw_base = Path(raw_base)
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    loader = RawSplitLoader(raw_base)
    iterator = RawDataIterator(loader)

    csv_files = {}
    writers = {}
    for split in splits:
        f = open(out_base / f"{split}.csv", "w", encoding="utf-8", newline="")
        w = csv.writer(f)
        w.writerow(["text", "label"])
        csv_files[split] = f
        writers[split] = w

    try:
        for split in splits:
            total = iterator.count_dmn_in_splt(split)
            seen = valid = 0
            start = time.perf_counter()

            for _, raw_dom, label in tqdm(
                iterator.iterate((split,)),
                total=total,
                desc=f"Normalizing {split}",
                unit="doms",
                colour="green",
                dynamic_ncols=True,
            ):
                seen += 1

                dom = DomainNormalizer.normalize(raw_dom, to_unicode)
                label_str = str(label)
                line = f"{dom},{label_str}"
                if not CSVQueryValidator.is_valid(line):
                    continue

                writers[split].writerow([dom, label_str])
                valid += 1

            elapsed = time.perf_counter() - start
            rate = seen / elapsed if elapsed > 0 else 0.0
            print(
                f"[{split:5s}] kept {valid}/{seen}"
                f"in {elapsed: .1f}s ({rate :.1f} dom/s)"
            )
    finally:
        for f in csv_files.values():
            f.close()


def remove_duplicates(
    split_csv: Union[str, Path],
    out_dir: Union[str, Path] = None,
    batch_size: int = 50000,
) -> Path:
    split_csv = Path(split_csv)
    out_dir = Path(out_dir) if out_dir else split_csv
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"{split_csv.stem}.db"
    uniq_csv = out_dir / f"{split_csv.stem}.csv"

    with split_csv.open("r", encoding="utf-8", newline="") as f:
        total = sum(1 for _ in f) - 1

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=OFF;")
        cur.execute("PRAGMA synchronous=OFF;")
        conn.commit()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS domains (
                text TEXT PRIMARY KEY,
                label TEXT NOT NULL
                );
        """
        )
        conn.commit()

        batch = []
        with split_csv.open("r", encoding="utf-8", newline="") as rf:
            reader = csv.reader(rf)
            header = next(reader)
            for row in tqdm(
                reader,
                total=total,
                desc="Inserting into DB",
                unit="rows",
                colour="green",
            ):
                dom, lbl = row[0].strip(), row[1].strip()
                batch.append((dom, lbl))
                if len(batch) >= batch_size:
                    cur.executemany(
                        "INSERT OR IGNORE INTO domains (text, label) VALUES (?, ?);",
                        batch,
                    )
                    conn.commit()
                    batch.clear()
            if batch:
                cur.executemany(
                    "INSERT OR IGNORE INTO domains (text, label) VALUES (?, ?);",
                    batch,
                )
                conn.commit()
        with uniq_csv.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.writer(wf)
            writer.writerow(header)
            for txt, lbl in tqdm(
                cur.execute("SELECT text, label FROM domains ORDER BY rowid;"),
                desc="Exporting to CSV",
                unit="rows",
                colour="green",
            ):
                writer.writerow([txt, lbl])
    finally:
        conn.close()
        try:
            db_path.unlink()
        except FileNotFoundError:
            pass

    return uniq_csv


# TODO: Change the logic of the file so the inner functions work on files and not on the whole directory then have outer logic to call the inner functions


if __name__ == "__main__":
    DIR = Path(__file__).parent.parent.parent.resolve()

    raw_base = DIR / "data" / "raw"
    out_base_csv = DIR / "data" / "processed" / "original"
    raw_to_normalized_csv(
        raw_base,
        out_base_csv,
    )
    out_base = DIR / "data" / "processed" / "deduped"

    splits = ("train", "val", "test")
    for split in splits:
        inp_file = out_base_csv / f"{split}.csv"
        remove_duplicates(
            inp_file,
            out_dir=out_base,
        )
        print(f"Removed duplicates from {split} split.")
    print("All done.")
