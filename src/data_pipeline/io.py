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
            ["null",
            "none",
            "NaN",]
        )

        labels = domain.split(".")
        for label in labels:
            if not cls.LABEL_PATTERN.match(label) or label in blacklisted_labels:
                return False
            
        return True
    
class CSVQueryValidator:
    LINE_PATTERN: ClassVar[re.Pattern] = re.compile(
        r'^(?P<domain>[^,]+),(?P<label>[01])$'
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
    split_txt: Union[str, Path],
    split_lbl: Union[str, Path],
    tmp_dir: Union[str, Path] = None,
) -> Tuple[Path, Path]:
    tmp_dir = Path(tmp_dir) if tmp_dir else Path(split_txt).parent
    db_path = tmp_dir / f"{split_txt.stem}.db"
    uniq_txt = tmp_dir / f"{split_txt.stem}.uniq.txt"
    uniq_lbl = tmp_dir / f"{split_lbl.stem}.uniq.labels"

    cnt_rows = sum(1 for _ in split_txt.open("r", encoding="utf-8")) - 1

    conn = sqlite3.connect(db_path)

    try:
        c = conn.cursor()
        c.execute("PRAGMA journal_mode=OFF;")
        c.execute("PRAGMA synchronous=OFF;")
        conn.commit()

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS domains (
                domain TEXT PRIMARY KEY,
                label INTEGER
            );
        """
        )
        conn.commit()

        batch = []
        batch_size = 50000
        with split_txt.open("r", encoding="utf-8") as tf, split_lbl.open(
            "r", encoding="utf-8"
        ) as lf:

            iterator = zip(tf, lf)
            iterator = tqdm(
                iterator,
                total=cnt_rows,
                desc="Inserting rows",
                unit="rows",
                dynamic_ncols=True,
            )

            for dom, lbl in iterator:
                dom = dom.rstrip("\n")
                lbl = int(lbl.rstrip("\n"))
                batch.append((dom, lbl))

                if len(batch) >= batch_size:
                    c.executemany(
                        "INSERT OR IGNORE INTO domains (domain, label) VALUES (?, ?);",
                        batch,
                    )
                    conn.commit()
                    batch.clear()

            if batch:
                c.executemany(
                    "INSERT OR IGNORE INTO domains (domain, label) VALUES (?, ?);",
                    batch,
                )
                conn.commit()

        with uniq_txt.open("w", encoding="utf-8") as tf, uniq_lbl.open(
            "w", encoding="utf-8"
        ) as lf:

            rows = c.execute(
                "SELECT domain, label FROM domains ORDER BY rowid;"
            )
            for dom, lbl in tqdm(
                rows,
                desc="Exporting rows",
                unit="rows",
                dynamic_ncols=True,
            ):
                tf.write(f"{dom}\n")
                lf.write(f"{lbl}\n")
    finally:
        conn.close()

        try:
            db_path.unlink()
        except FileNotFoundError:
            pass
    return uniq_txt, uniq_lbl


if __name__ == "__main__":
    DIR = Path.cwd()
    print(DIR)
    raw_base = DIR / "data" / "raw"
    out_base_csv = DIR / "data" / "raw" / "normalized"
    raw_to_normalized_csv(
        raw_base,
        out_base_csv,
    )
    """out_base = DIR / "data" / "processed"
    raw_to_normalized(
        raw_base,
        out_base,
    )

    splits = ("train", "val", "test")
    for split in splits:
        txt_path = out_base / f"{split}.txt"
        lbl_path = out_base / f"{split}.labels"
        uniq_txt, uniq_lbl = remove_duplicates(txt_path, lbl_path)
        uniq_txt.replace(txt_path)
        uniq_lbl.replace(lbl_path)
        print(f"Removed duplicates from {split} split.")
    print("All done.")"""