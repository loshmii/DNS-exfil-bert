from pathlib import Path
from typing import List, IO, Iterator, Tuple, Sequence, ClassVar, Union
import time
from tqdm import tqdm
import csv
import re
import idna
import zlib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

tqdm.pandas(desc="Processing", unit="rows", colour="green")


def load_all(
    base_path: Union[str, Path],
    splits: Tuple[str, ...] = ("train", "val", "test"),
):
    base_path = Path(base_path)
    dfs = []
    for split in splits:
        fp = base_path / f"{split}.csv"
        df = pd.read_csv(
            fp,
            dtype={
                "text": str,
                "label": int,
                "ok": bool,
                "reason": int,
                "dup_gid": "uint32",
            },
        )
        dfs.append(df)
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df[
        full_df["text"].notna() & full_df["text"].str.strip().ne("")
    ].reset_index(drop=True)
    return full_df


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
    LABEL_CHARS = re.compile(r"^[a-z0-9\*_?@()\[\]\\-]+$")
    DOTS = re.compile(r"\.{2,}")

    @classmethod
    def is_valid(cls, domain: str) -> Tuple[bool, int]:
        reasons = 0
        if len(domain) > 253:
            reasons |= 0x01  # Too long
        if any(ord(ch) < 32 for ch in domain):
            reasons |= 0x02  # Control characters
        if DomainValidator.DOTS.search(domain):
            reasons |= 0x04  # Consecutive dots
        labels = domain.split(".")
        for lbl in labels:
            if lbl == "":
                reasons |= 0x08  # Empty label
                break
            if lbl == "*":
                continue
            if not (lbl[0].isalnum() and lbl[-1].isalnum()):
                reasons |= (
                    0x10  # Label does not start and end with alphanumeric
                )
            if not DomainValidator.LABEL_CHARS.match(lbl):
                reasons |= 0x20  # Invalid characters in label

        ok = reasons == 0
        return ok, reasons


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

    DELIM = re.compile(r"[`[\x00-\x1F]")

    IPV4_FULL = re.compile(
        r"^(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}$"
    )

    DOTS = re.compile(r"\.{2,}")
    PROTO = re.compile(r"^(?:https?|ftp)://", re.I)
    USER = re.compile(r"^[^@]+@", re.I)
    PORT = re.compile(r":\d+$")

    @staticmethod
    def normalize(raw: str, *, keep_puncycode: bool = True) -> tuple[str, str]:
        d = raw.strip()

        d = d.replace(r"\(", "(").replace(r"\)", ")")

        if d.startswith("[") and "]" not in d:
            d = d.lstrip("[")
        d = d.rstrip("]")

        d = DomainNormalizer.PROTO.sub("", d)
        d = DomainNormalizer.USER.sub("", d)
        d = d.split("/", 1)[0]

        if d.startswith("[") and "]" not in d:
            d = d.lstrip("[")
        d = d.rstrip("]")

        d = DomainNormalizer.PORT.sub("", d)
        d = d.rstrip(".")
        d = DomainNormalizer.DOTS.sub(".", d).lower()

        if DomainNormalizer.IPV4_FULL.match(d):
            return (d, "")

        try:
            uidomain = idna.decode(d, strict=False)
        except idna.IDNAError:
            uidomain = d

        try:
            asidomain = idna.encode(uidomain).decode("ascii")
        except idna.IDNAError:
            asidomain = d

        return (asidomain, uidomain if keep_puncycode else "")


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
        w.writerow(["text", "label", "ok", "reason", "dup_gid"])
        csv_files[split] = f
        writers[split] = w

    try:
        for split in splits:
            total = iterator.count_dmn_in_splt(split)
            seen = 0
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

                clean_ascii, _ = DomainNormalizer.normalize(
                    raw_dom, keep_puncycode=True
                )
                is_valid, reason = DomainValidator.is_valid(clean_ascii)
                gid = zlib.crc32(clean_ascii.encode("utf-8"))

                if clean_ascii == "":
                    continue

                writers[split].writerow(
                    [clean_ascii, str(label), is_valid, reason, gid]
                )

            elapsed = time.perf_counter() - start
            rate = seen / elapsed if elapsed > 0 else 0.0
            print(
                f"[{split:5s}] written {seen}"
                f"in {elapsed: .1f}s ({rate :.1f} dom/s)"
            )
    finally:
        for f in csv_files.values():
            f.close()


def raw_to_normalized_csv_one_file(
    raw_base: Union[str, Path],
    out_base: Union[str, Path],
    prefix: str = "all",
):
    raw_base = Path(raw_base)
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw data from {raw_base}")
    df = pd.read_csv(
        raw_base,
        dtype={
            "Subdomain": str,
            "Exfiltration": int,
        },
    )

    df_copy = df.copy()
    df_copy["text"] = df_copy["Subdomain"].progress_apply(
        lambda x: DomainNormalizer.normalize(x, keep_puncycode=True)[0]
    )
    df_copy["ok"] = df_copy["Subdomain"].progress_apply(
        lambda x: DomainValidator.is_valid(x)[0]
    )
    df_copy["reason"] = df_copy["Subdomain"].progress_apply(
        lambda x: DomainValidator.is_valid(x)[1]
    )
    df_copy["dup_gid"] = df_copy["Subdomain"].progress_apply(
        lambda x: zlib.crc32(x.encode("utf-8"))
    )
    print("Converting Exfiltration to label")
    df_copy["label"] = df_copy["Exfiltration"].astype(int)

    print("Cleaning up")
    df_copy = df_copy[["text", "label", "ok", "reason", "dup_gid"]]
    df_copy = df_copy[
        df_copy["text"].notna() & df_copy["text"].str.strip().ne("")
    ]

    print(f"Writing to {out_base / 'all.csv'}")
    df_copy.to_csv(
        out_base / f"{prefix}.csv",
        index=False,
        encoding="utf-8",
    )


def build_mlm_csvs(
    raw_base: Union[str, Path],
    out_base: Union[str, Path],
    splits: Tuple[str, ...] = ("train", "val", "test"),
    proportions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    full_df = load_all(raw_base, splits)
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    mlm_candidates = full_df[full_df["ok"] == True].copy()
    mlm_deduped = mlm_candidates.drop_duplicates(
        subset="dup_gid", keep="first"
    )

    out_base = out_base / "mlm"
    out_base.mkdir(parents=True, exist_ok=True)

    texts = mlm_deduped[["text"]].copy()
    texts = texts.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(texts)
    n_train = int(n * proportions[0])
    n_val = int(n * proportions[1])

    train_df = texts.iloc[:n_train]
    val_df = texts.iloc[n_train : n_train + n_val]
    test_df = texts.iloc[n_train + n_val :]

    for split_name, df_split in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        out_fp = out_base / f"{split_name}.csv"
        df_split.to_csv(out_fp, index=False)
        print(f"MLM: wrote {len(df_split)} rows to {out_fp}")


def build_mlm_csvs_one_file(
    raw_base: Union[str, Path],
    out_base: Union[str, Path],
    splits: Tuple[str, ...] = ("train", "val", "test"),
    proportions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    raw_base = Path(raw_base)
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    mlm_candidates = pd.read_csv(
        raw_base,
        dtype={
            "text": str,
            "label": int,
            "ok": bool,
            "reason": int,
            "dup_gid": "uint32",
        },
    )
    mlm_candidates = mlm_candidates[mlm_candidates["ok"] == True].copy()
    mlm_deduped = mlm_candidates.drop_duplicates(
        subset="dup_gid", keep="first"
    )

    out_base = out_base / "mlm"
    out_base.mkdir(parents=True, exist_ok=True)

    texts = mlm_deduped[["text"]].copy()
    texts = texts.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n = len(texts)
    n_train = int(n * proportions[0])
    n_val = int(n * proportions[1])
    n_test = n - n_train - n_val

    train_df = texts.iloc[:n_train]
    val_df = texts.iloc[n_train : n_train + n_val]
    test_df = texts.iloc[n_train + n_val : n_train + n_val + n_test]

    for split_name, df_split in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        out_fp = out_base / f"{split_name}.csv"
        df_split.to_csv(out_fp, index=False)
        print(f"MLM: wrote {len(df_split)} rows to {out_fp}")


def build_cls_csvs(
    raw_base: Union[str, Path],
    out_base: Union[str, Path],
    splits: Tuple[str, ...] = ("train", "val", "test"),
    proportions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    full_df = load_all(raw_base, splits)
    out_base = Path(out_base) / "cls"
    out_base.mkdir(parents=True, exist_ok=True)

    df = full_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    x = np.zeros(len(df), dtype=int)
    y = df["label"].to_numpy()
    groups = df["dup_gid"].to_numpy()

    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(sgkf1.split(x, y, groups))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    y_temp = temp_df["label"].to_numpy()
    groups_temp = temp_df["dup_gid"].to_numpy()
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_subidx, test_subidx = next(
        sgkf2.split(np.zeros(len(temp_df)), y_temp, groups_temp)
    )
    val_df = temp_df.iloc[val_subidx]
    test_df = temp_df.iloc[test_subidx]

    for split_name, df_split in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        out_fp = out_base / f"{split_name}.csv"
        df_split.to_csv(out_fp, index=False)
        print(f"CLS: wrote {len(df_split)} rows to {out_fp}")


def build_cls_csvs_one_file(
    raw_base: Union[str, Path],
    out_base: Union[str, Path],
    splits: Tuple[str, ...] = ("train", "val", "test"),
    proportions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    raw_base = Path(raw_base)
    out_base = Path(out_base) / "cls"
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw data from {raw_base}")
    df = pd.read_csv(
        raw_base,
        dtype={
            "text": str,
            "label": int,
            "ok": bool,
            "reason": int,
            "dup_gid": "uint32",
        },
    )

    print("Splitting and shuffling data")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    x = np.zeros(len(df), dtype=int)
    y = df["label"].to_numpy()
    groups = df["dup_gid"].to_numpy()

    sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, temp_idx = next(sgkf1.split(x, y, groups))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    y_temp = temp_df["label"].to_numpy()
    groups_temp = temp_df["dup_gid"].to_numpy()
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_subidx, test_subidx = next(
        sgkf2.split(np.zeros(len(temp_df)), y_temp, groups_temp)
    )
    val_df = temp_df.iloc[val_subidx]
    test_df = temp_df.iloc[test_subidx]

    print("Writing to CSV files")
    for split_name, df_split in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        out_fp = out_base / f"{split_name}.csv"
        df_split.to_csv(out_fp, index=False)
        print(f"CLS: wrote {len(df_split)} rows to {out_fp}")


def remove_duplicates(
    split_csv: Union[str, Path],
    out_dir: Union[str, Path] = None,
) -> Path:
    split_csv = Path(split_csv)
    out_dir = Path(out_dir) if out_dir else split_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    uniq_csv = out_dir / f"{split_csv.stem}.csv"

    seen = set()

    with split_csv.open("r", encoding="utf-8", newline="") as f_in:
        total = sum(1 for _ in f_in) - 1

    with split_csv.open(
        "r", encoding="utf-8", newline=""
    ) as rf, uniq_csv.open("w", encoding="utf-8", newline="") as wf:

        reader = csv.reader(rf)
        writer = csv.writer(wf)

        header = next(reader, None)
        if header:
            writer.writerow(header)

        for row in tqdm(
            reader,
            total=total,
            desc="removing duplicates",
            unit="rows",
            colour="blue",
        ):
            if not row:
                continue

            dom = row[0].strip()
            if dom not in seen:
                seen.add(dom)
                writer.writerow(row)

    return uniq_csv


# TODO: Change the logic of the file so the inner functions work on files and not on the whole directory then have outer logic to call the inner functions


if __name__ == "__main__":
    DIR = Path(__file__).parent.parent.parent.resolve()

    """raw_base = DIR / "data" / "raw"
    out_base_csv = DIR / "data" / "processed"
    raw_to_normalized_csv(
        raw_base,
        out_base_csv / "original",
    )

    build_mlm_csvs(
        out_base_csv / "original",
        out_base_csv,
    )
    build_cls_csvs(
        out_base_csv / "original",
        out_base_csv,
    )"""
    RAW_DIR = Path("/home/milos.tomic.etf/Downloads/data/dup_capped_data_wo_antiv.csv")
    OUT_DIR = Path("/home/milos.tomic.etf/Downloads/data/new_processed")
    MLM_DIR = Path(
        "/home/milos.tomic.etf/projects/DNS-exfil-bert/data/processed_antiv"
    )
    CLS_DIR = Path(
        "/home/milos.tomic.etf/projects/DNS-exfil-bert/data/processed_antiv"
    )
    """raw_to_normalized_csv_one_file(
        RAW_DIR,
        OUT_DIR,
        prefix="wo_antiv",
    )"""
    build_mlm_csvs_one_file(
        OUT_DIR / "wo_antiv.csv",
        MLM_DIR,
    )
    build_cls_csvs_one_file(OUT_DIR / "wo_antiv.csv", CLS_DIR, seed=0)
