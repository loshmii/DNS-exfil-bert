import numpy as np
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState
from typing import List, Any
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd


def dedup_dataset_keep_first(
    dataset: Dataset,
    dedup_column: str,
):
    tbl = dataset.data.table

    row_ids = pa.array(range(tbl.num_rows), type=pa.int64())

    tbl = tbl.append_column("row_id", row_ids)
    tbl = tbl.group_by(dedup_column, use_threads=False).aggregate(
        [("row_id", "hash_min")]
    )

    rows = tbl["row_id_min"].to_pylist()

    return dataset.select(rows)


def stratified_subsets(
    dataset: Dataset,
    num_subsets: int,
    label_key: str = "label",
    seed: int = 42,
) -> List[Dataset]:
    if num_subsets < 1:
        raise ValueError("Number of subsets must be at least 1.")

    labels = np.array(dataset[label_key])
    unique = np.unique(labels)
    rng = np.random.default_rng(seed)
    idx_by_label = {lbl: np.where(labels == lbl)[0] for lbl in unique}
    for idx in idx_by_label.values():
        rng.shuffle(idx)

    splits = [[] for _ in range(num_subsets)]
    for lbl, indices in idx_by_label.items():
        q, r = divmod(len(indices), num_subsets)
        start = 0
        for i in range(num_subsets):
            end = start + q + (1 if i < r else 0)
            splits[i].extend(indices[start:end])
            start = end

    subsets = []
    for idx in splits:
        rng.shuffle(idx)
        subsets.append(dataset.select(idx))
    return subsets

def stratified_subsamples(
    dataset: Dataset,
    fraction: float =  0.1,
    label_key: str = "label",
    seed: int = 42,
):
    tbl = dataset.data.table
    tbl = tbl.append_column(
        "row_id", pa.array(np.arange(tbl.num_rows, dtype=np.int64))
    )

    grouped = tbl.group_by(["dup_gid", label_key]).aggregate(
        [("row_id", "list"), (label_key, "count")])
    
    meta = grouped.to_pandas()
    
    total_pos = meta.loc[meta[label_key] == 1, f"{label_key}_count"].sum()
    total_neg = meta.loc[meta[label_key] == 0, f"{label_key}_count"].sum()
    target = {
        0 : max(1,int(round(total_neg * fraction))),
        1 : max(1,int(round(total_pos * fraction)))
    }

    rng = np.random.default_rng(seed)
    keep_row_ids = []
    meta = meta.sample(frac=1, random_state=seed)
    consumed = {0: 0, 1: 0}

    for _, row in meta.iterrows():
        lbl = row[label_key]
        if consumed[lbl] >= target[lbl]:
            continue
        need = min(target[lbl] - consumed[lbl], len(row["row_id_list"]))
        chosen = rng.choice(
            row["row_id_list"], size = need, replace=False
        )
        keep_row_ids.extend(chosen)
        consumed[lbl] += need
        if all(consumed[l] >= target[l] for l in target):
            break

    return dataset.select(keep_row_ids).shuffle(seed=seed)
    

class EvalSubsetCallback(TrainerCallback):
    def __init__(self, trainer, subsets: List[Dataset]):
        self.trainer = trainer
        self.subsets = subsets
        self._idx = 0

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict,
        **kwargs: Any,
    ) -> TrainerControl:
        prefix = list(metrics.keys())[0].split("_")[0] if metrics else ""
        if prefix != "test":
            self._idx = (self._idx + 1) % len(self.subsets)
            self.trainer.eval_dataset = self.subsets[self._idx]
        return control

if __name__ == "__main__":
    data = {
        "text": ["a", "b", "c", "a", "b", "c"],
        "labels": [0, 1, 0, 1, 0, 1],
        "dup_gid": [1, 1, 2, 1, 1, 2]
    }
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    dataset = stratified_subsamples(dataset, fraction=1.0, label_key='labels', seed=42)
    print(dataset['labels'])