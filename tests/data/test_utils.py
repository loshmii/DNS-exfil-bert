from datasets import Dataset
import pytest
import numpy as np
from training_pipeline.utils import (
    stratified_subsets,
    dedup_dataset_keep_first,
    stratified_subsamples,
)


def test_stratified_subsets_balanced():
    labels = [0, 0, 1, 0, 1, 0]
    ds = Dataset.from_dict(
        {
            "text": [str(i) for i in range(len(labels))],
            "label": labels,
        }
    )
    subsets = stratified_subsets(ds, num_subsets=2, label_key="label", seed=0)
    assert len(subsets) == 2
    total_pos = sum(labels)
    total_ratio = total_pos / len(labels)
    total_len = 0
    for sub in subsets:
        total_len += len(sub)
        ratio = sum(sub["label"]) / len(sub)
        assert abs(ratio - total_ratio) < 1e-6
    assert total_len == len(ds)


def test_dedup_dataset_keep_first():
    labels = [0, 0, 1, 1, 0, 1, 1]
    text = ["a", "b", "c", "d", "e", "f", "g"]
    dup_gids = [10, 10, 11, 12, 12, 13, 13]
    ds = Dataset.from_dict(
        {
            "text": text,
            "label": labels,
            "dup_gid": dup_gids,
        }
    )
    deduped_ds = dedup_dataset_keep_first(ds, "dup_gid")

    assert len(deduped_ds) == 4
    assert deduped_ds["dup_gid"] == [10, 11, 12, 13]
    assert deduped_ds["text"] == ["a", "c", "d", "f"]

def _make_dummy_ds():
    n_dup = 10
    repeats = 3
    dup_gids = np.repeat(np.arange(n_dup), repeats)
    labels = np.tile([0,0,1], n_dup)
    text = [f"row{i}" for i in range(len(labels))]
    return Dataset.from_dict(
        {
            "text": text,
            "label": labels.tolist(),
            "dup_gid": dup_gids.tolist(),
        }
    )

@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5])
def test_fraction_and_label_targets(fraction):
    ds = _make_dummy_ds()
    samples = stratified_subsamples(ds, fraction=fraction, seed=0)

    total_pos = int(round(fraction * ds["label"].count(1)))
    total_neg = int(round(fraction * ds["label"].count(0)))
    assert len(samples) == total_pos + total_neg
    assert samples["label"].count(1) == total_pos
    assert samples["label"].count(0) == total_neg

def test_unique_rows():
    ds = _make_dummy_ds()
    samples = stratified_subsamples(ds, fraction=0.25, seed=0)
    assert len(set(samples["text"])) == len(samples)

def test_seed():
    ds = _make_dummy_ds()

    s1 = stratified_subsamples(ds, fraction=0.25, seed=0)
    s2 = stratified_subsamples(ds, fraction=0.25, seed=0)
    s3 = stratified_subsamples(ds, fraction=0.25, seed=1)

    assert s1['text'] == s2['text']
    assert set(s1['text']) != set(s3['text'])