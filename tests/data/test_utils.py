from datasets import Dataset
from training_pipeline.utils import (
    stratified_subsets,
    dedup_dataset_keep_first,
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
