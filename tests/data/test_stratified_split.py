from datasets import Dataset
from training_pipeline.utils import stratified_subsets

def test_stratified_subsets_balanced():
    labels = [0, 0, 1, 0, 1, 0]
    ds = Dataset.from_dict({
        "text" : [str(i) for i in range(len(labels))],
        "label": labels,
    })
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