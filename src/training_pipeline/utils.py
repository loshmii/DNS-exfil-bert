import numpy as np
from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState
from typing import List, Any

def stratified_subsets(
        dataset: Dataset,
        num_subsets: int,
        label_key: str = "label",
        seed: int = 42
)-> List[Dataset]:
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

class EvalSubsetCallback(TrainerCallback):
    def __init__(self, trainer, subsets: List[Dataset]):
        self.trainer = trainer
        self.subsets = subsets
        self._idx = 0

    def on_step_end(
        self, 
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any
    ):
        if control.should_evaluate and self.subsets:
            self.trainer.eval_dataset = self.subsets[self._idx]
            self._idx = (self._idx + 1) % len(self.subsets)
        return control