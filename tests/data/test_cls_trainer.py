import math
import shutil
import tempfile
from collections import Counter, defaultdict
import numpy as np
import pytest
import torch
from datasets import Dataset
from training_pipeline.arguments import MLMTrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from training_pipeline.cls_trainer import CLSTrainer
from training_pipeline.data_collator import DnsDataCollatorForCLC
from training_pipeline.sampler import GroupWeightedRandomSampler


class DummTokenizer:
    def pad(self, features, return_tensors="pt", pad_to_multiple_of=None):
        max_len = max(len(f["input_ids"]) for f in features)
        ids, attn = [], []
        for f in features:
            pad = [0] * (max_len - len(f["input_ids"]))
            ids.append(f["input_ids"] + pad)
            attn.append([1] * len(f["input_ids"]) + pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("cfg", (), {})()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=torch.float)
        loss = torch.tensor(0.0)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def make_dataset(with_dup_gid: bool = True) -> Dataset:
    data = {
        "input_ids": [[1], [2]],
        "attention_mask": [[1], [1]],
        "label": [0, 1],
    }
    if with_dup_gid:
        data["dup_gid"] = [10, 11]
    return Dataset.from_dict(data)

def make_big_dataset():
    dup_gid = [i // 2 for i in range(20)]
    labels = [i % 2 for i in range(20)]
    data = {
        "input_ids": [[i+1] for i in range(20)],
        "attention_mask": [[1] for _ in range(20)],
        "label": labels,
        "dup_gid": dup_gid,
    }
    return Dataset.from_dict(data)


@pytest.fixture(autouse=True)
def patch_collator_post_init(monkeypatch):
    monkeypatch.setattr(
        DnsDataCollatorForCLC, "__post_init__", lambda self: None
    )


def make_trainer(dataset: Dataset, *, tmp_dir: str, fraction: float = 0.5) -> CLSTrainer:
    tokenizer = DummTokenizer()
    collator = DnsDataCollatorForCLC(tokenizer=tokenizer)
    model = DummyModel()

    args = MLMTrainingArguments(
        output_dir=tmp_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_dir=tmp_dir,
        do_train=False,
        disable_tqdm=True,
        report_to=["none"],
        use_duplicate_weights=True,
        train_fraction=fraction,
        seed=0,
    )

    model.config._dup_weight_map = {
        gid: 1.0 for gid in set(dataset["dup_gid"])
    }

    return CLSTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )


def test_sampler_is_group_weighted(tmp_path):
    ds = make_dataset(with_dup_gid=True)
    trainer = make_trainer(ds, tmp_dir=str(tmp_path))

    loader = trainer.get_train_dataloader()
    assert isinstance(loader.sampler, GroupWeightedRandomSampler)
    #assert isinstance(loader.sampler, GroupWeightedRandomSampler) #TODO: Fix this assertion when GroupWeightedRandomSampler is brought back


def test_sampler_draws_only_valid_indices(tmp_path):
    ds = make_dataset(with_dup_gid=True)
    trainer = make_trainer(ds, tmp_dir=str(tmp_path))

    loader = trainer.get_train_dataloader()
    indices = list(iter(loader.sampler))

    assert all(0 <= idx < len(ds) for idx in indices)

    gid_for_idx = {0: 10, 1: 11}
    gid_counts = Counter(gid_for_idx[idx] for idx in indices)
    assert gid_counts[10] > 0 and gid_counts[11] > 0


def test_evaluate_saves_logits_and_metrics(tmp_path):
    ds = make_dataset(with_dup_gid=True)
    trainer = make_trainer(ds, tmp_dir=str(tmp_path))

    metrics = trainer.evaluate(eval_dataset=ds)
    assert "eval_accuracy" in metrics
    assert trainer._last_eval_logits is not None
    assert trainer._last_eval_labels is not None


def test_logits_and_labels_values(tmp_path):
    ds = make_dataset(with_dup_gid=True)
    trainer = make_trainer(ds, tmp_dir=str(tmp_path))

    trainer.evaluate(eval_dataset=ds)

    logits = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    np.testing.assert_allclose(
        np.array(trainer._last_eval_logits, dtype=np.float32),
        logits,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        np.array(trainer._last_eval_labels, dtype=np.int64), labels
    )

@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_train_fraction_subsampling(tmp_path, fraction):
    ds = make_big_dataset()
    trainer = make_trainer(ds, tmp_dir=str(tmp_path), fraction=fraction)

    loader = trainer.get_train_dataloader()

    n_pos, n_neg = Counter(ds['label'])[1], Counter(ds['label'])[0]
    exp_rows = round(n_pos * fraction) + round(n_neg * fraction)
    assert len(loader.dataset) == exp_rows

    new_counts = Counter(loader.dataset['label'])
    assert new_counts[0] == round(n_neg * fraction)
    assert new_counts[1] == round(n_pos * fraction)

    sampled_idx = list(iter(loader.sampler))
    assert all (0 <= i < len(trainer.train_dataset) for i in sampled_idx)