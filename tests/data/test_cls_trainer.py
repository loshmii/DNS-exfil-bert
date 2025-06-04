import pytest
import torch
from datasets import Dataset
from torch.utils.data import WeightedRandomSampler
from training_pipeline.cls_trainer import CLSTrainer
from training_pipeline.data_collator import DnsDataCollatorForCLC
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np

class DummyTokenizer:
    def pad(self, features, return_tensors="pt", pad_to_multiple_of=None):
        max_len = max(len(f["input_ids"]) for f in features)
        batch_ids = []
        attn = []
        for f in features:
            ids = f["input_ids"] + [0] * (max_len - len(f["input_ids"]))
            batch_ids.append(ids)
            attn.append([1] * len(f["input_ids"]) + [0] * (max_len - len(f["input_ids"])))
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
    
@pytest.fixture(autouse=True)
def skip_post_init(monkeypatch):
    monkeypatch.setattr(DnsDataCollatorForCLC, "__post_init__", lambda x: None)

def make_dataset(with_weights=True):
    data = {
        "input_ids": [[1], [2]],
        "attention_mask": [[1], [1]],
        "labels": [0, 1],
    }
    if with_weights:
        data["sample_weight"] = [1.0, 2.0]
    return Dataset.from_dict(data)

class DummyModel(torch.nn.Module):
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]], dtype=torch.float)
        loss = torch.tensor(0.0)
        return SequenceClassifierOutput(
            loss = loss,
            logits=logits,
        )
    
def test_sampler_created_when_weights_provided():
    tokenizer = DummyTokenizer()
    ds = make_dataset(with_weights=True)
    collator = DnsDataCollatorForCLC(tokenizer=tokenizer)
    trainer = CLSTrainer(
        model=DummyModel(),
        args = None,
        train_dataset=ds,
        data_collator=collator,
    )
    loader = trainer.get_train_dataloader()
    assert isinstance(loader.sampler, WeightedRandomSampler)

def test_metrics_and_logits_saved():
    tokenizer = DummyTokenizer()
    ds = make_dataset(with_weights=True)
    collator = DnsDataCollatorForCLC(tokenizer=tokenizer)
    trainer = CLSTrainer(
        model=DummyModel(),
        args=None,
        train_dataset=ds,
        eval_dataset=ds,
        data_collator=collator,
    )
    metrics = trainer.evaluate()
    assert "eval_accuracy" in metrics
    assert trainer._last_eval_logits is not None
    assert trainer._last_eval_labels is not None

def test_logits_labels_and_metrics_are_correct(): #TODO : Revert collator post init
    tokenizer = DummyTokenizer()
    ds = make_dataset(with_weights=True)
    collator = DnsDataCollatorForCLC(tokenizer=tokenizer)
    
    trainer = CLSTrainer(
        model=DummyModel(),
        args=None,
        train_dataset=ds,
        eval_dataset=ds,
        data_collator=collator,
    )

    metrics = trainer.evaluate()

    expected_logits = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
    actual_logits = trainer._last_eval_logits
    assert isinstance(actual_logits, np.ndarray) or isinstance(actual_logits, torch.Tensor)
    actual_logits = np.array(actual_logits, dtype=np.float32)
    assert np.array_equal(actual_logits, expected_logits)

    actual_labels = trainer._last_eval_labels
    assert isinstance(actual_labels, np.ndarray) or isinstance(actual_labels, torch.Tensor)
    actual_labels = np.array(actual_labels, dtype=np.int64)
    expected_labels = np.array([0, 1], dtype=np.int64)
    assert np.array_equal(actual_logits, expected_logits)