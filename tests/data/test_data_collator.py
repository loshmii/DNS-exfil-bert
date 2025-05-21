import torch
import pytest

from training_pipeline.data_collator import DnsDataCollatorForMLM

@pytest.fixture(autouse=True)
def skip_post_init(monkeypatch):
    monkeypatch.setattr(
        DnsDataCollatorForMLM, 
        "__post_init__", 
        lambda self: None
    )

class DummyTokenizer:
    def __init__(self):
        self.mask_token_id = 1
        self.pad_token_id = 0
        self.vocab_size = 256
        self.special_tokens_map = {"mask_token": self.mask_token_id, "pad_token": self.pad_token_id}

    def pad(self, features, return_tensors="pt", pad_to_multiple_of=None):
        lengths = [len(f["input_ids"]) for f in features]
        max_length = max(lengths)
        if pad_to_multiple_of:
            rem = max_length % pad_to_multiple_of
            if rem:
                max_length += pad_to_multiple_of - rem

        batch_input_ids = []
        batch_attention_mask = []
        batch_special_tokens_mask = []
        for f in features:
            ids = f["input_ids"].copy()
            attention = [1] * len(ids)
            pad_len = max_length - len(ids)
            ids.extend([self.pad_token_id] * pad_len)
            attention.extend([0] * pad_len)
            batch_input_ids.append(ids)
            batch_attention_mask.append(attention)
            batch_special_tokens_mask.append([0] * len(ids))

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "special_tokens_mask": torch.tensor(batch_special_tokens_mask, dtype=torch.long),
        }

class DummyMaskSampler:
    def __init__(self, mask_positions):
        self.mask_positions = mask_positions

    def __call__(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for batch_idx, seq_idx in self.mask_positions:
            if batch_idx < mask.size(0) and seq_idx < mask.size(1):
                mask[batch_idx, seq_idx] = True
        return mask


def test_shaped_and_padding_and_types():
    features = [
        {"input_ids": [1, 2, 3, 4, 5]},
        {"input_ids": [6, 7, 8]},
    ]
    pad_to_multiple_of = 4
    tokenizer = DummyTokenizer()
    sampler = DummyMaskSampler([])
    collator = DnsDataCollatorForMLM(
        tokenizer,
        sampler,
        pad_to_multiple_of=pad_to_multiple_of,
        mask_token_prob=0.0,
        random_token_prob=0.0,
    )

    batch = collator(features)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    assert input_ids.ndim == 2
    bs, seq_len = input_ids.shape
    assert bs == 2
    assert seq_len % pad_to_multiple_of == 0
    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long

    pad_positions = (input_ids == tokenizer.pad_token_id)
    non_pad_positions = ~pad_positions
    assert torch.all(attention_mask[pad_positions] == 0)
    assert torch.all(attention_mask[non_pad_positions] == 1)


def test_masking_and_label_assignment():
    features = [
        {"input_ids": [10, 20, 30]},
        {"input_ids": [40, 50, 60, 70]},
    ]
    pad_to_multiple_of = 2
    tokenizer = DummyTokenizer()
    mask_positions = [(0, 0), (1, 3)]
    sampler = DummyMaskSampler(mask_positions)
    collator = DnsDataCollatorForMLM(
        tokenizer,
        sampler,
        pad_to_multiple_of=pad_to_multiple_of,
        mask_token_prob=1.0,
        random_token_prob=0.0,
    )

    batch = collator(features)
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert input_ids.shape == labels.shape

    for b, s in mask_positions:
        assert input_ids[b, s].item() == tokenizer.mask_token_id
        assert labels[b, s].item() == features[b]["input_ids"][s]

    pad_positions = (input_ids == tokenizer.pad_token_id)
    applied_mask = sampler(input_ids)
    non_pad_non_mask = (~pad_positions) & (~applied_mask)
    assert torch.all(labels[non_pad_non_mask] == -100)

    assert torch.all(labels[pad_positions] == -100)

def test_mask_random_untouched_distribution():
    torch.manual_seed(0)
    batch_size = 32
    seq_len = 50
    N = batch_size * seq_len
    p_mask = 0.2
    p_random = 0.1

    tokenizer = DummyTokenizer()
    tokenizer.vocab_size = 10

    original_id = 100
    features = [{"input_ids": [original_id] * seq_len} for _ in range(batch_size)]

    class FullMaskSampler:
        def __call__(self, input_ids, **kwargs):
            return torch.ones_like(input_ids, dtype=torch.bool)

    sampler = FullMaskSampler()

    collator = DnsDataCollatorForMLM(
        tokenizer,
        sampler,
        pad_to_multiple_of=1,
        mask_token_prob=p_mask,
        random_token_prob=p_random,
    )

    batch = collator(features)
    input_ids = batch["input_ids"]

    mask_count = (input_ids == tokenizer.mask_token_id).sum().item()
    random_count = ((input_ids != tokenizer.mask_token_id) & (input_ids < tokenizer.vocab_size)).sum().item()
    untouched_count = (input_ids == original_id).sum().item()
    assert mask_count + random_count + untouched_count == N

    mask_ratio = mask_count / N
    random_ratio = random_count / N
    untouched_ratio = untouched_count / N

    import math
    tolerance_mask = 3 * math.sqrt(p_mask * (1 - p_mask) / N)
    tolerance_random = 3 * math.sqrt(p_random * (1 - p_random) / N)
    p_untouched = 1 - p_mask - p_random
    tolerance_untouched = 3 * math.sqrt(p_untouched * (1 - p_untouched) / N)

    assert abs(mask_ratio - p_mask) <= tolerance_mask
    assert abs(random_ratio - p_random) <= tolerance_random
    assert abs(untouched_ratio - p_untouched) <= tolerance_untouched
