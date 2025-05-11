from training_pipeline.masker import MaskSampler
import pytest
import torch

@pytest.fixture
def dummy_inputs():
    batch, seq = 16, 32
    input_ids = torch.arange(batch * seq).view(batch, seq)
    attn = torch.ones((batch, seq), dtype=torch.bool)
    special = torch.zeros((batch, seq), dtype=torch.bool)
    special[:, 0] = True
    return input_ids, attn, special

@pytest.mark.parametrize("strategy", ["token", "span"])
def test_shape_and_specials(dummy_inputs, strategy):
    input_ids, attn, special = dummy_inputs
    sampler = MaskSampler(
        mlm_probability=0.2,
        span_lambda=3.0,
        strategy=strategy,
        seed=42,
    )
    mask = sampler(input_ids, attn, special)
    assert (mask[:, 0] == False).all()
    attn[:, 5] = False
    mask2 = sampler(input_ids, attn, special)
    assert (mask2[:, 5] == False).all()

def test_token_prob(dummy_inputs):
    input_ids, attn, special = dummy_inputs
    p = 0.8
    sampler = MaskSampler(
        mlm_probability=p,
        span_lambda=3.0,
        strategy="token",
        seed=0,
    )
    mask = sampler(input_ids, attn, special)
    frac = mask.sum().item() / mask.numel()
    assert abs(frac - p) < 0.05, f"Expected {p}, got {frac}"

def test_randomness(dummy_inputs):
    input_ids, attn, special = dummy_inputs
    sampler = MaskSampler(
        mlm_probability=0.5,
        span_lambda=3.0,
        strategy="token",
        seed=7,
    )
    sampler.set_epoch(1)
    m1 = sampler(input_ids, attn, special)
    sampler.set_epoch(1)
    m2 = sampler(input_ids, attn, special)
    assert torch.equal(m1, m2)
    sampler.set_epoch(2)
    m3 = sampler(input_ids, attn, special)
    assert not torch.equal(m1, m3)
    
def test_span_one_per_row(dummy_inputs):
    input_ids, attn, special = dummy_inputs
    sampler = MaskSampler(
        mlm_probability=1e-6,
        span_lambda=1e9,
        strategy="span",
        seed=0,
    ) 
    mask = sampler(input_ids, attn, special)
    counts = mask.sum(dim=1)
    assert (counts >= 1).all()
