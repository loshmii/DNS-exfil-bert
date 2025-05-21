import torch
import pytest

from training_pipeline.trainer import MaskingCallback, PerplexityCallback
from transformers import TrainerControl
from training_pipeline.masker import MaskSampler


class DummyState:
    def __init__(self, epoch=None):
        self.epoch = epoch


class DummySampler:
    def __init__(self):
        self.epoch_set = None

    def set_epoch(self, epoch):
        self.epoch_set = epoch


class DummyCollator:
    def __init__(self, sampler):
        self.mask_sampler = sampler


def test_masking_callback_epoch_advances():
    sampler = DummySampler()
    collator = DummyCollator(sampler)
    callback = MaskingCallback(collator)
    state = DummyState(epoch=0)
    control = TrainerControl()

    returned_control = callback.on_epoch_begin(
        args=None, state=state, control=control
    )

    assert returned_control is control
    assert sampler.epoch_set == 0


def test_perplexity_callback_calculates_perplexity():
    callback = PerplexityCallback()
    control = TrainerControl()
    loss_ln2 = torch.log(torch.tensor(2.0)).item()
    logs = {"loss": loss_ln2, "eval_loss": loss_ln2}
    state = DummyState()

    returned_control = callback.on_log(
        args=None, state=state, control=control, logs=logs
    )

    assert "perplexity" in logs
    assert "eval_perplexity" in logs
    assert logs["perplexity"] == pytest.approx(2.0)
    assert logs["eval_perplexity"] == pytest.approx(2.0)
    assert returned_control is control


def test_masksampler_varies_across_epochs():
    seed = 123
    batch_size = 16
    seq_len = 32
    token_sampler = MaskSampler(
        mlm_probability=0.3,
        strategy="token",
        seed=seed,
    )
    span_sampler = MaskSampler(
        mlm_probability=0.3,
        strategy="span",
        span_lambda=3.0,
        seed=seed,
    )

    input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    special_tokens_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    token_sampler.set_epoch(0)
    span_sampler.set_epoch(0)

    token_mask1 = token_sampler(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
    )
    span_mask1 = span_sampler(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
    )

    token_sampler.set_epoch(1)
    span_sampler.set_epoch(1)

    token_mask2 = token_sampler(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
    )
    span_mask2 = span_sampler(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
    )

    assert torch.any(
        token_mask1 != token_mask2
    ), "Token masks DO NOT vary across epochs"
    assert torch.any(
        span_mask1 != span_mask2
    ), "Span masks DO NOT vary across epochs"
