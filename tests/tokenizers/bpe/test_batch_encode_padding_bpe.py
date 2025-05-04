import pytest
from data_pipeline.tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer
from hydra import initialize, compose
from omegaconf import DictConfig


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize(
        config_path="../../../configs", job_name="test", version_base="1.3"
    ):
        cfg = compose(config_name="config", overrides=["tokenizer=bpe8k"])
    return cfg


def test_tok_build_bpe(cfg: DictConfig):
    tok = BpeTokenizer.from_pretrained(path=cfg.training.tokenizer.save_dir)
    assert isinstance(tok, BpeTokenizer)


def test_padding_and_attention_mask_bpe(cfg: DictConfig):
    input_strings = [
        "a",
        "ab",
        "abcde",
        "f" * (cfg.tokenizer.max_length + 5),
    ]
    tokenizer = BpeTokenizer.from_pretrained(
        path=cfg.training.tokenizer.save_dir
    )
    batch = tokenizer(
        input_strings,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
    )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    st_mask = batch["special_tokens_mask"]

    lengths = [len(seq) for seq in input_ids]
    st_set = tokenizer.all_special_ids

    assert len(set(lengths)) == 1
    assert lengths[0] <= cfg.tokenizer.max_length

    for seq, mask, sts in zip(input_ids, attention_mask, st_mask):
        for tok, m, st in zip(seq, mask, sts):
            if tok == tokenizer.pad_token_id:
                assert m == 0
                assert st == 1
            elif tok in st_set:
                assert m == 1
                assert st == 1
            else:
                assert m == 1
                assert st == 0
