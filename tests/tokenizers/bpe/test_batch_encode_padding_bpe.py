import pytest
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)
from hydra import initialize_config_dir, compose
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

BASE = Path(__file__).parent.parent.parent.parent.resolve()


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize_config_dir(
        config_dir=str(BASE / "configs"),
        job_name="test",
        version_base="1.3",
    ):
        cfg = compose(
            config_name="bpe_test_config",
            overrides=["tokenizer=bpe8k_pretrained", f"+paths.root={BASE}"],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)
    return cfg


def test_tok_build_bpe(cfg: DictConfig):
    tokenizer = BpeTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True),
    )
    assert isinstance(tokenizer, BpeTokenizer)


def test_padding_and_attention_mask_bpe(cfg: DictConfig):
    tokenizer = BpeTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True),
    )
    input_strings = [
        "a",
        "ab",
        "abcde",
        "f" * (tokenizer.model_max_length + 5),
    ]
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
    assert lengths[0] <= tokenizer.model_max_length

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
