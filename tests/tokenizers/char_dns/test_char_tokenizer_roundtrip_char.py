import hypothesis.strategies as st
from hypothesis import given
import pytest
from pathlib import Path

from data_pipeline.tokenizers.char_dns.v0_1.char_tokenizer import CharTokenizer
from data_pipeline.tokenizers.char_dns.v0_1.config.CharTokConfig import (
    get_config_for_char_tok,
)

domain_strat = st.from_regex(r"(?:[a-z0-9\-]+\.)+[a-z]{2,}", fullmatch=True)


@pytest.fixture(scope="module")
def tokenizer():
    cfg = get_config_for_char_tok(
        Path("configs/tokenizer_char.yaml").resolve()
    )
    return CharTokenizer(cfg)


@given(domain_str=domain_strat)
def test_roundtrip_char(domain_str: str, tokenizer: CharTokenizer) -> None:
    ids = tokenizer(domain_str, add_special_tokens=False)["input_ids"]
    recovered = tokenizer.decode(ids, skip_special_tokens=True)
    assert recovered == domain_str
