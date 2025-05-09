import pytest
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import CharTokenizer
from data_pipeline.dns_tokenizers.char_dns.v0_1.config.CharTokConfig import (
    get_config_for_char_tok,
)
from pathlib import Path


@pytest.fixture(scope="module")
def tokenizer() -> CharTokenizer:
    cfg = get_config_for_char_tok(Path("configs/tokenizer_char.yaml").resolve())
    return CharTokenizer(cfg)


def test_padding_and_attention_mask_char(tokenizer: CharTokenizer) -> None:
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
