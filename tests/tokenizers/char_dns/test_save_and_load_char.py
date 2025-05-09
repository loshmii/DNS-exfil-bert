import tempfile
import pytest
from pathlib import Path
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import (
    CharTokenizer,
)
from data_pipeline.dns_tokenizers.char_dns.v0_1.config.CharTokConfig import (
    get_config_for_char_tok,
)


@pytest.fixture(scope="module")
def tokenizer():
    cfg = get_config_for_char_tok(
        Path("configs/tokenizer_char.yaml").resolve()
    )
    return CharTokenizer(cfg)


def test_save_pretrained_roundtrip_char(tokenizer: CharTokenizer) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save_pretrained(tmpdir)
        tok2 = CharTokenizer.from_pretrained(tmpdir)

        assert tok2.vocab_size == tokenizer.vocab_size
        assert tok2._pad_token_type_id == tokenizer._pad_token_type_id
        example = "example.com"
        decoded = tok2.decode(tok2.encode(example), skip_special_tokens=True)
        assert decoded == example


def test_half_filled_dict_char() -> None:
    half = {"alphabet": "abc"}
    cfg = get_config_for_char_tok(half)
    assert cfg.max_length == 256
