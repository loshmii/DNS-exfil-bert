import pytest
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import (
    CharTokenizer,
)
from data_pipeline.dns_tokenizers.char_dns.v0_1.config.CharTokConfig import (
    get_config_for_char_tok,
)


@pytest.fixture(scope="module")
def tokenizer() -> CharTokenizer:
    cfg = get_config_for_char_tok("configs/tokenizer/char.yaml")
    return CharTokenizer(cfg)


def test_encode_speed(benchmark, tokenizer: CharTokenizer) -> None:
    samples = [f"test{i}.com" for i in range(10000)]

    def run() -> None:
        tokenizer(samples, padding=False, truncation=False)

    _ = benchmark(run)
