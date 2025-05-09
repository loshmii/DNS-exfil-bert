import hypothesis.strategies as st
from hypothesis import given
import pytest
from hydra import initialize_config_dir, compose
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pathlib import Path
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer

domain_start = st.from_regex(r"(?:[a-z0-9\-]+\.)+[a-z]{2,}", fullmatch=True)


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    with initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="test",
        version_base="1.3",
    ):
        cfg = compose(
            config_name="config",
            overrides=["tokenizer=bpe8k", "hydra.run.dir=."],
        )
    return cfg


@given(domain_str=domain_start)
def test_roundtrip(domain_str: str, cfg: DictConfig):
    tokenizer = BpeTokenizer.from_pretrained(
        path=Path(to_absolute_path(cfg.training.tokenizer.save_dir))
    )
    ids = tokenizer(domain_str, add_special_tokens=False)["input_ids"]
    recovered = tokenizer.decode(ids, skip_special_tokens=True)
    assert recovered == domain_str
