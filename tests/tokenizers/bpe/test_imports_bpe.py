from pathlib import Path
import tempfile
from data_pipeline.tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer
from data_pipeline.tokenizers.bpe_dns.v0_1.config.BpeTokConfig import (
    BpeTokConfig,
)


def test_bpe_tokenizer_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        BASE = Path.cwd()
        cfg_path = BASE / "configs" / "bpe_tok_toy.yaml"
        cfg = BpeTokConfig(cfg_path)

        files = [str(BASE / "data" / "toy" / "BPETrainToy.txt")]
        save_path = Path(tmpdir) / "tok"
        tok = BpeTokenizer.from_scratch(cfg=cfg, files=files, save_dir=save_path)
        tok("sample.com")
        tok1 = BpeTokenizer.from_pretrained(path=save_path)
        tok2 = BpeTokenizer.from_pretrained(path=save_path)

        dummy = BpeTokConfig({"alphabet": "abcdefghijklmnopqrstuvwxyz"})

        sample = "xn--sample.com"

        assert tok1(sample) == tok2(sample)
        assert tok1.cfg.vocab_size == tok2.cfg.vocab_size
        assert tok1.cfg.alphabet[0] == tok2.cfg.alphabet[0]
        assert dummy.max_length == 256
