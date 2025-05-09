from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

if __name__ == "__main__":
    with hydra.initialize_config_dir(config_dir=str(Path.cwd() / "configs"), job_name="dot_check", version_base="1.3"):
        cfg = hydra.compose(config_name="config", overrides=["tokenizer=bpe8k"], return_hydra_config=True)
        HydraConfig().set_config(cfg)
        tok = BpeTokenizer.from_pretrained(
            path=str(cfg.training.tokenizer.save_dir),
        )
        print(tok.decode(tok("hello.com")["input_ids"], skip_special_tokens=True))
        print("Passed")