from data_pipeline.dns_tokenizers.bpe_dns.v0_1.config.BpeTokConfig import (
    BpeTokConfig,
)
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)
import hydra
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

with hydra.initialize_config_dir(
    str(Path.cwd() / "configs"), job_name="bpe_tok_test", version_base="1.3"
):
    cfg = hydra.compose(
        config_name="config",
        overrides=["tokenizer=bpe8k"],
        return_hydra_config=True,
    )
    HydraConfig().set_config(cfg)
    tok_cfg = BpeTokConfig(cfg.tokenizer)
    files = [str(f) for f in cfg.training.tokenizer.training_files]
    save_dir = str(cfg.training.tokenizer.save_dir)
    print(tok_cfg)
    print(files)
    print(save_dir)
    tok = BpeTokenizer.from_scratch(
        cfg=tok_cfg, files=files, save_dir=save_dir
    )
    print("Example tokenized text:", tok.encode("hello"))
    print(
        "Tokenizer created and saved at:",
        cfg["training"]["tokenizer"]["save_dir"],
    )
    print("Passed")
