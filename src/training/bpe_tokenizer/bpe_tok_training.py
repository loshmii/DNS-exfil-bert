from data_pipeline.tokenizers.bpe_dns.v0_1.config.BpeTokConfig import (
    BpeTokConfig,
    get_config_for_bpe_tok,
)
from data_pipeline.tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    tok_cfg = BpeTokConfig(cfg.tokenizer)
    files = [str(f) for f in cfg.training.tokenizer.training_files]
    save_dir = str(cfg.training.tokenizer.save_dir)
    print(tok_cfg)
    print(files)
    print(save_dir)
    tok = BpeTokenizer.from_scratch(
        cfg = tok_cfg,
        files = files,
        save_dir= save_dir
    )
    print("Tokenizer created and saved at:", cfg["training"]["tokenizer"]["save_dir"])


if __name__ == "__main__" :
    main()