import hydra
from omegaconf import DictConfig
from data_pipeline.tokenizers.bpe_dns.v0_1.config.BpeTokConfig import BpeTokConfig
from data_pipeline.tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    tok_cfg = BpeTokConfig(cfg.tokenizer)
    load_file = str(cfg.training.tokenizer.save_dir)
    tok = BpeTokenizer.from_pretrained(
        cfg = tok_cfg,
        path=load_file,
    )
    print("Tokenizer loaded from:", load_file)
    print(tok.decode(tok("hello")['input_ids'], skip_special_tokens=True) == "hello")

if __name__ == "__main__":
    main()