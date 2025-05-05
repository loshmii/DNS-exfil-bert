import hydra
from omegaconf import DictConfig
from data_pipeline.tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    load_file = str(cfg.training.tokenizer.save_dir)
    tok = BpeTokenizer.from_pretrained(
        path=load_file,
    )
    print("Tokenizer loaded from:", load_file)
    print(tok.decode(tok("hello")["input_ids"], skip_special_tokens=True) == "hello")
    print("Passed")


if __name__ == "__main__":
    main()
