from data_pipeline.tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    tok = BpeTokenizer.from_pretrained(
        path=str(cfg.training.tokenizer.save_dir),
    )
    print(tok.decode(tok("hello.com")["input_ids"], skip_special_tokens=True))
    print("Passed")


if __name__ == "__main__":
    main()
