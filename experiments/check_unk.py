from datasets import load_from_disk
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import CharTokenizer
from data_pipeline.dns_tokenizers.char_dns.v0_1.config.CharTokConfig import (
    get_config_for_char_tok,
)
from pathlib import Path

BASE = Path(__file__).parent.parent

def contains_unk(example):
    for tid in example["input_ids"]:
        if tid == 1:
            return True
    return False

if __name__ == "__main__":
    dataset = load_from_disk(str(BASE / "experiments" / "cache"))
    cfg = get_config_for_char_tok(BASE / "configs" / "tokenizer" / "char.yaml")
    tokenizer = CharTokenizer(cfg)
    train_with_unk = dataset["train"].select(range(100000)).filter(contains_unk)
    print(f"Number of examples with UNK in train set: {len(train_with_unk)}")