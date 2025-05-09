from transformers import PreTrainedTokenizerFast
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    processors,
    normalizers,
)
from typing import Optional
from data_pipeline.dns_tokenizers.char_dns.v0_1.config.CharTokConfig import (
    CharTokConfig,
    get_config_for_char_tok,
)
import os
import yaml
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def _build_specials():
    return {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
    }


class CharTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        cfg: CharTokConfig,
        tokenizer_object: Optional[Tokenizer] = None,
        **kwargs,
    ):
        if tokenizer_object is not None:
            tok_obj = tokenizer_object
        else:
            specials = _build_specials()

            char_ids = {ch: idx + len(specials) for idx, ch in enumerate(cfg.alphabet)}

            full_vocab = {**specials, **char_ids}

            tok_obj = Tokenizer(
                models.WordLevel(
                    vocab=full_vocab, unk_token=cfg.special_tokens["unk_token"]
                )
            )

            tok_obj.normalizer = normalizers.Sequence(
                [
                    normalizers.Lowercase(),
                    normalizers.Replace(pattern=r"\s+", content=""),
                ]
            )

            tok_obj.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
            cls_tok = cfg.special_tokens["cls_token"]
            sep_tok = cfg.special_tokens["sep_token"]
            tok_obj.post_processor = processors.TemplateProcessing(
                single=f"{cls_tok} $A {sep_tok}",
                special_tokens=[
                    (
                        cfg.special_tokens["cls_token"],
                        specials[cfg.special_tokens["cls_token"]],
                    ),
                    (
                        cfg.special_tokens["sep_token"],
                        specials[cfg.special_tokens["sep_token"]],
                    ),
                ],
            )
            if cfg.padding:
                tok_obj.enable_padding(
                    direction="right",
                    pad_id=specials[cfg.special_tokens["pad_token"]],
                    pad_token=cfg.special_tokens["pad_token"],
                    pad_type_id=0,
                    length=cfg.max_length,
                    pad_to_multiple_of=64,
                )

            if cfg.truncation:
                tok_obj.enable_truncation(
                    max_length=cfg.max_length,
                    stride=0,
                    strategy="longest_first",
                    direction="right",
                )

        super().__init__(
            tokenizer_object=tok_obj,
            model_max_length=cfg.max_length,
            padding_side="right",
            truncation_side="right",
            pad_token=cfg.special_tokens["pad_token"],
            unk_token=cfg.special_tokens["unk_token"],
            cls_token=cfg.special_tokens["cls_token"],
            sep_token=cfg.special_tokens["sep_token"],
            mask_token=cfg.special_tokens["mask_token"],
            **kwargs,
        )

        self.cfg = cfg

    def decode(self, token_ids, skip_special_tokens=False):

        text = self.backend_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return text.replace(" ", "")

    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        cfg_path = Path(save_directory) / "char_tok_config.yaml"
        self.cfg.to_file(cfg_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        cfg_path = os.path.join(pretrained_model_name_or_path, "char_tok_config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = CharTokConfig(config=yaml.safe_load(f))

        tok_json = os.path.join(pretrained_model_name_or_path, "tokenizer.json")
        rust_tok = Tokenizer.from_file(tok_json)

        return cls(cfg, tokenizer_object=rust_tok, **kwargs)


if __name__ == "__main__":
    DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
    cfg_path = DIR / "configs" / "tokenizer_char.yaml"
    cfg = get_config_for_char_tok(cfg_path)
    tokenizer = CharTokenizer(cfg)
    encodings = tokenizer(
        ["helloworld", "a", "XN--example.com", "example.com"],
        padding=True,
        truncation=True,
        return_attention_mask=True,
    )
    print(encodings["input_ids"])
    print(encodings["attention_mask"])
    print(tokenizer.decode(encodings["input_ids"][2], skip_special_tokens=True))
    print("Done")
