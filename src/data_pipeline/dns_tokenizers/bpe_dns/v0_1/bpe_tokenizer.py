from transformers import PreTrainedTokenizerFast
from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    processors,
    normalizers,
    trainers,
    decoders,
)
from typing import Sequence
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.config.BpeTokConfig import (
    BpeTokConfig,
)
from pathlib import Path
from typing import Union

DIR = Path(__file__).parent.parent.parent.parent.parent.parent


class BpeTokenizer(PreTrainedTokenizerFast):
    def __init__(
        self,
        cfg: BpeTokConfig,
        core: Tokenizer,
        **kwargs,
    ):
        super().__init__(
            tokenizer_object=core,
            model_max_length=cfg.max_length,
            padding_side="right",
            truncation_side="right",
            pad_token=cfg.pad_token,
            unk_token=cfg.unk_token,
            cls_token=cfg.cls_token,
            sep_token=cfg.sep_token,
            mask_token=cfg.mask_token,
            **kwargs,
        )

        self.cfg = cfg

    @classmethod
    def from_scratch(
        cls,
        cfg: BpeTokConfig,
        files: Sequence[Union[str, Path]],
        save_dir: Union[str, Path],
    ) -> "BpeTokenizer":
        tok_obj = Tokenizer(models.BPE(unk_token=cfg.unk_token))

        tok_obj.normalizer = normalizers.Sequence(
            [
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
            ]
        )

        tok_obj.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Punctuation(behavior="isolated"),
            ]
        )

        trainer = trainers.BpeTrainer(
            vocab_size=cfg.vocab_size,
            special_tokens=[
                cfg.pad_token,
                cfg.unk_token,
                cfg.cls_token,
                cfg.sep_token,
                cfg.mask_token,
            ],
            initial_alphabet=list(cfg.alphabet),
            min_frequency=1,
            show_progress=True,
        )
        files = list(map(str, files))
        tok_obj.train(files=files, trainer=trainer)

        tok_obj.decoder = decoders.ByteLevel()

        cls_id = tok_obj.token_to_id(cfg.cls_token)
        sep_id = tok_obj.token_to_id(cfg.sep_token)
        tok_obj.post_processor = processors.TemplateProcessing(
            single=f"{cfg.cls_token} $A {cfg.sep_token}",
            special_tokens=[
                (
                    cfg.cls_token,
                    cls_id,
                ),
                (
                    cfg.sep_token,
                    sep_id,
                ),
            ],
        )
        if cfg.padding:
            tok_obj.enable_padding(
                direction="right",
                pad_id=tok_obj.token_to_id(cfg.pad_token),
                pad_token=cfg.pad_token,
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

        tok = cls(cfg, tok_obj)
        path = Path(save_dir).absolute()
        path.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(path)
        cfg.to_file(path / "config.yaml")

        return tok

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        **kwargs,
    ) -> "BpeTokenizer":
        path = Path(path).resolve()
        core = PreTrainedTokenizerFast.from_pretrained(
            path, **kwargs
        ).backend_tokenizer
        cfg = BpeTokConfig.from_file(path / "config.yaml")
        return cls(cfg, core)


if __name__ == "__main__":

    cfg = BpeTokConfig(DIR / "experiments" / "toy_cfgs" / "bpe_tok_toy.yaml")
    print(cfg.alphabet)

    tok = BpeTokenizer.from_scratch(
        cfg,
        files=[
            DIR / "experiments" / "toy_artifacts" / "BPETrainToy.txt",
        ],
        save_dir=DIR / "experiments" / "toy_artifacts" / "params",
    )

    loaded_tok = BpeTokenizer.from_pretrained(
        DIR / "experiments" / "toy_artifacts" / "params",
        local_files_only=True,
    )
    sample = "xn--example.com"
    assert tok(sample) == loaded_tok(sample)
    print("Sample tokenization is the same")
