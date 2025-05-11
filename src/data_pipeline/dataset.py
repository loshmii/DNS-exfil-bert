from pathlib import Path
from typing import Dict, Sequence, Union
from datasets import load_dataset, DatasetDict
from training_pipeline.arguments import DataArguments, parse_dataclasses
import hydra
from hydra.core.hydra_config import HydraConfig
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)


def _build_file_map(
    root: Union[str, Path], layout: str
) -> Dict[str, Sequence[str]]:
    base = Path(root) / "data"
    if layout == "processed":
        proc = base / "processed"
        return {
            split: [str(proc / f"{split}.txt")]
            for split in ["train", "val", "test"]
        }
    elif layout == "raw":
        raw = base / "raw"
        return {
            split: [
                str(raw / split / "positive.csv"),
                str(raw / split / "negative.csv"),
            ]
            for split in ["train", "val", "test"]
        }
    else:
        raise ValueError(f"Unknown layout: {layout}")


def load_dns_dataset(
    data_args: DataArguments,
    tokenizer=None,
) -> DatasetDict:
    file_map = _build_file_map(data_args.root, data_args.layout)
    if data_args.layout == "processed":
        ds = load_dataset("text", data_files=file_map)
    else:
        ds = load_dataset("csv", data_files=file_map)
        ds = ds.rename_column("Subdomain", "text")
        ds = ds.remove_columns(["Exfiltration"])
    if tokenizer is not None:

        def _tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=data_args.block_size,
                padding=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                return_special_tokens_mask=True,
                return_attention_mask=True,
            )

        ds = ds.map(
            _tokenize, batched=True, remove_columns=["text"], num_proc=4
        )
    return ds


if __name__ == "__main__":
    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="dataset_test",
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tokenizer=bpe_from_pretrained", "model=bert_for_mlm"],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)

    print(cfg.tokenizer.load_dir)
    tokenizer = BpeTokenizer.from_pretrained(cfg.tokenizer.load_dir)
    _, data_args, _ = parse_dataclasses(cfg)
    data_set = load_dns_dataset(data_args, tokenizer=tokenizer)
    print(data_set["train"][0])
