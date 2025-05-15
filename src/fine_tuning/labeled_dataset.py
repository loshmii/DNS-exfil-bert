from datasets import (
    GeneratorBasedBuilder,
    BuilderConfig,
    DatasetInfo,
    SplitGenerator,
    Split,
    load_dataset,
    concatenate_datasets,
    DatasetDict,
    load_from_disk
)

from datasets.features import Features, Value
from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import csv

class LabeledDatasetBuilder(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="default",
            version="1.0.0",
            description="Labeled dataset for fine-tuning",
        )
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description="Labeled dataset for fine-tuning",
            features=Features(
                {
                    "text": Value("string"),
                    "label": Value("int32"),
                }
            ),
            supervised_keys=("text", "label"),
        )
    
    def _split_generators(self, dl_manager):
        files = self.config.data_files
        return [
            SplitGenerator(
                Split.TRAIN, gen_kwargs={"filepath": files["train"]}
            ),
            SplitGenerator(
                Split.VALIDATION,
                gen_kwargs={"filepath": files["validation"]},
            ),
            SplitGenerator(
                Split.TEST,
                gen_kwargs={"filepath": files["test"]},
            ),
        ]
    
    def _generate_examples(self, filepath):
        paths = filepath if isinstance(filepath, (list, tuple)) else [filepath]
        for file_idx, fp in enumerate(paths):
            with open(fp, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    yield f"{file_idx}-{idx}", {
                        "text": row["Subdomain"],
                        "label": int(row["Exfiltration"]),
                    }

def shuffle_and_split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    if (train_ratio + val_ratio) > 1.0:
        raise ValueError("The sum of train_ratio and val_ratio must be less than or equal to 1.0.")
    if train_ratio < 0.0 or val_ratio < 0.0:
        raise ValueError("train_ratio and val_ratio must be non-negative.")
    
    ds = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    train_test_split = ds.train_test_split(
        test_size=1 - (train_ratio + val_ratio),
        seed=seed,
    )
    test_ds = train_test_split["test"]
    train_val_split = train_test_split["train"].train_test_split(
        test_size = val_ratio / (train_ratio + val_ratio),
        seed=seed,
    )
    train_ds = train_val_split["train"]
    val_ds = train_val_split["test"]
    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

def convert_to_arrow(data_files, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("csv", data_files=data_files)

    #splits = shuffle_and_split_dataset(ds)
    splits = ds
    splits.save_to_disk(str(path))

    return splits

if __name__ == "__main__":

    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="labeled_dataset",
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "tokenizer=bpe8k_pretrained",
                "dataset=dataset_for_labeled",
            ],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)
    
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    files_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    ds = load_from_disk(files_cfg["data_dir"])
    def _tokenize(tokenizer, examples):
        return tokenizer(
            examples["Subdomain"],
            truncation=True,
            padding=True,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            return_tensors="pt",
        )
    ds = ds.map(
        lambda x: _tokenize(tokenizer, x),
        batched=True,
        remove_columns=["Subdomain"],
    )
    print(ds)