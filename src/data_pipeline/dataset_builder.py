from datasets import (
    GeneratorBasedBuilder,
    BuilderConfig,
    DatasetInfo,
    SplitGenerator,
    Split,
    load_dataset,
)
from datasets.features import Features, Value


class DnsDatasetBuilder(GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="default",
            version="1.0.0",
            description="DNS query streams for exfiltration detection",
        )
    ]

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description="DNS query streams for exfiltration detection",
            features=Features(
                {
                    "text": Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        files = self.config.data_files
        unwrap = lambda x: x[0] if isinstance(x, list) else x
        return [
            SplitGenerator(Split.TRAIN,      gen_kwargs={"filepath": unwrap(files["train"])}),
            SplitGenerator(Split.VALIDATION, gen_kwargs={"filepath": unwrap(files["validation"])}),
            SplitGenerator(Split.TEST,       gen_kwargs={"filepath": unwrap(files["test"])}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                yield idx, {"text": line.strip()}


if __name__ == "__main__":
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from pathlib import Path

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

    print(cfg.dataset.files.train)
    data_files = {
        "train": str(cfg.dataset.files.train),
        "validation": str(cfg.dataset.files.validation),
        "test": str(cfg.dataset.files.test),
    }

    ds = load_dataset(
        path="src/data_pipeline/dataset_builder.py",
        name="default",
        data_files=data_files,
        streaming=False,
        trust_remote_code=True,
    )
    print(ds)