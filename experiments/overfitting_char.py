from training_pipeline.arguments import parse_dataclasses
from training_pipeline.trainer import (
    MLMTrainer,
    MaskingCallback,
    FileLoggingCallback,
    PerplexityCallback,
)
from training_pipeline.masker import MaskSampler
from training_pipeline.data_collator import DnsDataCollatorForMLM
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict, IterableDatasetDict, IterableDataset
import logging
from transformers import (
    BertConfig,
    AutoModelForMaskedLM,
)
from omegaconf import OmegaConf
from data_pipeline.dns_tokenizers.char_dns.v0_1.config.CharTokConfig import get_config_for_char_tok
from data_pipeline.dns_tokenizers.char_dns.v0_1.char_tokenizer import CharTokenizer
from training_pipeline.trainer import (
    add_parent_resolver,
    add_num_resolver
)
import hydra
from omegaconf import OmegaConf, DictConfig
from training_pipeline.builders import MLMDatasetBuilder

DIR = Path(__file__).parent.parent.resolve()

@hydra.main(
    version_base="1.3",
    config_path=str(DIR / "configs"),
    config_name="config_for_conv_test_char",
)
@add_parent_resolver
@add_num_resolver
def main(cfg : DictConfig):

    model_args, train_args = parse_dataclasses(cfg)

    output_dir = Path(train_args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    ch = logging.FileHandler(str(output_dir / "overfitting.log"), mode="w")
    ch.setFormatter(fmt)
    root.addHandler(ch)


    tok_cfg = get_config_for_char_tok(
        OmegaConf.to_container(cfg.tokenizer, resolve=True),
    )

    tokenizer = CharTokenizer(tok_cfg)

    model_cfg = BertConfig(
        **OmegaConf.to_container(cfg.model_config, resolve=True),
        vocab_size=tokenizer.vocab_size,
    )
    model = AutoModelForMaskedLM.from_config(model_cfg)

    builder = MLMDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.MLM_builder_args, resolve=True),
    )

    ds = builder.build()

    train_ds = ds["train"]
    eval_ds = ds["validation"]
    test_ds = ds["test"]

    mask_sampler = MaskSampler(
        **OmegaConf.to_container(
            cfg.training_arguments.mask_args, resolve=True
        )
    )
    data_collator = DnsDataCollatorForMLM(
        tokenizer=tokenizer,
        mask_sampler=mask_sampler,
        **OmegaConf.to_container(
            cfg.training_arguments.MLM_collator_args, resolve=True
        ),
    )

    trainer = MLMTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[
            FileLoggingCallback(),
            MaskingCallback(data_collator),
            PerplexityCallback(),
        ],
        compute_metrics=None,
    )

    trainer.train()

    test_loss = trainer.evaluate(
        eval_dataset=test_ds,
        metric_key_prefix="test",
    )

    trainer.save_model(
        output_dir=str(output_dir / "model"),
    )
    trainer.save_state()


if __name__ == "__main__":
    main()
