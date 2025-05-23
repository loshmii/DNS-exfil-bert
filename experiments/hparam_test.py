import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from training_pipeline.trainer import (
    MLMTrainer,
    add_parent_resolver,
    add_num_resolver,
    parse_dataclasses,
    MaskingCallback,
    FileLoggingCallback,
    PerplexityCallback,
    TensorBoardCallback,
)
from training_pipeline.builders import (
    MLMDatasetBuilder,
    
)
from training_pipeline.data_collator import DnsDataCollatorForMLM
from training_pipeline.masker import MaskSampler
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer
from transformers import (
    BertConfig,
    AutoModelForMaskedLM
)

@hydra.main(
    config_path=str(Path(__file__).parent.parent/ "configs"),
    config_name="config",
    version_base="1.3",
)
@add_parent_resolver
@add_num_resolver
def main(cfg: DictConfig):

    model_args, train_args = parse_dataclasses(cfg)

    output_dir = Path(train_args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ch = logging.FileHandler(
        str(output_dir / "training.log"), mode="w"
    )
    ch.setFormatter(fmt)
    root.addHandler(ch)

    tokenizer = BpeTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True)
    )

    model_cfg = BertConfig(
        **OmegaConf.to_container(cfg.model_config, resolve=True),
        vocab_size=tokenizer.vocab_size,
    )

    model = AutoModelForMaskedLM.from_config(model_cfg)

    builder = MLMDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.builder_args, resolve=True)
    )
    ds = builder.build()

    train_ds = ds["train"].select(range(50))
    eval_ds = ds["validation"].select(range(10))
    test_ds = ds["test"].select(range(10))

    mask_sampler = MaskSampler(
        **OmegaConf.to_container(cfg.training_arguments.mask_args, resolve=True)
    )

    data_collator = DnsDataCollatorForMLM(
        tokenizer=tokenizer,
        mask_sampler=mask_sampler,
        **OmegaConf.to_container(cfg.training_arguments.collator_args, resolve=True)
    )

    trainer = MLMTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            MaskingCallback(data_collator),
            FileLoggingCallback(),
            PerplexityCallback(),
        ],
        compute_metrics=None,
    )

    trainer.train()

    loss = trainer.evaluate(
        eval_dataset=test_ds,
        metric_key_prefix="test",
    )

    trainer.save_model(str(output_dir / "model"))
    trainer.save_state()
    
if __name__ == "__main__":
    main()