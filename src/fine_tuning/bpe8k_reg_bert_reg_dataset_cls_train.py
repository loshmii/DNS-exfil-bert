import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import transformers
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
)
from training_pipeline.trainer import (
    add_num_resolver,
    add_parent_resolver,
    FileLoggingCallback,
)
from training_pipeline.cls_trainer import CLSTrainer, ROCCurveCallback
from training_pipeline.arguments import parse_dataclasses
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)
from training_pipeline.builders import CLSDatasetBuilder
from training_pipeline.data_collator import DnsDataCollatorForCLC


@hydra.main(
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="bpe8k_reg_bert_reg_dataset_cls_train",
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
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    ch = logging.FileHandler(str(output_dir / "training.log"), mode="w")
    ch.setFormatter(fmt)
    root.addHandler(ch)

    tokenizer = BpeTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        **OmegaConf.to_container(cfg.model, resolve=True),
    )

    builder = CLSDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.CLS_builder_args, resolve=True),
    )
    ds = builder.build()

    weights = builder.get_class_weights(ds)

    train_ds = ds["train"]
    eval_ds = ds["validation"]
    test_ds = ds["test"]

    data_collator = DnsDataCollatorForCLC(
        tokenizer=tokenizer,
        **OmegaConf.to_container(
            cfg.training_arguments.CLS_collator_args, resolve=True
        ),
    )

    writer = SummaryWriter(log_dir=str(train_args.logging_dir))
    trainer = CLSTrainer(
        model=model,
        args=train_args,
        class_weights=weights,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[FileLoggingCallback()],
        compute_metrics=None,
    )

    roc_cb = ROCCurveCallback(
        writer=writer,
        eval_dataset=eval_ds,
        trainer=trainer,
    )
    trainer.add_callback(roc_cb)

    trainer.train()

    loss = trainer.evaluate(
        eval_dataset=eval_ds,
        metric_key_prefix="test",
    )

    trainer.save_model(
        output_dir=str(output_dir / "model"),
    )
    trainer.save_state()


if __name__ == "__main__":
    main()
