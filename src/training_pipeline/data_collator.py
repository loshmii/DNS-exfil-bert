from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from training_pipeline.masker import MaskSampler

@dataclass
class DnsDataCollatorForMLM:
    tokenizer: PreTrainedTokenizerFast
    mask_sampler: MaskSampler
    pad_to_multiple_of: Optional[int] = 8
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1

    def __post_init__(self):
        assert isinstance(self.tokenizer, PreTrainedTokenizerFast), "tokenizer must be a PreTrainedTokenizerFast instance"
        assert self.tokenizer.mask_token_id is not None, "tokenizer must have a mask token id under 'tokenizer.mask_token_id'"

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        print(f"features: {features}")
        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        special_tokens_mask = batch["special_tokens_mask"]
        vocab_size = self.tokenizer.vocab_size
        device = ids.device

        mask = self.mask_sampler(
            input_ids=ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )

        labels = ids.clone()
        labels[~mask] = -100

        probs = torch.rand(ids.shape, device=device)

        flag1 = mask & (probs < self.mask_token_prob)
        ids = torch.where(flag1, self.tokenizer.mask_token_id * torch.ones(ids.shape, device=device), ids).long()

        flag2 = mask & (probs >= self.mask_token_prob) & (probs < self.mask_token_prob + self.random_token_prob)
        ids = torch.where(flag2, torch.randint(vocab_size, ids.shape, device=device), ids).long()

        batch["input_ids"] = ids
        batch["labels"] = labels
        batch["attention_mask"] = attention_mask
        batch.pop("special_tokens_mask", None)
        return batch
    
if __name__ == "__main__":
    from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer
    from training_pipeline.dataset_builder import DnsDatasetBuilder
    from datasets import load_dataset
    import hydra
    from hydra.core.hydra_config import HydraConfig
    from pathlib import Path

    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="dns_data_collator",
        version_base="1.3",
    ):
        cfg = hydra.compose(config_name="config", 
                            overrides=["tokenizer=bpe_from_pretrained", "model=bert_for_mlm", "dataset=dataset_for_mlm"],
                            return_hydra_config=True,
                            )
        HydraConfig().set_config(cfg)
    
    tokenizer = BpeTokenizer.from_pretrained(cfg.tokenizer.load_dir)
    mask_sampler = MaskSampler(
        mlm_probability=0.15,
        strategy="token",
    )

    data_files = {
        "train": [str(f) for f in cfg.dataset.files.train],
        "validation": [str(f) for f in cfg.dataset.files.validation],
        "test": [str(f) for f in cfg.dataset.files.test],
    }

    ds = load_dataset(
        path="src/training_pipeline/dataset_builder.py",
        name="default",
        data_files=data_files,
        streaming=False,
        trust_remote_code=True,
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )
    
    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    
    data_collator = DnsDataCollatorForMLM(
        tokenizer=tokenizer,
        mask_sampler=mask_sampler,
        pad_to_multiple_of=8,
        mask_token_prob=0.8,
        random_token_prob=0.1,
    )

    batch = data_collator([ds["train"][i] for i in range(2)])
    print(batch["input_ids"])