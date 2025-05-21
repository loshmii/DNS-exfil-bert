from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from training_pipeline.masker import MaskSampler
from omegaconf import OmegaConf
from training_pipeline.builders import MLMDatasetBuilder
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import (
    BpeTokenizer,
)

ROOT = Path(__file__).parent.parent.parent


@dataclass
class DnsDataCollatorForMLM:
    tokenizer: PreTrainedTokenizerFast
    mask_sampler: MaskSampler
    pad_to_multiple_of: Optional[int] = 8
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1

    def __post_init__(self):
        assert isinstance(
            self.tokenizer, PreTrainedTokenizerFast
        ), "tokenizer must be a PreTrainedTokenizerFast instance"
        assert (
            self.tokenizer.mask_token_id is not None
        ), "tokenizer must have a mask token id under 'tokenizer.mask_token_id'"
        if self.mask_token_prob + self.random_token_prob > 1.0:
            raise ValueError(
                "mask_token_prob + random_token_prob must be less than or equal to 1.0"
            )
        if self.mask_token_prob < 0.0 or self.random_token_prob < 0.0 or self.mask_token_prob > 1.0 or self.random_token_prob > 1.0:
            raise ValueError(
                "mask_token_prob and random_token_prob must be between 0.0 and 1.0"
            )
        

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
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
        ids = torch.where(
            flag1,
            self.tokenizer.mask_token_id
            * torch.ones(ids.shape, device=device),
            ids,
        ).long()

        flag2 = (
            mask
            & (probs >= self.mask_token_prob)
            & (probs < self.mask_token_prob + self.random_token_prob)
        )
        ids = torch.where(
            flag2, torch.randint(low=len(self.tokenizer.special_tokens_map), high=vocab_size, size=ids.shape, device=device), ids
        ).long()

        batch["input_ids"] = ids
        batch["labels"] = labels
        batch["attention_mask"] = attention_mask
        batch.pop("special_tokens_mask", None)
        return batch


@hydra.main(
    config_path=str(ROOT / "configs"),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):
    tokenizer = BpeTokenizer.from_pretrained(
        **OmegaConf.to_container(cfg.tokenizer, resolve=True),
    )

    mask_sampler = MaskSampler(
        **OmegaConf.to_container(
            cfg.training_arguments.mask_args, resolve=True
        )
    )

    builder = MLMDatasetBuilder(
        tokenizer=tokenizer,
        **OmegaConf.to_container(cfg.dataset.builder_args, resolve=True),
    )

    ds = builder.build()

    data_collator = DnsDataCollatorForMLM(
        tokenizer=tokenizer,
        mask_sampler=mask_sampler,
        **OmegaConf.to_container(
            cfg.training_arguments.collator_args, resolve=True
        ),
    )

    batch = data_collator([ds["train"][i] for i in range(2)])
    print(batch["input_ids"])


if __name__ == "__main__":
    main()
