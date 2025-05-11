import torch
from torch import Generator
from typing import Literal, Optional
from data_pipeline.dns_tokenizers.bpe_dns.v0_1.bpe_tokenizer import BpeTokenizer
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path


class MaskSampler:

    def __init__(self,
        mlm_probability: float = 0.15,
        strategy: Literal["token", "span"] = "token",
        span_lambda: float = 3.0,
        seed: Optional[int] = None,):

        self.mlm_probability = mlm_probability
        self.span_lambda = span_lambda
        self.strategy = strategy

        if seed is not None:
            self._base_seed = int(seed)
            self.base_generator = Generator().manual_seed(self._base_seed)
        else:
            self.base_generator = Generator()
            self._base_seed = self.base_generator.initial_seed()
        
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch
    
    def __call__(self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        special_tokens_mask: torch.BoolTensor,
    ) -> torch.BoolTensor:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        gen = Generator(device).manual_seed(self._base_seed + self._epoch)

        if self.strategy == "token":
            mask = torch.bernoulli(
                torch.full((batch_size, seq_len), self.mlm_probability, device=device), generator=gen,
            ).bool()
        
        else:
            start_prob = self.mlm_probability / self.span_lambda
            starts = torch.bernoulli(
                torch.full((batch_size, seq_len), start_prob, device=device), generator=gen
            ).bool()

            start_idx = starts.nonzero(as_tuple=True)
            if start_idx[0].numel() == 0:
                mask = torch.zeros_like(starts, dtype=torch.bool)

            else:
                raw_lengths = torch.poisson(
                    torch.full((start_idx[0].numel(),), self.span_lambda, device=device), 
                    generator=gen).long().clamp_min(1)
                
                max_allowed = (seq_len - start_idx[1]).clamp_min(1)
                lengths = torch.minimum(raw_lengths, max_allowed)

                flat_starts = start_idx[0] * seq_len + start_idx[1]
                span_offsets = torch.repeat_interleave(flat_starts, lengths)

                max_len = int(lengths.max().item())
                rel = torch.arange(max_len, device=device)
                rel = rel.unsqueeze(0).expand(lengths.size(0), -1) < lengths.unsqueeze(1)
                rel = rel.nonzero(as_tuple=True)[1]
                flat_positions = span_offsets + rel

                mask = torch.zeros(batch_size * seq_len, device=device, dtype=torch.bool)
                mask[flat_positions] = True
                mask = mask.view(batch_size, seq_len)

        mask &= attention_mask.bool()
        mask &= ~special_tokens_mask.bool()

        empty = mask.sum(dim=1) == 0
        if empty.any():
            allowed_mask = attention_mask.bool() & ~special_tokens_mask.bool()
            first_token = allowed_mask[empty].float().multinomial(1, generator=gen).squeeze(-1)
            mask[empty, first_token] = True
            
        return mask

        
if __name__ == "__main__":
    with hydra.initialize_config_dir(
        config_dir=str(Path.cwd() / "configs"),
        job_name="masker_test",
        version_base="1.3",
    ):
        cfg = hydra.compose(
            config_name="config",
            overrides=["tokenizer=bpe_from_pretrained", "model=bert_for_mlm", "dataset=dataset_for_mlm"],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)
    
    tokenizer = BpeTokenizer.from_pretrained(cfg.tokenizer.load_dir)
    encoding = tokenizer(
        text=["a.com", "b.net", "c.org", "idegascina.edufuau"],
        return_attention_mask=True,
        return_special_tokens_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    masker = MaskSampler(
        mlm_probability=0.15,
        strategy="token",
        span_lambda=1.0,
        seed=42,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    special_tokens_mask = encoding["special_tokens_mask"]
    mask = masker(input_ids, attention_mask, special_tokens_mask)
    print(mask)