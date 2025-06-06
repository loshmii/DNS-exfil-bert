from collections import defaultdict
from typing import Dict, List, Iterable, Sequence

import torch
from torch.utils.data import Sampler


class GroupWeightedRandomSampler(Sampler[int]):
    def __init__(
        self,
        dup_gids: Sequence[int],
        dup_gid_to_indices: Dict[int, List[int]],
        group_weights: Sequence[float],
        num_samples: int,
        generator: torch.Generator | None = None,
    ):
        if len(dup_gids) >= 2**24:
            raise ValueError(
                "dup_gids must have < 2**24 unique values, "
                f"got {len(dup_gids)}"
            )

        if len(dup_gids) != len(group_weights):
            raise ValueError(
                "`dup_gids` and `group_weights` must be the same length "
                f"({len(dup_gids)} vs {len(group_weights)})"
            )

        self.dup_gids = torch.as_tensor(dup_gids, dtype=torch.long)
        self.group_weights = torch.as_tensor(group_weights, dtype=torch.double)

        self.dup_gid_to_indices = dup_gid_to_indices
        self.num_samples = num_samples
        self.generator = generator or torch.Generator()

    def __iter__(self) -> Iterable[int]:
        sampled_gids = torch.multinomial(
            self.group_weights,
            self.num_samples,
            replacement=True,
            generator=self.generator,
        )

        for gid in sampled_gids.tolist():
            gid = self.dup_gids[gid].item()
            candidates = self.dup_gid_to_indices[gid]

            pick = torch.randint(
                low=0,
                high=len(candidates),
                size=(),
                generator=self.generator,
            ).item()
            yield candidates[pick]

    def __len__(self) -> int:
        return self.num_samples
