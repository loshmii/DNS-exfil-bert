import math
import random
from collections import Counter

import pytest
import torch

from training_pipeline.sampler import GroupWeightedRandomSampler


@pytest.fixture(scope="module")
def toy_setup():
    dup_gid_to_indices = {
        10: list(range(0, 5)),
        11: list(range(5, 7)),
        12: list(range(7, 8)),
    }

    dup_gids = sorted(dup_gid_to_indices.keys())
    group_weights = [math.sqrt(len(dup_gid_to_indices[g])) for g in dup_gids]
    return dup_gids, dup_gid_to_indices, group_weights


def test_length_and_membership(toy_setup):
    dup_gids, mapping, w = toy_setup
    N = 50
    gen = torch.Generator().manual_seed(0)

    sampler = GroupWeightedRandomSampler(
        dup_gids=dup_gids,
        dup_gid_to_indices=mapping,
        group_weights=w,
        num_samples=N,
        generator=gen,
    )

    samples = list(iter(sampler))

    assert len(samples) == N

    allowed = {idx for lst in mapping.values() for idx in lst}
    assert all(idx in allowed for idx in samples)


def test_weighted_sampling_bias(toy_setup):
    dup_gids, mapping, w = toy_setup
    N = 10000
    gen = torch.Generator().manual_seed(0)

    sampler = GroupWeightedRandomSampler(
        dup_gids=dup_gids,
        dup_gid_to_indices=mapping,
        group_weights=w,
        num_samples=N,
        generator=gen,
    )

    idx2gid = {}
    for gid, lst in mapping.items():
        for idx in lst:
            idx2gid[idx] = gid

    gid_counts = Counter(idx2gid[i] for i in sampler)

    assert gid_counts[10] > gid_counts[12] > 0


def test_repeatability(toy_setup):
    dup_gids, mapping, w = toy_setup
    gen1 = torch.Generator().manual_seed(0)
    gen2 = torch.Generator().manual_seed(0)

    s1 = list(
        GroupWeightedRandomSampler(
            dup_gids=dup_gids,
            dup_gid_to_indices=mapping,
            group_weights=w,
            num_samples=100,
            generator=gen1,
        )
    )
    s2 = list(
        GroupWeightedRandomSampler(
            dup_gids=dup_gids,
            dup_gid_to_indices=mapping,
            group_weights=w,
            num_samples=100,
            generator=gen2,
        )
    )

    assert s1 == s2


def test_too_many_groups():
    with pytest.raises(ValueError):
        big = list(range(2**24))
        m = {i: [i] for i in big}
        w = [1.0] * len(big)
        GroupWeightedRandomSampler(
            dup_gids=big,
            dup_gid_to_indices=m,
            group_weights=w,
            num_samples=1,
        )


def test_length_mismatch_raises():
    dup_gids = [1, 2, 3]
    mapping = {1: [0], 2: [1], 3: [2]}
    wrong_w = [1.0, 2.0]

    with pytest.raises((ValueError, RuntimeError, IndexError)):
        GroupWeightedRandomSampler(
            dup_gids=dup_gids,
            dup_gid_to_indices=mapping,
            group_weights=wrong_w,
            num_samples=1,
        )
