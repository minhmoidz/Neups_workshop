"""Deterministic attacker loader scaffold (G0.2 §4).

Standalone — does not import research_agent.m2_dev or any real dataset class,
so it cannot accidentally touch real data. Intended as the design reference
for a future, separately-authorized fix to the real `build_dev_loaders()`
gap identified in reproduction/reports/G0_1_PROTOCOL_REPAIR_SPEC_2026-08-21.md
§2.1 (the seeded torch.Generator created there is never passed to the
DataLoader, so shuffle falls back to global RNG state).
"""
import hashlib

import torch
from torch.utils.data import DataLoader, Sampler


def worker_seed_fn(worker_id, base_seed):
    """Explicit worker seeding for a future num_workers>0 configuration.

    Not exercised by G0.2 tests (all tests use num_workers=0); provided so a
    future multi-worker configuration has a defined, deterministic seed
    policy instead of relying on PyTorch's implicit default.
    """
    seed = (base_seed + worker_id) % (2 ** 32)
    torch.manual_seed(seed)


class DeterministicEpochSampler(Sampler):
    """Yields a fresh permutation per epoch from an explicit, caller-owned
    torch.Generator — never reads or writes global RNG state.

    Records every epoch's index order for hashing (see `epoch_order_hash`).
    """

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.epoch_indices = []

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randperm(n, generator=self.generator).tolist()
        self.epoch_indices.append(indices)
        return iter(indices)

    def epoch_order_hash(self, epoch: int, sample_ids=None) -> str:
        """SHA256 of the epoch's sample order, using semantic sample IDs when
        available (falls back to raw indices otherwise)."""
        if epoch >= len(self.epoch_indices):
            raise IndexError('Epoch %d order not recorded yet (have %d)' % (epoch, len(self.epoch_indices)))
        indices = self.epoch_indices[epoch]
        h = hashlib.sha256()
        for idx in indices:
            token = sample_ids[idx] if sample_ids is not None else idx
            h.update(str(token).encode('utf-8'))
            h.update(b'\n')
        return h.hexdigest()


def build_deterministic_loader(dataset, seed: int, batch_size: int, shuffle: bool,
                                num_workers: int = 0, sample_ids=None):
    """Build a (loader, sampler_or_None) pair with fully explicit determinism.

    shuffle=True  -> DeterministicEpochSampler with an explicit generator.
    shuffle=False -> plain sequential loader (validation-style), no sampler
                      needed since order is inherently deterministic.

    `sample_ids`, if given, is used for semantic order hashing instead of
    raw integer indices (mirrors the real project's pair-row hashing).
    """
    if shuffle:
        sampler = DeterministicEpochSampler(dataset, seed=seed)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        return loader, sampler
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, None
