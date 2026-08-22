"""P0_SAMPLER_V1 / P0_ORDERHASH_V1 deterministic epoch sampler.

The epoch permutation is a pure function of
    (schema version, master attacker seed, train-order domain, epoch index,
     dataset length)
via a fresh local CPU torch.Generator per call. It is independent of global RNG
state, of model initialization, and of arm execution order.
"""
import hashlib
import struct

import torch
from torch.utils.data import DataLoader

try:
    from .seed_contract import derive_epoch_order_seed
except ImportError:
    from seed_contract import derive_epoch_order_seed

SAMPLER_SCHEMA = "P0_SAMPLER_V1"
ORDERHASH_SCHEMA = "P0_ORDERHASH_V1"

try:
    from torch.utils.data import Sampler

    class DeterministicEpochSampler(Sampler):
        """Yields the SAME fixed permutation every epoch (pure function)."""

        def __init__(self, master_seed, dataset_length):
            if dataset_length <= 0:
                raise ValueError("dataset_length must be positive")
            self.master_seed = int(master_seed)
            self.dataset_length = int(dataset_length)
            self._perm = self.permutation(self.master_seed, 0, self.dataset_length)
            self.current_epoch = 0

        def permutation(self, master_seed, epoch_index, dataset_length):
            return build_permutation(master_seed, epoch_index, dataset_length)

        def set_epoch(self, epoch_index):
            if not isinstance(epoch_index, int) or epoch_index < 0:
                raise ValueError("epoch_index must be a non-negative int")
            self.current_epoch = epoch_index
            self._perm = self.permutation(self.master_seed, epoch_index,
                                          self.dataset_length)

        def __iter__(self):
            return iter(self._perm)

        def __len__(self):
            return self.dataset_length

except ImportError:  # pragma: no cover - torch always present in this repo
    Sampler = object


def build_permutation(master_seed, epoch_index, dataset_length):
    """Pure deterministic permutation of range(dataset_length)."""
    order_seed = derive_epoch_order_seed(master_seed, epoch_index,
                                         dataset_length, SAMPLER_SCHEMA)
    generator = torch.Generator()
    generator.manual_seed(order_seed)
    return torch.randperm(dataset_length, generator=generator).tolist()


def order_hash(perm, master_seed, epoch_index, dataset_length):
    """Stable SHA-256 over the complete ordered index sequence.

    Serialization: schema prefix bytes; little-endian signed 64-bit encodings of
    dataset_length and epoch_index; then each index as little-endian signed
    64-bit. No image paths or patient identifiers are hashed. Consuming this
    function does NOT advance any sampler state.
    """
    h = hashlib.sha256()
    h.update(ORDERHASH_SCHEMA.encode("utf-8"))
    h.update(b"|")
    h.update(struct.pack("<q", int(dataset_length)))
    h.update(struct.pack("<q", int(epoch_index)))
    for idx in perm:
        idx_i = int(idx)
        if idx_i < 0 or idx_i >= int(dataset_length):
            raise ValueError("index %d outside [0, %d)" % (idx_i, dataset_length))
        h.update(struct.pack("<q", idx_i))
    return h.hexdigest()


def expected_epoch_order_hashes(master_seed, epochs, dataset_length):
    """All expected per-epoch order hashes WITHOUT constructing a sampler."""
    return {
        e: order_hash(build_permutation(master_seed, e, dataset_length),
                      master_seed, e, dataset_length)
        for e in range(epochs)
    }


def make_paired_dataloader(dataset, master_seed, worker_base_seed, num_workers=0,
                           batch_size=32):
    """DataLoader bound to the deterministic sampler; shuffle MUST be False.

    The sampler is constructed internally from the master seed so arms never
    share mutable generator state.
    """
    length = len(dataset)
    perm = build_permutation(master_seed, 0, length)

    class _FixedOrderSampler(torch.utils.data.Sampler):
        def __init__(self, permutation):
            self.permutation = list(permutation)

        def __iter__(self):
            return iter(self.permutation)

        def __len__(self):
            return len(self.permutation)

    worker_init = _make_worker_init(worker_base_seed)

    def _build(epoch_index):
        sampler = _FixedOrderSampler(
            build_permutation(master_seed, epoch_index, length))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          sampler=sampler, num_workers=num_workers,
                          worker_init_fn=worker_init,
                          persistent_workers=False)

    return _build


def _make_worker_init(worker_base_seed):
    def _worker_init_fn(worker_id):
        seed = derive_seed(int(worker_base_seed), "dataloader_worker_%d" % worker_id)
        import random
        import numpy as np
        import torch as _torch
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        _torch.manual_seed(seed)
    return _worker_init_fn
