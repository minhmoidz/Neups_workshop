"""P0_SAMPLER_V1_1 / P0_ORDERHASH_V1_1 deterministic epoch sampler.

Revision P0_2_1 (external source-review closeout):
- epoch-order seed binds (sampler schema, derived train_order seed, epoch, N);
- order_hash validates completeness/duplicates/range/ambiguous inputs;
- explicit per-loader CPU torch.Generator from dataloader_worker_base + epoch;
- top-level picklable worker initializer (no closures);
- DataLoader construction/iteration never touches main-process global RNG;
- strict type validation everywhere.
"""
import functools
import hashlib
import struct

import torch
from torch.utils.data import DataLoader

from seed_contract import (
    derive_seed,
    derive_epoch_order_seed,
    validate_dataset_length,
    validate_epoch_index,
    _validate_master_seed,
)

SAMPLER_SCHEMA = "P0_SAMPLER_V1_1"
ORDERHASH_SCHEMA = "P0_ORDERHASH_V1_1"
LOADERGEN_SCHEMA = "P0_LOADERGEN_V1_1"


class DeterministicEpochSampler(torch.utils.data.Sampler):
    """Yields the SAME fixed permutation every epoch (pure function).

    set_epoch(epoch) rebinds the served permutation deterministically.
    """

    def __init__(self, master_seed, dataset_length):
        _validate_master_seed(master_seed)
        self.master_seed = int(master_seed)
        self.dataset_length = validate_dataset_length(dataset_length)
        self.current_epoch = 0
        self._perm = build_permutation(self.master_seed, 0,
                                       self.dataset_length)

    def set_epoch(self, epoch_index):
        validate_epoch_index(epoch_index)
        self.current_epoch = int(epoch_index)
        self._perm = build_permutation(self.master_seed, epoch_index,
                                       self.dataset_length)

    def __iter__(self):
        return iter(self._perm)

    def __len__(self):
        return self.dataset_length


def build_permutation(master_seed, epoch_index, dataset_length):
    """Pure deterministic permutation of range(dataset_length)."""
    _validate_master_seed(master_seed)
    validate_epoch_index(epoch_index)
    validate_dataset_length(dataset_length)
    order_seed = derive_epoch_order_seed(master_seed, epoch_index,
                                         dataset_length, SAMPLER_SCHEMA)
    generator = torch.Generator()
    generator.manual_seed(order_seed)
    return [int(i) for i in torch.randperm(dataset_length,
                                           generator=generator).tolist()]


def _validated_permutation(perm, dataset_length):
    if isinstance(perm, bool) or not isinstance(perm, (list, tuple)):
        raise TypeError("perm must be a list/tuple of plain ints")
    if len(perm) != int(dataset_length):
        raise ValueError(
            "incomplete order sequence: len=%d expected=%d"
            % (len(perm), int(dataset_length)))
    seen = set()
    for idx in perm:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError("ambiguous index input: %r" % (idx,))
        if idx < 0 or idx >= int(dataset_length):
            raise ValueError("index %d outside [0, %d)" % (idx, dataset_length))
        if idx in seen:
            raise ValueError("duplicate index in order sequence: %d" % idx)
        seen.add(idx)
    return [int(i) for i in perm]


def order_hash(perm, master_seed, epoch_index, dataset_length):
    """Stable SHA-256 over the complete validated ordered index sequence.

    Serialization: schema prefix bytes; little-endian signed 64-bit encodings of
    dataset_length and epoch_index; then each index as little-endian signed
    64-bit. No image paths or patient identifiers are hashed. Consuming this
    function does NOT advance any sampler state.
    """
    _validate_master_seed(master_seed)
    validate_epoch_index(epoch_index)
    validate_dataset_length(dataset_length)
    clean = _validated_permutation(perm, dataset_length)
    h = hashlib.sha256()
    h.update(ORDERHASH_SCHEMA.encode("utf-8"))
    h.update(b"|")
    h.update(struct.pack("<q", int(dataset_length)))
    h.update(struct.pack("<q", int(epoch_index)))
    for idx in clean:
        h.update(struct.pack("<q", idx))
    return h.hexdigest()


def expected_epoch_order_hashes(master_seed, epochs, dataset_length):
    """All expected per-epoch order hashes WITHOUT constructing a sampler."""
    return {
        e: order_hash(build_permutation(master_seed, e, dataset_length),
                      master_seed, e, dataset_length)
        for e in range(epochs)
    }


def loader_generator_seed(worker_base_seed, epoch_index):
    payload = "%s|%d|%d" % (LOADERGEN_SCHEMA, int(worker_base_seed),
                            int(epoch_index))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)


def p0_worker_init(worker_id, worker_base_seed=0):
    """Top-level, picklable worker initializer (no closure)."""
    seed = derive_seed(int(worker_base_seed),
                       "dataloader_worker_%d" % int(worker_id))
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def make_paired_dataloader(dataset, master_seed, num_workers=0, batch_size=32):
    """Build an epoch-indexed DataLoader factory for ONE arm.

    - shuffle is ALWAYS False; ordering comes exclusively from the
      deterministic sampler;
    - a fresh explicit CPU torch.Generator is passed via `generator=`,
      derived from the arm's dataloader_worker_base domain and the epoch —
      independent objects per arm with identical values for equal seeds;
    - the worker initializer is a top-level partial (picklable);
    - construction and iteration consume no main-process global Torch RNG.
    """
    _validate_master_seed(master_seed)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) \
            or num_workers < 0:
        raise TypeError("num_workers must be a non-negative plain int")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) \
            or batch_size <= 0:
        raise TypeError("batch_size must be a positive plain int")
    length = validate_dataset_length(len(dataset))
    worker_base_seed = derive_seed(master_seed, "dataloader_worker_base")
    worker_init = functools.partial(p0_worker_init,
                                    worker_base_seed=worker_base_seed)

    class _FixedOrderSampler(torch.utils.data.Sampler):
        """Serves one precomputed deterministic permutation."""

        def __init__(self, permutation):
            self.permutation = list(permutation)

        def __iter__(self):
            return iter(self.permutation)

        def __len__(self):
            return len(self.permutation)

    def _build(epoch_index):
        validate_epoch_index(epoch_index)
        sampler = _FixedOrderSampler(
            build_permutation(master_seed, epoch_index, length))
        loader_gen = torch.Generator()
        loader_gen.manual_seed(loader_generator_seed(worker_base_seed,
                                                     epoch_index))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          sampler=sampler, num_workers=num_workers,
                          worker_init_fn=worker_init, generator=loader_gen,
                          persistent_workers=False)

    return _build
