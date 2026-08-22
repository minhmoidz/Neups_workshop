"""P0_SEED_V1 domain-separated deterministic seed derivation.

Pure function of (master_seed, domain). No global RNG state of any kind
(Python hash(), random, numpy, torch) is consulted. Output is constrained to
the valid non-negative signed 63-bit PyTorch seed range [0, 2**63 - 1].

Serialization contract (exact UTF-8 bytes, "|" delimiters):
    message = "P0_SEED_V1|" + str(master_seed) + "|" + domain
    derived = big-endian int of sha256(message)[0:8]  mod  2**63

Revision P0_2_1: adds a real per-arm seed-bundle constructor and hardens
input validation.
"""
import hashlib
import random
import re

import numpy as np
import torch

SCHEMA = "P0_SEED_V1"
_MAX_63BIT = 2 ** 63 - 1
_DOMAIN_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")

DOMAINS = (
    "attacker_weight_init",
    "train_order",
    "dataloader_worker_base",
    "statistical_sensitivity",
)


def _validate_master_seed(master_seed):
    if isinstance(master_seed, bool) or not isinstance(master_seed, int):
        raise TypeError("master_seed must be a plain int, got %r" % type(master_seed))
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative, got %d" % master_seed)
    return master_seed


def _validate_domain(domain):
    if not isinstance(domain, str) or not _DOMAIN_RE.match(domain):
        raise ValueError(
            "domain must match ^[A-Za-z0-9_]{1,64}$, got %r" % (domain,))
    return domain


def validate_epoch_index(epoch_index):
    if isinstance(epoch_index, bool) or not isinstance(epoch_index, int):
        raise TypeError("epoch_index must be a plain int")
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    return epoch_index


def validate_dataset_length(dataset_length):
    if isinstance(dataset_length, bool) or not isinstance(dataset_length, int):
        raise TypeError("dataset_length must be a plain int")
    if dataset_length <= 0:
        raise ValueError("dataset_length must be positive")
    return dataset_length


def derive_seed(master_seed, domain):
    """Deterministic 63-bit sub-seed for (master_seed, domain)."""
    _validate_master_seed(master_seed)
    _validate_domain(domain)
    message = ("%s|%d|%s" % (SCHEMA, master_seed, domain)).encode("utf-8")
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], "big") % (_MAX_63BIT + 1)


def train_order_seed(master_seed):
    """Locked-domain helper: the train_order sub-seed for a master seed."""
    return derive_seed(master_seed, "train_order")


def derive_epoch_order_seed(master_seed, epoch_index, dataset_length,
                            sampler_schema="P0_SAMPLER_V1_1"):
    """Epoch-specific permutation seed; pure function of the locked inputs.

    Binds sampler schema + derived train_order seed + epoch index + length.
    """
    _validate_master_seed(master_seed)
    validate_epoch_index(epoch_index)
    validate_dataset_length(dataset_length)
    payload = "%s|%d|%d|%d" % (
        sampler_schema, train_order_seed(master_seed), epoch_index,
        dataset_length)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (_MAX_63BIT + 1)


def build_arm_seed_bundle(master_seed):
    """Independently construct the full derived-seed bundle for ONE arm.

    Each call returns a NEW dictionary object; two arms calling this with the
    same master seed get equal-valued but identity-distinct bundles.
    """
    _validate_master_seed(master_seed)
    bundle = {}
    for domain in DOMAINS:
        bundle[domain] = derive_seed(master_seed, domain)
    bundle["statistical_sensitivity_master"] = None  # set once globally elsewhere
    return bundle


def seed_everything_for_attacker_construction(weight_seed):
    """Seed Python, NumPy and CPU torch immediately before attacker construction.

    Contract: call this immediately before constructing the attacker network so
    that identical weight seeds yield byte-identical initializations across
    generator arms. CUDA is NEVER seeded here; torch.cuda is not touched.
    """
    _validate_master_seed(weight_seed)
    random.seed(weight_seed)
    np.random.seed(weight_seed % (2 ** 32))
    torch.manual_seed(weight_seed)
