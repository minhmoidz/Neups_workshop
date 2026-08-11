"""Weights-actually-updated detection (STEP 2B Part 4).

Detects the 'optimizer never executed' failure class robustly and independently of
training/test performance: deterministic checksum of the trainable parameters taken
before and after training, compared exactly.
"""

import hashlib
import os

import torch


def parameters_bytes(net):
    """Concatenate the float32 bytes of every trainable parameter of ``net``.

    Order is fixed by iterating ``net.parameters()`` (module registration order), so
    two calls on the same architecture/initialization produce identical bytes.

    :return: bytes
    """
    parts = []
    for p in net.parameters():
        if not p.requires_grad:
            continue
        data = p.detach().cpu().contiguous()
        parts.append(data.numpy().tobytes())
    if not parts:
        raise ValueError("model has no trainable parameters")
    return b''.join(parts)


def parameters_hash(net, algo='sha256'):
    """Deterministic hash of the trainable parameters of ``net``.

    :param net: torch.nn.Module
    :param algo: str, hashlib algorithm name
    :return: str hex digest
    """
    h = hashlib.new(algo)
    h.update(parameters_bytes(net))
    return h.hexdigest()


def weights_changed(before_hash, after_hash):
    """Return True iff the parameter hash changed between the two snapshots."""
    return before_hash != after_hash


def snapshot_parameters(net):
    """Take a hash snapshot of the current trainable parameters.

    Convenience so callers (and tests) only hold a digest, not a model reference.
    """
    return parameters_hash(net)


def initialized_weights_identical(net_a, net_b):
    """Exact element-wise comparison of trainable parameters of two models.

    :return: bool True if every trainable parameter is exactly identical.
    """
    pa = [p.detach() for p in net_a.parameters() if p.requires_grad]
    pb = [p.detach() for p in net_b.parameters() if p.requires_grad]
    if len(pa) != len(pb):
        return False
    for a, b in zip(pa, pb):
        if a.shape != b.shape:
            return False
        if not torch.equal(a.cpu(), b.cpu()):
            return False
    return True


def checkpoint_loadable(path):
    """Return True iff a real PyTorch checkpoint at ``path`` can actually be loaded.

    Must genuinely attempt the load (protocol R-2: ``checkpoint_loadable`` reflects
    reality, not the schema). ``weights_only=False`` matches how the repo saves state
    dicts (plain state_dict) and accepts older torch formats; loading is done on CPU so
    the check does not need a GPU.
    """
    if not path or not os.path.exists(path):
        return False
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
        return isinstance(state, dict)
    except Exception:
        return False