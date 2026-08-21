"""Canonical generator-state invariants (G0.2 §3).

Pure helpers, no model/dataset/CUDA dependency of their own. Callers pass in
any torch.nn.Module (real or synthetic).
"""
import hashlib

import torch


def canonical_tensor_state_hash(module: torch.nn.Module) -> str:
    """SHA256 over every entry of module.state_dict(), sorted by key.

    Hashes key, dtype, shape, and raw contiguous CPU bytes PER TENSOR
    (never concatenates tensors of different dtype into one buffer).
    Covers both parameters and buffers (state_dict() includes both).
    """
    h = hashlib.sha256()
    sd = module.state_dict()
    for key in sorted(sd.keys()):
        tensor = sd[key]
        h.update(key.encode('utf-8'))
        h.update(str(tensor.dtype).encode('utf-8'))
        h.update(str(tuple(tensor.shape)).encode('utf-8'))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def buffer_only_hash(module: torch.nn.Module) -> str:
    """Cheaper hash over named_buffers() only, for frequent runtime checks."""
    h = hashlib.sha256()
    for name, buf in sorted(module.named_buffers(), key=lambda kv: kv[0]):
        h.update(name.encode('utf-8'))
        h.update(str(buf.dtype).encode('utf-8'))
        h.update(str(tuple(buf.shape)).encode('utf-8'))
        h.update(buf.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_version_signature(module: torch.nn.Module):
    """Sorted (name, param._version, shape, dtype) tuples.

    Cheap in-process mutation detector (torch bumps `_version` on every
    in-place write to a parameter's data). NOT an artifact identity hash —
    `_version` counters are process-local and reset on reload; do not persist
    this as a checkpoint fingerprint.
    """
    sig = []
    for name, p in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        sig.append((name, int(p._version), tuple(p.shape), str(p.dtype)))
    return tuple(sig)


class preserved_eval_forward:
    """Exception-safe context manager for a forward-only, mode-preserving call.

    Usage:
        with preserved_eval_forward(generator):
            out = generator(x)   # generator is .eval() here, no grad tracked

    On exit (normal or exception), the generator's original .training mode is
    restored exactly, and no parameter/buffer is mutated (BatchNorm running
    stats are frozen because the module is in eval() during the forward,
    regardless of whether the caller nests this inside no_grad()).
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self._original_mode = None
        self._inference_ctx = None

    def __enter__(self):
        self._original_mode = self.module.training
        self.module.eval()
        self._inference_ctx = torch.inference_mode()
        self._inference_ctx.__enter__()
        return self.module

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore inference_mode context first (independent of module mode).
        self._inference_ctx.__exit__(exc_type, exc_val, exc_tb)
        # Always restore original training/eval mode, even on exception.
        self.module.train(self._original_mode)
        return False  # never swallow exceptions
