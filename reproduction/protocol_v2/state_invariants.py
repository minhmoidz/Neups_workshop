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


def _snapshot_mode_vector(module: torch.nn.Module):
    """Per-submodule (module, was_training) list, in module.modules() order
    (includes `module` itself). Captures the FULL mode topology, not just the
    top-level flag — a parent and its children can legitimately disagree
    (e.g. someone previously called child.eval() without affecting the
    parent)."""
    return [(m, m.training) for m in module.modules()]


def _restore_mode_vector(mode_vector):
    """Restore each module's individual .training flag directly (bypassing
    .train()/.eval(), which recurse into children and would overwrite a
    child's restored flag with the parent's when applied top-down)."""
    for m, was_training in mode_vector:
        m.training = was_training


class preserved_eval_forward:
    """Exception-safe context manager for a forward-only, mode-preserving call
    whose OUTPUT remains usable in a later autograd graph (e.g. feeding a
    trainable critic that will be backpropagated).

    Usage:
        with preserved_eval_forward(generator):
            out = generator(x)   # generator (and every submodule) is .eval()
                                  # here; `out` is an ordinary no_grad tensor,
                                  # not an inference-mode tensor.

    Fix 1 (G0.2A.2): uses torch.no_grad(), not torch.inference_mode(). A
    tensor created under inference_mode() is an "inference tensor" that
    PyTorch autograd refuses to use in any graph that will later be
    backpropagated (e.g. training a critic on the generator's fake output)
    — using it there raises a RuntimeError. no_grad() tensors carry no such
    restriction: they are ordinary tensors that simply weren't tracked
    during their own creation, and are fully usable as fresh inputs to a
    downstream graph that DOES require grad.

    Fix 2 (G0.2A.2): restores the exact per-submodule .training topology
    that existed before entry (not just the top-level flag) — see
    `_snapshot_mode_vector`/`_restore_mode_vector`. On exit (normal or
    exception), every submodule's original mode is restored exactly, and no
    parameter/buffer is mutated (BatchNorm running stats are frozen because
    every submodule is in eval() during the forward).
    """

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self._mode_vector = None
        self._no_grad_ctx = None

    def __enter__(self):
        self._mode_vector = _snapshot_mode_vector(self.module)
        self.module.eval()
        self._no_grad_ctx = torch.no_grad()
        self._no_grad_ctx.__enter__()
        return self.module

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore no_grad context first (independent of module mode).
        self._no_grad_ctx.__exit__(exc_type, exc_val, exc_tb)
        # Always restore the exact original per-submodule mode topology,
        # even on exception.
        _restore_mode_vector(self._mode_vector)
        return False  # never swallow exceptions
