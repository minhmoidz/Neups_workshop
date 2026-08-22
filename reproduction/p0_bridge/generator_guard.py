"""Frozen-generator state guard for attacker-training forwards.

Revision P0_2_1 (external source-review closeout):
- rejects training=True on ANY submodule (nested Dropout/BatchNorm included);
- detects added, removed and mutated buffers;
- recursively rejects inference tensors in nested outputs;
- binds the callable as callable_fn(generator, *args);
- model-state hash schema bumped to P0_MODELSTATE_V1_1 with length-prefixed
  serialization and explicit sparse/meta rejection.

torch.no_grad() only — never torch.inference_mode() for attacker-consumed
outputs.
"""
import hashlib
import struct

import torch


class GeneratorStateMutationError(RuntimeError):
    """Raised when a protected forward mutates frozen generator state."""


def _training_topology(module):
    return [(name, bool(sub.training))
            for name, sub in sorted(module.named_modules())]


def _assert_all_submodules_eval(module):
    offenders = [name for name, flag in _training_topology(module) if flag]
    if offenders:
        raise GeneratorStateMutationError(
            "generator has submodule(s) with training=True: %s"
            % offenders[:8])


def _param_snapshot(module):
    """Efficient parameter identity snapshot (no GPU->CPU byte copies).

    Tracks per-parameter: storage pointer, mutation counter (_version),
    dtype and shape. In-place mutation bumps _version; replacement changes
    the tensor object (data_ptr/shape/dtype); addition/removal changes the
    name set. Full byte-level hashing remains available at trajectory
    boundaries via canonical_model_state_hash.
    """
    snap = {}
    for name, p in module.named_parameters():
        snap[name] = (int(p.data_ptr()), int(getattr(p, "_version", 0)),
                      str(p.dtype), tuple(p.shape), bool(p.requires_grad))
    return snap


def _buffer_snapshot(module):
    snap = {}
    for name, buf in module.named_buffers():
        snap[name] = (
            str(buf.dtype),
            tuple(buf.shape),
            buf.detach().cpu().contiguous().numpy().tobytes(),
        )
    return snap


def assert_generator_frozen_state(generator):
    """Fail-closed pre-checks: every submodule eval + all params frozen."""
    _assert_all_submodules_eval(generator)
    trainable = [n for n, p in generator.named_parameters() if p.requires_grad]
    if trainable:
        raise GeneratorStateMutationError(
            "generator parameters must have requires_grad=False; trainable: %s"
            % sorted(trainable)[:8])


def _assert_no_inference_tensors(obj, path="output"):
    """Recursively reject inference tensors inside nested structures."""
    if isinstance(obj, torch.Tensor):
        if obj.is_inference():
            raise GeneratorStateMutationError(
                "inference tensor found at %s; it could not participate in a "
                "downstream attacker backward" % path)
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _assert_no_inference_tensors(item, "%s[%d]" % (path, i))
    elif isinstance(obj, dict):
        for key, item in obj.items():
            _assert_no_inference_tensors(item, "%s[%r]" % (path, key))


def protected_forward(generator, callable_fn, *args, **kwargs):
    """Run `callable_fn(generator, *args, **kwargs)` under no_grad, verified.

    The callable is explicitly bound to the supplied generator so an unrelated
    module cannot be forwarded by accident. Detects (never silently restores)
    any mutation of training flags or buffers, including added/removed buffers,
    and raises GeneratorStateMutationError. Returns ordinary no_grad outputs
    usable in the downstream attacker's autograd graph.
    """
    if not callable(callable_fn):
        raise TypeError("callable_fn must be callable")
    assert_generator_frozen_state(generator)
    topology_before = _training_topology(generator)
    buffers_before = _buffer_snapshot(generator)
    params_before = _param_snapshot(generator)

    with torch.no_grad():
        output = callable_fn(generator, *args, **kwargs)

    params_after = _param_snapshot(generator)
    p_added = sorted(set(params_after) - set(params_before))
    p_removed = sorted(set(params_before) - set(params_after))
    p_changed = [n for n in set(params_before) & set(params_after)
                 if params_after[n] != params_before[n]]
    param_problems = []
    if p_added:
        param_problems.append("added=%s" % p_added[:4])
    if p_removed:
        param_problems.append("removed=%s" % p_removed[:4])
    if p_changed:
        param_problems.append("mutated_or_replaced=%s" % sorted(p_changed)[:4])
    if param_problems:
        raise GeneratorStateMutationError(
            "frozen generator PARAMETERS changed across protected forward "
            "(in-place mutation or replacement): %s" % "; ".join(param_problems))

    if _training_topology(generator) != topology_before:
        before = dict(topology_before)
        after = dict(_training_topology(generator))
        changed = [n for n in set(before) | set(after)
                   if before.get(n) != after.get(n)]
        raise GeneratorStateMutationError(
            "submodule training flags changed across protected forward: %s"
            % sorted(changed)[:8])

    buffers_after = _buffer_snapshot(generator)
    added = sorted(set(buffers_after) - set(buffers_before))
    removed = sorted(set(buffers_before) - set(buffers_after))
    mutated = [n for n in set(buffers_before) & set(buffers_after)
               if buffers_after[n] != buffers_before[n]]
    problems = []
    if added:
        problems.append("added=%s" % added[:4])
    if removed:
        problems.append("removed=%s" % removed[:4])
    if mutated:
        problems.append("mutated=%s" % sorted(mutated)[:4])
    if problems:
        raise GeneratorStateMutationError(
            "frozen generator state changed across protected forward: %s"
            % "; ".join(problems))

    _assert_no_inference_tensors(output)
    return output


def canonical_model_state_hash(module):
    """P0_MODELSTATE_V1_1 deterministic hash of parameters+buffers.

    Length-prefixed fields over sorted entries: name, dtype, shape, byte count,
    dense contiguous CPU bytes. No pickle; independent of insertion order;
    unsupported sparse/meta tensors are rejected explicitly.
    """
    h = hashlib.sha256()
    h.update(b"P0_MODELSTATE_V1_1|")
    h.update(struct.pack("<q", 0))  # reserved version field

    def emit(field):
        blob = field.encode("utf-8") if isinstance(field, str) else field
        h.update(struct.pack("<q", len(blob)))
        h.update(blob)

    entries = []
    for name, tensor in module.state_dict(keep_vars=True).items():
        if tensor.is_sparse or tensor.layout in (torch.sparse_coo,
                                                 torch.sparse_csr):
            raise TypeError("unsupported sparse tensor in state: %r" % name)
        if tensor.device.type == "meta":
            raise TypeError("unsupported meta tensor in state: %r" % name)
        dense = tensor.detach().cpu().contiguous()
        entries.append((str(name), str(dense.dtype), tuple(dense.shape),
                        dense.numpy().tobytes()))
    for name, dtype, shape, blob in sorted(entries):
        emit(name)
        emit(dtype)
        emit(repr(shape))
        emit(blob)
    return h.hexdigest()
