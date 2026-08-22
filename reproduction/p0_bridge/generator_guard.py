"""Frozen-generator state guard for attacker-training forwards.

Runs a supplied deformation callable under torch.no_grad() and FAILS CLOSED if
any generator state mutates. torch.inference_mode() is deliberately NOT used:
inference tensors cannot participate in the downstream attacker's autograd
graph, while no_grad tensors can.
"""
import torch


class GeneratorStateMutationError(RuntimeError):
    """Raised when a protected forward mutates frozen generator state."""


def _training_topology(module):
    return [(name, bool(sub.training))
            for name, sub in sorted(module.named_modules())]


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
    """Fail-closed pre-checks: eval mode + all parameters requires_grad=False."""
    if generator.training:
        raise GeneratorStateMutationError(
            "generator must be in .eval() mode before protected forward")
    trainable = [n for n, p in generator.named_parameters() if p.requires_grad]
    if trainable:
        raise GeneratorStateMutationError(
            "generator parameters must have requires_grad=False; trainable: %s"
            % sorted(trainable)[:8])


def protected_forward(generator, callable_fn, *args, **kwargs):
    """Run `callable_fn(*args, **kwargs)` under no_grad with state verification.

    Detects (never silently restores) any mutation of training flags or buffers
    and raises GeneratorStateMutationError. Returns an ordinary no_grad tensor
    usable in the downstream attacker's autograd graph.
    """
    assert_generator_frozen_state(generator)
    topology_before = _training_topology(generator)
    buffers_before = _buffer_snapshot(generator)

    with torch.no_grad():
        output = callable_fn(*args, **kwargs)

    topology_after = _training_topology(generator)
    if topology_after != topology_before:
        changed = [n for (n, a), (_, b) in zip(topology_before, topology_after)
                   if a != b]
        raise GeneratorStateMutationError(
            "submodule training flags changed across protected forward: %s"
            % changed[:8])

    buffers_after = _buffer_snapshot(generator)
    mutated = [name for name in buffers_before
               if buffers_after[name] != buffers_before[name]]
    if mutated:
        raise GeneratorStateMutationError(
            "BatchNorm/buffer state changed across protected forward: %s"
            % sorted(mutated)[:8])

    if isinstance(output, torch.Tensor) and output.is_inference():
        raise GeneratorStateMutationError(
            "protected forward produced an inference tensor; this would break "
            "downstream attacker backward")
    return output


def canonical_model_state_hash(module):
    """Versioned deterministic hash of parameters+buffers.

    Sorted names; explicit dtype; explicit shape; contiguous CPU bytes; no
    pickle; independent of dict insertion order.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(b"P0_MODELSTATE_V1|")
    entries = []
    for name, tensor in module.state_dict(keep_vars=True).items():
        entries.append((str(name), str(tensor.dtype), tuple(tensor.shape),
                        tensor.detach().cpu().contiguous().numpy().tobytes()))
    for name, dtype, shape, blob in sorted(entries):
        h.update(name.encode("utf-8")); h.update(b"|")
        h.update(dtype.encode("utf-8")); h.update(b"|")
        h.update(repr(shape).encode("utf-8")); h.update(b"|")
        h.update(blob)
    return h.hexdigest()
