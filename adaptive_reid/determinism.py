"""Per-arm determinism check (STEP 2B Part 11).

For a deterministic anonymizer: generating the same fixed input twice with the same
checkpoint/config must give bit-identical output. If the outputs differ, the arm is
identified as stochastic.

STEP 2B implement + unit-test only; no full real-data run.
"""

from typing import Callable


def check_deterministic(generate: Callable, *args, **kwargs):
    """Run ``generate`` twice on identical inputs and compare outputs bit-for-bit.

    :param generate: callable that returns a torch.Tensor.
    :return: dict {'deterministic': bool, 'same': bool, 'max_abs_diff': float}
    """
    out1 = generate(*args, **kwargs)
    out2 = generate(*args, **kwargs)

    import torch
    if not isinstance(out1, torch.Tensor) or not isinstance(out2, torch.Tensor):
        raise TypeError('determinism checker expects torch.Tensor outputs')

    same = torch.equal(out1.cpu(), out2.cpu())
    max_abs_diff = float((out1.cpu() - out2.cpu()).abs().max()) if out1.numel() else 0.0
    return {
        'deterministic': bool(same),
        'same': bool(same),
        'max_abs_diff': max_abs_diff,
    }


def arm_is_stochastic(result: dict) -> bool:
    """Interpret a determinism-check result: an arm is stochastic iff outputs differ."""
    return not result['same']