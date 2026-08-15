"""Shared test utilities for M0. All tests must run on CPU with tiny synthetic
tensors and must NOT open TEST files or load real checkpoints that would be
modified. Set M0_TESTS_ALLOW_CUDA=1 to allow CUDA; default is CPU-only."""
import os
import sys

ALLOW_CUDA = os.environ.get('M0_TESTS_ALLOW_CUDA', '0') == '1'


def device():
    import torch
    return torch.device('cuda') if (ALLOW_CUDA and torch.cuda.is_available()) else torch.device('cpu')


def report(name, passed, detail=''):
    status = 'PASS' if passed else 'FAIL'
    print('%-8s %s  %s' % (status, name, detail))
    return passed


def run_all(tests):
    """tests: list of (name, callable). Runs in order, prints PASS/FAIL. Returns all-pass bool."""
    results = []
    for name, fn in tests:
        try:
            ok = bool(fn())
            results.append((name, ok, ''))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, 'EXC: %r' % e))
    all_ok = all(ok for _, ok, _ in results)
    for name, ok, detail in results:
        report(name, ok, detail)
    return all_ok