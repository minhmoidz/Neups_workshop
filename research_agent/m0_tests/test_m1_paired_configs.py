"""M1 — paired-config regression test for the frozen C4 development protocol.

Non-vacuous checks that B_dev and C4 dev configs exactly match the values frozen
in M1_C4_PROTOCOL_LOCK.json and are pairwise identical EXCEPT the single
allowed delta (feature_loss_weight / feature-preservation term).

FAILS when:
- a config drifts from the frozen hyperparameters, or
- the two arms differ in any field outside the frozen delta, or
- the frozen protocol lock file itself is missing or unreadable.
"""
import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, HERE)
from m0_common import run_all  # noqa: E402

LOCK = os.path.join(ROOT, 'research_agent', 'M1_C4_PROTOCOL_LOCK.json')
B_DEV = os.path.join(ROOT, 'config_files', 'config_dev_restored_baseline.json')
C4 = os.path.join(ROOT, 'config_files', 'config_dev_c4.json')

# frozen hyperparameters that must match the protocol lock exactly
FROZEN_KEYS = [
    'mu', 'image_size', 'batch_size', 'accumulation_steps', 'optimizer',
    'learning_rate', 'max_epochs', 'ac_loss_weight', 'ver_loss_weight',
    'ac_pos_weight', 'generator_checkpoint_path', 'generator_checkpoint_sha256',
    'generator_initial_checkpoint', 'generator_initial_checkpoint_sha256',
    'classifier_checkpoint_path', 'classifier_checkpoint_sha256',
    'verification_model_path', 'verification_model_sha256', 'seed',
    'checkpoint_selection_rule', 'checkpoint_selection_tiebreak',
    'use_budget_map', 'stochastic_lambda', 'ver_ensemble_size',
    'ver_restart_every', 'ver_warmup_iters', 'mode', 'test_firewall',
]

# per-arm frozen values that may legitimately differ
ARM_KEYS = {'experiment_description', 'model', 'development_mode'}

# the ONLY permitted training-protocol delta between arms
ALLOWED_DELTA = {
    'feature_loss_weight',
    'feature_loss',
    'feature_representation',
    'feature_loss_detach_source',
}


def _load(p):
    if not os.path.exists(p):
        raise RuntimeError('missing file: %s' % p)
    with open(p) as f:
        return json.load(f)


def _pair_check():
    lock = _load(LOCK)
    frozen = lock['frozen_hyperparameters']
    b_dev = _load(B_DEV)
    c4 = _load(C4)

    for k in FROZEN_KEYS:
        if k not in b_dev:
            raise RuntimeError('missing key %s in B_dev config' % k)
        v = b_dev.get(k)
        lv = frozen.get(k)
        if lv is not None and lv != v:
            raise RuntimeError('B_dev config %s=%r != frozen %r' % (k, v, lv))
        if b_dev.get(k) != c4.get(k):
            raise RuntimeError('arm mismatch on %s: %r vs %r' % (k, b_dev.get(k), c4.get(k)))
    # batch/epochs/seed pairing
    if b_dev['batch_size'] != 16 or c4['batch_size'] != 16:
        raise RuntimeError('batch_size must be frozen at 16: %r / %r' % (b_dev['batch_size'], c4['batch_size']))
    if b_dev['max_epochs'] != 250 or c4['max_epochs'] != 250:
        raise RuntimeError('max_epochs must be frozen at 250: %r / %r' % (b_dev['max_epochs'], c4['max_epochs']))
    if b_dev['seed'] != 42 or c4['seed'] != 42:
        raise RuntimeError('S1 seed must be 42: %r / %r' % (b_dev['seed'], c4['seed']))

    # single allowed delta: feature-preservation term
    b_keys, c_keys = set(b_dev), set(c4)
    only_b = b_keys - c_keys
    only_c = c_keys - b_keys
    if only_b:
        raise RuntimeError('keys present only in B_dev: %s' % sorted(only_b))
    if not only_c.issubset(ALLOWED_DELTA):
        raise RuntimeError('keys present only in C4 outside allowed delta: %s' % sorted(only_c - ALLOWED_DELTA))
    diffs = {k for k in b_keys & c_keys if b_dev[k] != c4[k]}
    unexpected = diffs - ALLOWED_DELTA - ARM_KEYS
    if unexpected:
        raise RuntimeError('unexpected arm delta: %s' % sorted(unexpected))
    if 'feature_loss_weight' not in diffs:
        raise RuntimeError('feature_loss_weight must be the arm delta (B_dev=0.0, C4=1.0)')
    if b_dev['feature_loss_weight'] != 0.0 or c4['feature_loss_weight'] != 1.0:
        raise RuntimeError('feature_loss_weight: B_dev=%r C4=%r (want 0.0/1.0)' % (b_dev['feature_loss_weight'], c4['feature_loss_weight']))

    # C4-only feature-term config must be present and sane
    for k in ('feature_loss', 'feature_representation', 'feature_loss_detach_source'):
        if k not in c4 or c4[k] is None:
            raise RuntimeError('C4 missing feature-term key %s' % k)
    if c4['feature_representation'] != 'densenet121_penultimate_pooled_1024':
        raise RuntimeError('C4 feature representation drift: %r' % c4['feature_representation'])
    return True


def t_m1_pairing_frozen():
    return _pair_check()


if __name__ == '__main__':
    ok = run_all([
        ('M1 B_dev/C4 configs paired & frozen', t_m1_pairing_frozen),
    ])
    sys.exit(0 if ok else 1)
