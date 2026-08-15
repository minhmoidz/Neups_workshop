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
SEG_MANIFEST = os.path.join(ROOT, 'research_agent', 'M1_1_SEGMENTATION_PROVENANCE.md')

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


def _m11_gates_check():
    lock = _load(LOCK)
    gates = lock.get('gates')
    if not gates:
        raise RuntimeError('protocol lock missing gates section')

    # S1 validity is separate from S1 scientific promotion
    if 's1_run_validity' not in gates or 's1_scientific_promotion' not in gates:
        raise RuntimeError('S1 validity must be separate from S1 scientific promotion')
    prom = gates['s1_scientific_promotion']

    # S1 privacy ceiling == +0.03 (frozen)
    pn = prom.get('privacy_non_regression', '')
    if '0.03' not in pn.replace(' ', ''):
        raise RuntimeError('S1 privacy ceiling must be +0.03: %r' % pn)
    if not ('AUC_C4_VAL' in pn and 'AUC_Bdev_VAL' in pn):
        raise RuntimeError('S1 privacy gate must compare AUC_C4_VAL vs AUC_Bdev_VAL')

    # S1 classification requirement exists
    cu = prom.get('classification_utility', '')
    if 'macro_AUC_C4_VAL' not in cu or 'macro_AUC_Bdev_VAL' not in cu:
        raise RuntimeError('S1 classification gate must exist (macro_AUC_C4_VAL >= macro_AUC_Bdev_VAL)')

    # segmentation gate: must not silently pass while BLOCKED
    seg = prom.get('segmentation', {})
    if seg.get('never_silently_pass') is not True:
        raise RuntimeError('segmentation must not be silently treated as PASS')
    if seg.get('status') != 'NOT_APPLICABLE_WHILE_BLOCKED':
        raise RuntimeError('segmentation status must be NOT_APPLICABLE_WHILE_BLOCKED')
    if 'Dice_C4' not in seg.get('if_certified_require', ''):
        raise RuntimeError('segmentation if-certified criterion must exist (Dice_C4 >= Dice_Bdev - 0.005)')
    if seg.get('if_certified_require') and '- 0.005' not in seg['if_certified_require']:
        raise RuntimeError('segmentation Dice ceiling must be -0.005')

    # S2 seeds
    seeds = lock['frozen_hyperparameters']['seeds']
    if seeds['s2']['anonymizer'] != [42]:
        raise RuntimeError('S2 anonymizer seeds must be [42] only: %r' % seeds['s2']['anonymizer'])
    if seeds['s2']['attacker'] != [42, 43, 44]:
        raise RuntimeError('S2 attacker seeds must be [42, 43, 44]: %r' % seeds['s2']['attacker'])
    if seeds['s2_reuses_s1_generators'] is not True:
        raise RuntimeError('S2 must reuse frozen S1 generator checkpoints')
    if seeds['s2_anonymizer_requires_retrain'] is not False:
        raise RuntimeError('S2 must NOT retrain anonymizers')

    # B_dev/C4 same anonymizer seed (S1 [42] both)
    if seeds['s1']['anonymizer'] != [42]:
        raise RuntimeError('S1 anonymizer seed must be [42]')

    # S2 promotion rule
    s2 = gates['s2']
    if '0.03' not in s2.get('privacy_promotion_ceiling', '').replace(' ', ''):
        raise RuntimeError('S2 privacy promotion ceiling must be +0.03')
    if s2.get('anonymizer_seeds') != [42]:
        raise RuntimeError('S2 anonymizer seeds must be [42]')
    if s2.get('attacker_seeds') != [42, 43, 44]:
        raise RuntimeError('S2 attacker seeds must be [42, 43, 44]')
    if s2.get('reuses_s1_generators') is not True or s2.get('no_anonymizer_retrain') is not True:
        raise RuntimeError('S2 must reuse S1 generators and not retrain anonymizers')
    defn = s2.get('Delta_priv_definition', '')
    if 'AUC_C4_VAL' not in defn or 'AUC_Bdev_VAL' not in defn:
        raise RuntimeError('S2 Delta_priv definition must compare AUC_C4_VAL vs AUC_Bdev_VAL')

    # C2/C3 OFF, mu frozen, TEST closed (config + lock)
    for cfg_path in (B_DEV, C4):
        cfg = _load(cfg_path)
        if cfg['ver_ensemble_size'] != 1 or cfg['ver_restart_every'] != 0 or cfg['ver_warmup_iters'] != 0:
            raise RuntimeError('%s: C3 not OFF' % cfg_path)
        if cfg['use_budget_map'] is not False:
            raise RuntimeError('%s: C2 budget map must be off' % cfg_path)
        if cfg['mu'] != 0.01:
            raise RuntimeError('%s: mu must be 0.01' % cfg_path)
        if cfg['test_firewall'] != 'CLOSED':
            raise RuntimeError('%s: TEST must be CLOSED' % cfg_path)
    c2c3 = lock['frozen_hyperparameters']['c2_c3_off']
    if c2c3['ver_ensemble_size'] != 1 or c2c3['ver_restart_every'] != 0 or c2c3['ver_warmup_iters'] != 0 or c2c3['use_budget_map'] is not False:
        raise RuntimeError('lock c2_c3_off drift')
    if lock['frozen_hyperparameters']['mu'] != 0.01:
        raise RuntimeError('lock mu drift')

    # GPU cost estimate present and consistent (no hidden factorial expansion)
    cost = lock.get('gpu_cost_estimate')
    if not cost:
        raise RuntimeError('GPU cost estimate must be present')
    s1_total = cost['s1']['s1_total_hours']
    s2_incr = cost['s2_incremental']['s2_incremental_hours']
    total = cost['total_through_s2_hours']
    if not (s1_total > 0 and s2_incr >= 0 and abs(total - (s1_total + s2_incr)) < 1e-6):
        raise RuntimeError('GPU cost estimate inconsistent: %r %r %r' % (s1_total, s2_incr, total))

    # protocol version 1.2.0 + M1.1 / M1.2 reasons recorded
    if lock.get('version') not in ('1.1.0', '1.2.0'):
        raise RuntimeError('protocol version must be 1.2.0: %r' % lock.get('version'))
    if not lock.get('m1_1_reason') or not lock.get('m1_2_reason'):
        raise RuntimeError('M1.1 and M1.2 reasons must be recorded')

    # segmentation manifest must exist and must not claim certification while BLOCKED
    if not os.path.exists(SEG_MANIFEST):
        raise RuntimeError('M1_1_SEGMENTATION_PROVENANCE.md missing')
    with open(SEG_MANIFEST) as f:
        manifest = f.read()
    if 'SEGMENTATION_STILL_BLOCKED' not in manifest:
        raise RuntimeError('segmentation manifest must record STILL_BLOCKED')
    if 'SEGMENTATION_CERTIFIED' in manifest:
        raise RuntimeError('segmentation manifest must not claim CERTIFIED while BLOCKED')
    return True


def t_m11_gates_frozen():
    return _m11_gates_check()


if __name__ == '__main__':
    ok = run_all([
        ('M1 B_dev/C4 configs paired & frozen', t_m1_pairing_frozen),
        ('M1.1 S1/S2 gates, seeds, cost, segmentation frozen', t_m11_gates_frozen),
    ])
    sys.exit(0 if ok else 1)
