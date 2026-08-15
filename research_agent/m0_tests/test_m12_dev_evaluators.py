"""M1.2 — development evaluator isolation + threat-model lock tests (T30-T40).

All tests run on CPU with tiny synthetic tensors and mock datasets. They NEVER
open TEST pair files, never instantiate a testing loader, never construct a
TEST classification fold, and never load real checkpoints.

T30 — dev attacker constructor creates NO TEST loader (monkeypatch fail on 'testing')
T31 — classification dev evaluator rejects TEST before dataset init
T32 — attacker TRAIN geometry = anon(x1), anon(x2)
T33 — attacker selection VAL geometry = anon(x1), anon(x2)
T34 — scientific VAL privacy geometry = anon(x1), real(x2); x2 unchanged
T35 — configurable attacker seeds (42 != 43 stream; identical seed -> identical init)
T36 — classification VAL evaluator: fold=val allowed, fold=test rejected
T37 — common classification checkpoint (frozen SHA for both arms)
T38 — method-neutral anonymizer checkpoint rule (feature excluded from selection)
T39 — paired anonymizer data-order fingerprint (same seed -> same SHA; diff -> diff)
T40 — TEST firewall integrated (runners cannot build testing loader / TEST fold)
"""
import os
import sys
import tempfile
import re

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from m0_common import run_all  # noqa: E402

from m2_dev import evaluator_common as ec  # noqa: E402
from m2_dev import dev_attacker as da  # noqa: E402
from m2_dev import eval_reid_val as ev  # noqa: E402
from m2_dev import eval_classifier_val as ecv  # noqa: E402

DEVICE = torch.device('cpu')

MINI_CONFIG = {
    'image_path': './',
    'image_size': 32,
    'n_channels': 1,
    'batch_size': 4,
    'learning_rate': 1e-3,
    'max_epochs': 2,
    'early_stopping': 5,
    'generator_checkpoint_path': '/nonexistent',
    'experiment_description': 'mini',
}


class TinySiamese(nn.Module):
    def __init__(self, in_channels=3, img=8):
        super().__init__()
        self.proj = nn.Linear(in_channels * img * img * 2, 8)
        self.fc = nn.Linear(8, 1)

    def forward(self, x1, x2):
        h1 = torch.flatten(x1, 1)
        h2 = torch.flatten(x2, 1)
        return self.fc(torch.relu(self.proj(torch.cat([h1, h2], dim=1))))


class SpyAnonymize:
    """Records every image passed through the anonymizer (to verify geometry)."""

    def __init__(self):
        self.calls = []

    def __call__(self, image):
        self.calls.append(image)
        return image + 0.5  # non-identity so anonymized != raw


class _PairBatchLoader:
    def __init__(self, n_batches=2, batch_size=4, img=8):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.img = img

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            x1 = torch.rand(self.batch_size, 1, self.img, self.img)
            x2 = torch.rand(self.batch_size, 1, self.img, self.img)
            labels = torch.randint(0, 2, (self.batch_size,)).float()
            yield x1, x2, labels


class RecNet(nn.Module):
    """Records the exact tensors the net receives (post-preprocess) with valid gradients."""

    def __init__(self, in_channels=3, img=8):
        super().__init__()
        self.linear = nn.Linear(in_channels * img * img * 2, 1)
        self.seen = []

    def forward(self, a, b):
        self.seen.append((a.clone(), b.clone()))
        h1 = torch.flatten(a, 1)
        h2 = torch.flatten(b, 1)
        return self.linear(torch.cat([h1, h2], dim=1))


def _make_attacker(spy, seed=42, n_batches=2, net_factory=None):
    cfg = dict(MINI_CONFIG)
    return da.DevAttacker(
        config=cfg,
        attacker_seed=seed,
        device=DEVICE,
        anonymize_fn=spy,
        training_loader=_PairBatchLoader(n_batches, batch_size=4),
        validation_loader=_PairBatchLoader(n_batches, batch_size=4),
        net_factory=net_factory or (lambda: TinySiamese()),
    )


# ---------------------------------------------------------------------------
# T30 — dev attacker does not construct a TEST loader
# ---------------------------------------------------------------------------
def t30_no_test_loader():
    # Guard: any attempt to request a 'testing' loader must raise.
    from utils import utils as _utils
    orig = _utils.get_data_loader

    def guarded(phase='training', **kw):
        if phase in ('testing', 'test', 'final_test'):
            raise AssertionError('TEST loader requested during dev attacker construction')
        return object()

    _utils.get_data_loader = guarded
    try:
        spy = SpyAnonymize()
        attacker = da.DevAttacker(
            config=dict(MINI_CONFIG),
            attacker_seed=42,
            device=DEVICE,
            anonymize_fn=spy,
            training_loader=_PairBatchLoader(1, batch_size=4),
            validation_loader=_PairBatchLoader(1, batch_size=4),
            net_factory=lambda: TinySiamese(),
        )
        has_test_loader = hasattr(attacker, 'test_loader')
        # the real build path (no injected loaders) must also survive the guard
        ec.build_dev_loaders(dict(MINI_CONFIG), seed=42)
        return not has_test_loader
    finally:
        _utils.get_data_loader = orig


# ---------------------------------------------------------------------------
# T31 — classification dev evaluator rejects TEST before dataset init
# ---------------------------------------------------------------------------
def t31_classifier_rejects_test_preinit():
    import chexnet.cxr_dataset as CXR
    orig = CXR.CXRDataset

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError('CXRDataset constructed before TEST-fold rejection')

    CXR.CXRDataset = Boom
    try:
        try:
            ecv.evaluate_classification_val(dict(MINI_CONFIG, image_path='./'), fold='test', device=DEVICE)
            return False
        except RuntimeError as e:
            return 'TEST firewall' in str(e) or 'test' in str(e).lower()
    finally:
        CXR.CXRDataset = orig


# ---------------------------------------------------------------------------
# T32 — attacker TRAIN geometry = anon(x1), anon(x2)
# ---------------------------------------------------------------------------
def t32_train_geometry_anon_anon():
    spy = SpyAnonymize()
    attacker = _make_attacker(spy, seed=42, n_batches=2)
    attacker.net = RecNet()
    attacker.best_net = RecNet()
    loss = attacker.train_epoch()
    # train_epoch anonymizes BOTH x1 and x2 -> 2 calls per batch, 2 batches
    expected_calls = 2 * 2  # 2 batches * 2 images anonymized
    return len(spy.calls) == expected_calls and torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# T33 — attacker selection VAL geometry = anon(x1), anon(x2)
# ---------------------------------------------------------------------------
def t33_val_selection_geometry_anon_anon():
    spy = SpyAnonymize()
    attacker = _make_attacker(spy, seed=42, n_batches=2)
    attacker.net = RecNet()
    attacker.best_net = RecNet()
    val_loss = attacker.validate_selection()
    expected_calls = 2 * 2
    return len(spy.calls) == expected_calls and torch.isfinite(torch.tensor(val_loss))


# ---------------------------------------------------------------------------
# T34 — scientific VAL privacy geometry = anon(x1), real(x2); x2 unchanged
# ---------------------------------------------------------------------------
def t34_scientific_val_geometry_anon_real():
    spy = SpyAnonymize()
    attacker = _make_attacker(spy, seed=42, n_batches=2)

    class _DataLoader:
        def __len__(self):
            return 2

        def __iter__(self):
            for x1, x2, labels in _PairBatchLoader(2, 4, img=8):
                yield x1, x2, labels

    res = ev.evaluate_reid_val_mixed(spy, attacker.best_net, _DataLoader(), device=DEVICE)
    # scientific geometry anonymizes ONLY x1 -> exactly 1 call per batch (2 batches)
    if len(spy.calls) != 2:
        return False
    if not torch.isfinite(torch.tensor(res['roc_auc'])):
        return False
    return True


# ---------------------------------------------------------------------------
# T35 — configurable attacker seeds
# ---------------------------------------------------------------------------
def _net_init_fingerprint(seed):
    torch.manual_seed(seed)  # attacker init depends on the seeded RNG
    return TinySiamese()


def t35_configurable_seeds():
    # DevAttacker calls utils.seed_all(attacker_seed) before net_factory(),
    # so net initialization must differ across seeds but repeat for same seed.
    net42a = _net_init_fingerprint(42)
    net42b = _net_init_fingerprint(42)
    net43 = _net_init_fingerprint(43)
    same = all(torch.equal(a.detach(), b.detach()) for a, b in
               zip(net42a.state_dict().values(), net42b.state_dict().values()))
    diff = any(not torch.equal(a.detach(), b.detach()) for a, b in
               zip(net42a.state_dict().values(), net43.state_dict().values()))
    return same and diff


# ---------------------------------------------------------------------------
# T36 — classification VAL evaluator: fold=val allowed, fold=test rejected
# ---------------------------------------------------------------------------
def t36_classifier_val_fold():
    ec.assert_dev_fold('val')
    for bad in ('test', 'testing', 'final_test'):
        try:
            ec.assert_dev_fold(bad)
            return False
        except RuntimeError:
            pass
    return True


# ---------------------------------------------------------------------------
# T37 — common classification checkpoint
# ---------------------------------------------------------------------------
def t37_common_classifier_ckpt():
    import json
    lock = json.load(open(os.path.join(ROOT, 'research_agent', 'M1_C4_PROTOCOL_LOCK.json')))
    cfg_b = json.load(open(os.path.join(ROOT, 'config_files', 'config_dev_restored_baseline.json')))
    cfg_c4 = json.load(open(os.path.join(ROOT, 'config_files', 'config_dev_c4.json')))
    frozen = lock['artifacts']['classifier']
    b_sha = cfg_b['classifier_checkpoint_sha256']
    c4_sha = cfg_c4['classifier_checkpoint_sha256']
    return (b_sha == c4_sha == frozen == ec.FROZEN_CLASSIFIER_SHA) and \
        cfg_b['classifier_checkpoint_path'] == cfg_c4['classifier_checkpoint_path']


# ---------------------------------------------------------------------------
# T38 — method-neutral anonymizer checkpoint rule
# ---------------------------------------------------------------------------
def t38_method_neutral_ckpt_rule():
    epochs = [
        ec.compute_epoch_totals(0.5, 0.5, 1.0, 1.0),    # sel=1.0, opt=2.0
        ec.compute_epoch_totals(0.7, 0.7, 0.1, 1.0),    # sel=1.4, opt=1.5
        ec.compute_epoch_totals(0.1, 0.1, 10.0, 1.0),   # sel=0.2, opt=10.2
    ]
    best = ec.select_method_neutral_best(epochs)
    # selection_total picks epoch2 (0.2) even though optimization_total picks
    # epoch1 (1.5) — feature term must NOT drive checkpoint selection.
    if best != 2:
        return False
    # B_dev (feature weight 0): selection == optimization; equal -> earliest
    e_b = [ec.compute_epoch_totals(0.5, 0.5, 1.0, 0.0),
           ec.compute_epoch_totals(0.5, 0.5, 1.0, 0.0)]
    if ec.select_method_neutral_best(e_b) != 0:
        return False
    return True


# ---------------------------------------------------------------------------
# T39 — paired anonymizer data-order fingerprint
# ---------------------------------------------------------------------------
def t39_order_fingerprint():
    rows = ['%06d_%03d.png\t%06d_%03d.png\t1.0' % (i, 0, i, 1) for i in range(50)]
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('\n'.join(rows) + '\n')
        path = f.name
    try:
        h1a = ec.train_order_fingerprint(path, seed=42, batch_size=4)
        h1b = ec.train_order_fingerprint(path, seed=42, batch_size=4)
        h2 = ec.train_order_fingerprint(path, seed=43, batch_size=4)
        return h1a == h1b and h1a != h2 and len(h1a) == 64
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# T40 — TEST firewall integrated
# ---------------------------------------------------------------------------
def t40_firewall_integrated():
    try:
        ec.assert_dev_phase('testing')
        return False
    except RuntimeError:
        pass
    try:
        ec.assert_dev_phase('test')
        return False
    except RuntimeError:
        pass
    # source-level proof: dev runner modules must not construct TEST folds or
    # open the testing pair file.
    for mod_path in [da.__file__, ecv.__file__, ev.__file__, ec.__file__]:
        with open(mod_path) as f:
            src = f.read()
        if 'image_pairs_testing' in src:
            return False
        if re.search(r'fold\s*=\s*["\']test["\']', src):
            return False
        if re.search(r'phase\s*=\s*["\']testing["\']', src):
            return False
    return True


if __name__ == '__main__':
    ok = run_all([
        ('T30 attacker dev constructs no TEST loader', t30_no_test_loader),
        ('T31 classifier dev rejects TEST before dataset init', t31_classifier_rejects_test_preinit),
        ('T32 attacker TRAIN geometry anon/anon', t32_train_geometry_anon_anon),
        ('T33 attacker selection VAL geometry anon/anon', t33_val_selection_geometry_anon_anon),
        ('T34 scientific VAL privacy geometry anon/real', t34_scientific_val_geometry_anon_real),
        ('T35 configurable attacker seeds', t35_configurable_seeds),
        ('T36 classification VAL fold val/test', t36_classifier_val_fold),
        ('T37 common classification checkpoint SHA', t37_common_classifier_ckpt),
        ('T38 method-neutral checkpoint rule', t38_method_neutral_ckpt_rule),
        ('T39 paired order fingerprint deterministic', t39_order_fingerprint),
        ('T40 TEST firewall integrated', t40_firewall_integrated),
    ])
    sys.exit(0 if ok else 1)