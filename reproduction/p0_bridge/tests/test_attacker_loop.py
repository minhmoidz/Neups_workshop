"""Standalone CPU-only attacker-loop tests (P0_2_3).

Synthetic tiny network + synthetic dataset; NO real images, pair files,
checkpoints, or CUDA. Run: CUDA_VISIBLE_DEVICES="" python test_attacker_loop.py
"""
import hashlib
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn

assert not torch.cuda.is_initialized()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from manifest_io import (ManifestError, aggregate_manifests,
                         load_run_manifest)
PROTOCOL_SHA = "a" * 64
RUNNER_COMMIT = "c" * 40

PROTOCOL = {
    "schema_version": "P0_PROTOCOL_V1_1",
    "seeds": {"screen": [42, 43], "full": [42, 43]},
    "arms": {
        "U_PUBLISHED": {
            "generator_sha256": "4d" * 32,
            "execution_path": "fake_gen_U.pth"},
        "D_BDEV": {
            "generator_sha256": "18" * 32,
            "execution_path": "fake_gen_D.pth"}},
    "pair_files": {"train": {"sha256": "t" * 64, "pair_count": 64},
                   "val": {"sha256": "v" * 64, "pair_count": 16}},
    "attacker_protocol": {
        "domains": ["attacker_weight_init", "train_order",
                    "dataloader_worker_base", "statistical_sensitivity"],
        "score_direction": "sigmoid_logit_higher_means_same_patient_fixed_apriori",
        "max_epochs": 6,
        "early_stopping_patience": 3,
        "learning_rate": 0.001,
        "batch_size": 8,
        "num_workers": 0,
        "flow_operator": {"mu": 0.01, "gaussian_kernel": 9,
                          "gaussian_sigma": 2}},
}


class TinySiamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 2, 3, padding=1), nn.ReLU(),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(2, 1)

    def forward(self, a, b):
        return self.head(torch.abs(self.f(a) - self.f(b)))


class TinyGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x):
        return torch.tanh(self.cv(x))


class FakePairs(torch.utils.data.Dataset):
    """Deterministic tiny pair dataset; label = same-patient flag."""

    def __init__(self, n=64):
        g = torch.Generator().manual_seed(n)
        self.n = n
        self.imgs = torch.rand(n, 1, 8, 8, generator=g)
        self.labels = (torch.rand(n, generator=g) > 0.5).float()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        j = i if self.labels[i] == 1 else int((i + 7) % self.n)
        ac = torch.zeros(14)                      # dummy pathology vector
        return self.imgs[i], self.imgs[j], ac, self.labels[i]


def make_runner(root, arm="U_PUBLISHED", seed=42, anonymize=None,
                net_factory=TinySiamese):
    from attacker_loop import P0AttackerRunner
    # write fake frozen generators so execution_path loads succeed
    gp = os.path.join(root, "fake_gen_%s.pth" % arm[0])
    if not os.path.exists(gp):
        gstate = torch.get_rng_state()
        torch.manual_seed(20260823)           # deterministic fake generator
        torch.save(TinyGen().state_dict(), gp)
        torch.set_rng_state(gstate)

    def tiny_gen():
        return TinyGen()

    calls = {"anon": 0}

    def dataset_factory(phase):
        return FakePairs(64 if phase == "train" else 16)

    r = P0AttackerRunner(
        arm=arm, master_seed=seed, protocol=PROTOCOL, repo_root=root,
        protocol_sha256=PROTOCOL_SHA, runner_commit=RUNNER_COMMIT,
        device=torch.device("cpu"), dataset_factory=dataset_factory,
        net_factory=net_factory, generator_factory=tiny_gen)
    if anonymize is not None:
        r.anonymize = anonymize
    return r


def test_full_run_writes_bound_manifest():
    with tempfile.TemporaryDirectory() as root:
        r = make_runner(root, seed=42)
        r.run_training()
        rd, man = r.finalize(runs_root=root)
        assert man["status"] == "COMPLETE"
        assert set(man["epoch_order_hashes"].keys()) == \
            {str(e) for e in range(0, r.stop_epoch + 1)}
        loaded = load_run_manifest(rd)
        assert loaded["predictions_sha256"] == man["predictions_sha256"]
        # byte binding: recompute predictions hash from disk
        h = hashlib.sha256(open(os.path.join(rd, "predictions.csv"),
                                "rb").read()).hexdigest()
        assert h == loaded["predictions_sha256"]
        h2 = hashlib.sha256(open(os.path.join(rd, "attacker_best.pth"),
                                 "rb").read()).hexdigest()
        assert h2 == loaded["attacker_best_sha256"]
        auc = loaded["raw_roc_auc"]
        assert 0.0 <= auc <= 1.0


def test_same_seed_identical_trajectory_and_outputs():
    outs = []
    for _ in range(2):
        root = tempfile.mkdtemp()
        r = make_runner(root, seed=42)
        r.run_training()
        score = r.score_anon_real()
        outs.append((r.loss_history["training"],
                     r.epoch_order_hashes,
                     round(score["raw_roc_auc"], 10)))
    assert outs[0] == outs[1]


def test_different_seed_differs():
    runs = []
    for s in (42, 43):
        root = tempfile.mkdtemp()
        r = make_runner(root, seed=s)
        r.run_training()
        runs.append((r.loss_history["training"], r.epoch_order_hashes))
    assert runs[0] != runs[1]


def test_nan_guard_raises():
    with tempfile.TemporaryDirectory() as root:
        def bad_anon(gen, t):
            t = t.clone()
            t[:, :, 0, 0] = float("nan")
            return t
        r = make_runner(root, anonymize=None)
        r.anonymize = lambda x: bad_anon(None, x)
        try:
            r.train_epoch(r.train_loader_builder(0))
            raise AssertionError("NaN loss accepted")
        except FloatingPointError:
            pass


def test_aggregate_accepts_two_valid_runs(tmp_root=None):
    with tempfile.TemporaryDirectory() as root:
        runs_root = os.path.join(root, "runs")
        for arm in ("U_PUBLISHED", "D_BDEV"):
            r = make_runner(root, arm=arm, seed=42)
            r.run_training()
            _, man = r.finalize(runs_root=runs_root)
        proto = dict(PROTOCOL)
        proto["seeds"] = {"screen": [42], "full": [42]}   # unit-scale stage
        agg = aggregate_manifests(runs_root, proto, PROTOCOL_SHA,
                                  RUNNER_COMMIT, "screen")
        assert agg["count"] == 2


if __name__ == "__main__":
    names = sorted(k for k in globals() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("PASS", name)
    print("ALL PASS (%d)" % len(names))
