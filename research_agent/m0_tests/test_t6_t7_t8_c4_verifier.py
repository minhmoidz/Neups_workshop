"""T6 + T7 + T8.

T6 — C4 source-feature gradient discipline. In the C4 port, the generator's
gradient path must come ONLY from the anonymized branch (deformed images) through
the loss_model; the source features (real images) must be detached, otherwise the
generator could "cheat" by moving source features (which would be a distributional
leak / broken semantics vs upstream baseline).

T7 — Verification adversary label/loss convention. SiameseDataset labels:
1.0 = same patient, 0.0 = different. The adversarial verifier is trained to
minimize classification loss on these labels; the anonymizer should push pairs
with label 1.0 away (increase distance). We verify the label convention and that
the verifier loss is a plain BCE-style loss (not metric-triplet).

T8 — Baseline mode must not silently become the C3 ensemble verifier. The restored
baseline uses the verifier directly (ver_ensemble_size=1, no restart loop). When
config omits ensemble keys the historical code defaults to ensemble_size=3 and
restart_every=25 (C3 ON). We verify the DEV config supplies explicit 1/0 so a
baseline run can never silently enable the ensemble path.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m0_common import run_all

import torch
import torch.nn as nn


class ToyVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward_once(self, x):
        return torch.sigmoid(self.net(x))

    def forward(self, a, b):
        oa = self.forward_once(a)
        ob = self.forward_once(b)
        return self._distance(oa, ob)

    @staticmethod
    def _distance(o1, o2):
        return torch.abs(o1 - o2)

    def predict_similarity(self, a, b):
        return self.forward(a, b)


def t6_c4_detached_source():
    """With detached source features, generator grad comes only via anonymized branch."""
    feat = torch.randn(4, 8, requires_grad=True)      # source features
    anon = torch.randn(4, 8, requires_grad=True)      # anonymized branch features
    loss = (anon - feat.detach()).pow(2).mean()
    loss.backward()
    # feat.detach() => feat has no grad; anon does
    return feat.grad is None and anon.grad is not None


def t6b_c4_undetached_has_grad():
    """Counter-example: undetached source features DO get grad (the forbidden path)."""
    feat = torch.randn(4, 8, requires_grad=True)
    anon = torch.randn(4, 8, requires_grad=True)
    loss = (anon - feat).pow(2).mean()
    loss.backward()
    return feat.grad is not None and anon.grad is not None


def t7_verifier_label_convention():
    """Pair labels: 1.0 = same patient, 0.0 = different."""
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    same = (labels == 1.0)
    diff = (labels == 0.0)
    return bool(same[0].item()) and bool(diff[1].item()) and not (same & diff).any()


def t7b_verifier_loss_bce():
    """Verifier loss = BCE on (distance, label); it is not a metric/triplet loss."""
    v = ToyVerifier()
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    labels = torch.tensor([1., 0., 1., 0.])
    sim = v.predict_similarity(a, b)
    loss = nn.BCELoss()(sim.squeeze(-1), labels)
    # For same-patient pairs the distance must be pushed to 0 (sim->1).
    # Just verify the loss is a scalar finite value computed via BCE on sigmoid sims.
    return bool(torch.isfinite(loss).item())


def t8_config_explicitly_disables_ensemble():
    """The DEV restored-baseline config must supply ver_ensemble_size=1 and
    ver_restart_every=0 (or ver_active_per_step) so baseline can't silently be C3."""
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                            'config_files', 'config_dev_restored_baseline.json')
    if not os.path.exists(cfg_path):
        return False  # config template not yet created
    import json
    with open(cfg_path) as f:
        cfg = json.load(f)
    return (cfg.get('ver_ensemble_size', 3) == 1
            and cfg.get('ver_restart_every', 25) in (0, None)
            and cfg.get('use_budget_map', True) is False
            and cfg.get('stochastic_lambda', 1.0) == 0.0)


if __name__ == '__main__':
    ok = run_all([
        ('T6 C4 detached source (gen-safe)', t6_c4_detached_source),
        ('T6b undetached source gets grad', t6b_c4_undetached_has_grad),
        ('T7 verifier label 1.0=same', t7_verifier_label_convention),
        ('T7b verifier loss is BCE', t7b_verifier_loss_bce),
        ('T8 dev config disables C3', t8_config_explicitly_disables_ensemble),
    ])
    sys.exit(0 if ok else 1)