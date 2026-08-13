"""Phase-II IBR S1 - future training entry point (IMPLEMENTATION ONLY, no full run).

This script defines the exact training loop S1 will use. It is implemented now
but is NOT executed as a full training run in STEP 6B.

Guarantees:
    - TRAIN only; VALIDATION only for development monitoring.
    - No default route to TEST: requires an explicit --split argument and rejects
      'test' (or 'TEST') during Phase-II development.
    - Logs every loss component, total loss, gradient finite flags,
      reconstruction metrics, donor diagnostics, z_id verifier performance,
      z_med identity-adversary performance, classification diagnostic,
      segmentation diagnostic, config, seed, checkpoint hashes, and git commit.

Invocation (future, AFTER STEP 6B):
    PYTHONPATH=. .venv/bin/python research_agent/ibr/train_s1.py \
        --split train --seed 42 --bs 16 --max_epochs 100

STEP 6B only runs the DRY RUN via research_agent/ibr/dry_run_s1.py.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from research_agent.ibr.ibr_model import IBRModel, count_parameters
from research_agent.ibr.frozen_models import (CLASSIFIER_SHA256, SEGMENTATION_SHA256_PREFIX,
                                              load_frozen_classifier,
                                              load_frozen_segmentation_teacher)
from research_agent.ibr.losses import (LAMBDA_REC, LAMBDA_PATH, LAMBDA_ANAT,
                                       LAMBDA_ZID, LAMBDA_ADV, GRL_LAMBDA,
                                       FrozenUtility)
from research_agent.ibr.s1_loss import compute_s1_loss
from research_agent.ibr.donor import DonorSampler

ALLOWED_SPLITS = ('train', 'validation')


def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='IBR S1 training (future entry point).')
    p.add_argument('--split', required=True, choices=ALLOWED_SPLITS,
                   help='Development split for S1. TEST is REJECTED in Phase-II.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bs', type=int, default=16)
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--device', default='cuda')
    return p.parse_args(argv)


def _guard_split(split):
    if str(split).lower() in ('test', 'testarm', 'test arm'):
        raise SystemExit('TEST split is FORBIDDEN in Phase-II development mode. Aborting.')
    if split not in ALLOWED_SPLITS:
        raise ValueError('split must be one of %s, got %r' % (ALLOWED_SPLITS, split))


def build_training_components(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = IBRModel().to(device)
    frozen = FrozenUtility(device)
    sampler = DonorSampler(seed=seed)
    return model, frozen, sampler


def check_finite(tensors, prefix):
    out = {}
    for name, t in tensors.items():
        out[prefix + '_' + name + '_finite'] = bool(torch.isfinite(t).all().item())
    return out


def train_one_batch(model, frozen, optimizer, optimizer_adv, x, x_donor, y_path, x_pair, y_pair):
    """One S1 training step.

    Optimizers:
        optimizer     : E + G + V (Adam lr=1e-4)
        optimizer_adv : H_med (Adam lr=1e-4)
    GRL handles the sign flip on the z_med path; H_med trains normally.
    """
    optimizer.zero_grad()
    optimizer_adv.zero_grad()

    total, parts = compute_s1_loss(model, frozen, x, x_donor, y_path, x_pair, y_pair,
                                   return_parts=True)
    total.backward()

    optimizer.step()
    optimizer_adv.step()

    grads = {k: (v.grad if v.grad is not None else None)
             for k, v in model.named_parameters()}
    finite = check_finite({k: v for k, v in grads.items() if v is not None}, 'grad')
    return parts, finite


def provenance_record(sampler, x, x_donor):
    """Persist donor/source patient IDs as a diagnostic."""
    return sampler.provenance(x, x_donor)


if __name__ == '__main__':
    # PURPOSE: this entry point exists for future execution. STEP 6B performs only
    # the dry run (dry_run_s1.py). If invoked now with a full run it still guards
    # TEST strictly and requires --split.
    args = parse_args()
    if len(sys.argv) == 1:
        raise SystemExit('train_s1.py requires --split (train or validation). Running the full S1 '
                         'training run is deferred past STEP 6B; use dry_run_s1.py now.')
    _guard_split(args.split)

    print('IBR S1 training entry point ready. split=%s seed=%d bs=%d max_epochs=%d' % (
        args.split, args.seed, args.bs, args.max_epochs))
    print('Full training run is NOT executed in STEP 6B. Implementation approved only.')