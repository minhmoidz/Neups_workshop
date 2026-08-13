"""Phase-II IBR S1 - dry run (STEP 6B).

Allowed:
    - model construction
    - dataset loading sanity
    - 1 small batch forward
    - 1 backward pass

Not allowed:
    - epoch training
    - attacker training
    - generator/model checkpoint selection

Measures actual peak VRAM during the dry run and compares to the STEP 6A
estimate (bs=16 -> ~6.78 GB peak). If bs=16 OOMs or leaves unsafe headroom,
STOP and report before changing batch size. Gradient accumulation is never
introduced automatically.
"""

import argparse
import os
import time

import numpy as np
import torch

from research_agent.ibr.ibr_model import IBRModel, count_parameters
from research_agent.ibr.losses import FrozenUtility
from research_agent.ibr.s1_loss import compute_s1_loss
from research_agent.ibr.donor import DonorSampler

BS = 16


def build_synthetic_batch(bs, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(bs, 1, 256, 256, generator=g).clamp(-1, 1)
    x_donor = torch.randn(bs, 1, 256, 256, generator=g).clamp(-1, 1)
    y_path = (torch.rand(bs, 14, generator=g) > 0.5).float()
    y_pair = torch.zeros(bs, 1)  # different-patient pair by donor protocol
    return x, x_donor, y_path, y_pair


def dataset_sanity():
    """One-row dataset loading sanity (no training)."""
    try:
        from chexnet.cxr_dataset import CXRDataset
        ds = CXRDataset(path_to_images='/home/minhtt/datasets/nih/images/', fold='val',
                        perturbation_type='flow_field', sample=2)
        img, label, name = ds[0]
        ok = (tuple(img.shape) == (1, 256, 256)) and (label.shape == (14,)) and isinstance(name, str)
        return {'dataset': 'CXRDataset(val)', 'sample_ok': ok, 'img_shape': tuple(img.shape),
                'label_shape': tuple(label.shape), 'img_range': (float(img.min()), float(img.max()))}
    except Exception as e:  # pragmatic: report, do not abort dry run
        return {'dataset': 'CXRDataset(val)', 'sample_ok': False, 'error': str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bs', type=int, default=BS)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    bs = args.bs
    device = args.device
    if not torch.cuda.is_available():
        print('SKIP: CUDA unavailable; VRAM measurement requires GPU')
        return 0

    torch.cuda.reset_peak_memory_stats()

    model = IBRModel().to(device)
    frozen = FrozenUtility(device)
    sampler = DonorSampler(seed=0)

    n_train = count_parameters(model)
    print('S1 trainable params: %.2fM (E+G+V+H_med)' % (n_train / 1e6))
    print('frozen classifier params: %.2fM' % (frozen.classifier_meta['params'] / 1e6))
    print('frozen seg teacher params: %.2fM' % (frozen.seg_meta['params'] / 1e6))

    dset = dataset_sanity()
    print('dataset sanity:', dset)

    x, x_donor, y_path, y_pair = build_synthetic_batch(bs, seed=0)
    x, x_donor, y_path, y_pair = (x.to(device), x_donor.to(device),
                                  y_path.to(device), y_pair.to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_adv = torch.optim.Adam(model.adv.parameters(), lr=1e-4)

    total, parts = compute_s1_loss(model, frozen, x, x_donor, y_path, y_pair, return_parts=True)
    total.backward()
    optimizer.step()
    optimizer_adv.step()

    peak = torch.cuda.max_memory_allocated() / 1e9
    free, _ = torch.cuda.mem_get_info()
    print('parts: %s' % json_dumps(parts))
    print('PEAK VRAM: %.2f GB  (STEP 6A estimate 6.78 GB @ bs=16)  free=%.2f GB' % (peak, free / 1e9))

    finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    print('post-step params finite:', finite)

    # compare to STEP 6A estimate
    ok = peak < 12.0 and free > 2.0
    print('DRY RUN: %s (bs=%d)' % ('OK' if ok else 'BLOCKED', bs))
    return 0 if ok else 1


def json_dumps(d):
    import json
    return json.dumps(d)


if __name__ == '__main__':
    import sys
    sys.exit(main())