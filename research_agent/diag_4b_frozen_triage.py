"""STEP 4B — FROZEN-ATTACKER DISTRIBUTION-SHIFT TRIAGE (VALIDATION only).

For ONE selected CAA mechanism diagnostic transform, evaluate an existing
corrected-baseline attacker checkpoint (seed 4) on VALIDATION pairs under:

    reference  : corrected-baseline deformed image (no extra transform)
    mechanism  : transform( corrected-baseline deformed image )

This is a FROZEN-ATTACKER DISTRIBUTION-SHIFT TRIAGE only — it measures how the
fixed seed-4 weights respond to the input distribution shift, NOT what an adapted
attacker recovers. H2/H3 are NEVER concluded from these triage numbers.

Recorded: AUC_reference, AUC_mechanism, ΔAUC.  No privacy number.

Usage:
    python research_agent/diag_4b_frozen_triage.py --transform border
    python research_agent/diag_4b_frozen_triage.py --transform intensity
"""

import argparse
import json
import os

import numpy as np
import torch
from torchvision import transforms

from networks.SiameseNetwork import SiameseNetwork
from utils import utils
from datasets.SiameseDataset import SiameseDataset
from utils.GaussianSmoothing import GaussianSmoothing
from research_agent import caa_transforms

PAIR_VAL = 'image_pairs/image_pairs_validation_2000.txt'
ATTACKER = 'archive/retrain_snn_seed4/retrain_snn_seed4_best_network.pth'


def build_deform(device):
    generator = utils.load_flow_generator('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth').to(device)
    generator.eval()
    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)

    def deform(vals):
        return utils.deform(vals, generator, grid_identity, gauss, 0.01, 0.0, 'corrected')
    return deform


def load_attacker(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state and hasattr(state['model'], 'eval'):
        return state['model'].to(device)
    if isinstance(state, dict) and 'net' in state:
        return state['net'].to(device)
    net = SiameseNetwork().to(device)
    net.load_state_dict(state)
    return net


@torch.no_grad()
def evaluate(net, loader, transform_fn, device):
    net.eval()
    y_true = []
    y_scores = []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for inputs1, inputs2, labels in loader:
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        inputs1 = transform_fn(inputs1)
        inputs2 = transform_fn(inputs2)
        inputs1, inputs2 = inputs1.expand(-1, 3, -1, -1), inputs2.expand(-1, 3, -1, -1)
        inputs1, inputs2 = normalize(inputs1), normalize(inputs2)
        outputs = net(inputs1, inputs2).squeeze()
        y_true.append(labels.cpu())
        y_scores.append(outputs.cpu())
    y_true = torch.cat(y_true).numpy()
    y_scores = torch.cat(y_scores).numpy()
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_scores)), int(len(y_true))


def main():
    parser = argparse.ArgumentParser('STEP 4B frozen-attacker triage')
    parser.add_argument('--transform', choices=['border', 'intensity'], required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deform = build_deform(device)

    if args.transform == 'border':
        mechanism_fn = lambda t: caa_transforms.border_normalize(deform(t))
        label = 'border_normalize(BW=4)'
        out_default = 'research_agent/05B_artifacts/frozen_triage_border.json'
    else:
        mechanism_fn = lambda t: caa_transforms.intensity_normalize(deform(t), 0.0, 1.0)
        label = 'intensity_normalize(p1/p99)'
        out_default = 'research_agent/05B_artifacts/frozen_triage_intensity.json'
    out_path = args.out or out_default

    ds = SiameseDataset(phase='validation', n_channels=1, image_size=256, image_path='/home/minhtt/datasets/nih/images/')
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)

    net = load_attacker(ATTACKER, device)
    net.eval()

    auc_ref, n = evaluate(net, loader, deform, device)
    auc_mech, n2 = evaluate(net, loader, mechanism_fn, device)

    result = {
        '_label': 'FROZEN-ATTACKER DISTRIBUTION-SHIFT TRIAGE (not a privacy estimate)',
        'mechanism': args.transform,
        'mechanism_label': label,
        'attacker_checkpoint': ATTACKER,
        'attacker_seed': 4,
        'AUC_reference': auc_ref,
        'AUC_mechanism': auc_mech,
        'delta_auc': auc_mech - auc_ref,
        'n_pairs': n,
        'generator': 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth',
        'transform_mode': 'corrected',
        'mu': 0.01,
        'stochastic_lambda': 0.0,
        'validation_pairs': PAIR_VAL,
        'test_touched': False,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print('=== FROZEN TRIAGE (mechanism=%s) ===' % args.transform)
    print('AUC_reference = %.5f   AUC_mechanism = %.5f   delta = %+.5f  (n=%d)' % (auc_ref, auc_mech, auc_mech - auc_ref, n))
    print('wrote', out_path)


if __name__ == '__main__':
    main()