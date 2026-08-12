"""STEP 4A PART B — S0a frozen-attacker triage on VALIDATION.

Given an existing corrected-baseline attacker checkpoint, evaluate it on VALIDATION
pairs under three conditions of the anonymized (corrected-baseline deformed) input:

    original  : corrected-baseline deformed validation images (no band filter)
    low_pass  : LP( deformed validation image )
    high_pass : HP( deformed validation image )

This is a FROZEN-ATTACKER DISTRIBUTION-SHIFT DIAGNOSTIC only — it is NOT an adaptive
privacy estimate and produces no privacy number. H1 is never concluded from this stage.

Protocol fidelity:
    - Same SiameseDataset VALIDATION pairs (image_pairs_validation_2000.txt).
    - Same attacker A architecture: networks.SiameseNetwork (ResNet-50, 128-d head).
    - Same deformation: utils.deform, corrected generator, transform_mode=corrected,
      mu=0.01, stochastic_lambda=0.0 (bit-for-bit frozen operator).
    - Same 3-channel expansion + ImageNet normalization.
    - Band transform inserted ONLY at the deformed-image stage (the mechanism under test).
    - Scores are the continuous Siamese logits; ROC-AUC identical to the protocol metric.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from networks.SiameseNetwork import SiameseNetwork
from utils import utils
from datasets.SiameseDataset import SiameseDataset
from research_agent import band

PAIR_VAL = 'image_pairs/image_pairs_validation_2000.txt'


@torch.no_grad()
def evaluate(net, loader, band_fn, device):
    net.eval()
    y_true = []
    y_scores = []
    with torch.no_grad():
        for inputs1, inputs2, labels in loader:
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            if band_fn is not None:
                inputs1 = band_fn(inputs1)
                inputs2 = band_fn(inputs2)
            inputs1 = inputs1.expand(-1, 3, -1, -1)
            inputs2 = inputs2.expand(-1, 3, -1, -1)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            inputs1, inputs2 = normalize(inputs1), normalize(inputs2)
            outputs = net(inputs1, inputs2).squeeze()
            y_true.append(labels.cpu())
            y_scores.append(outputs.cpu())
    y_true = torch.cat(y_true).numpy()
    y_scores = torch.cat(y_scores).numpy()
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_scores)), int(len(y_true))


def load_attacker(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state and hasattr(state['model'], 'eval'):
        net = state['model'].to(device)
    elif isinstance(state, dict) and 'net' in state:
        net = state['net'].to(device)
    else:
        net = SiameseNetwork().to(device)
        net.load_state_dict(state)
    return net


def main():
    parser = argparse.ArgumentParser('STEP 4A PART B frozen-attacker triage')
    parser.add_argument('--attacker', default='archive/retrain_snn_seed4/retrain_snn_seed4_best_network.pth')
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--out', default='research_agent/05A_artifacts/frozen_attacker_triage.json')
    parser.add_argument('--no_band', action='store_true', help='run only the original condition')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generator = utils.load_flow_generator('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth').to(device)
    generator.eval()
    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    from utils.GaussianSmoothing import GaussianSmoothing
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)

    def deform(vals):
        return utils.deform(vals, generator, grid_identity, gauss, args.mu, 0.0, 'corrected')

    ds = SiameseDataset(phase='validation', n_channels=1, image_size=256, image_path='/home/minhtt/datasets/nih/images/')
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)

    net = load_attacker(args.attacker, device)
    net.eval()

    band_fns = {
        'original': None,
        'low_pass': lambda t: band.low_pass(deform(t)),
        'high_pass': lambda t: band.high_pass(deform(t)),
    }
    if args.no_band:
        band_fns = {'original': None}

    results = {}
    for name, fn in band_fns.items():
        auc, n = evaluate(net, loader, fn, device)
        results[name] = {'validation_auc': auc, 'n_pairs': n}
        label = 'FROZEN-ATTACKER DISTRIBUTION-SHIFT DIAGNOSTIC  %-10s  validation_auc = %.5f (n=%d)' % (name, auc, n)
        print(label)

    results['_label'] = 'FROZEN-ATTACKER DISTRIBUTION-SHIFT DIAGNOSTIC (not an adaptive privacy estimate)'
    results['attacker_checkpoint'] = args.attacker
    results['generator'] = 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth'
    results['transform_mode'] = 'corrected'
    results['mu'] = args.mu
    results['stochastic_lambda'] = 0.0
    results['validation_pairs'] = PAIR_VAL
    results['band_lowpass_sigma'] = band.SIGMA_LP

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()