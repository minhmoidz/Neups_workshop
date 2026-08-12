"""STEP 4B — adaptive CAA mechanism diagnostic (one fresh attacker per mechanism, seed 42).

For ONE selected CAA mechanism diagnostic transform, train exactly ONE fresh
attacker A (canonical recipe, seed 42) with the SAME transform applied consistently
on TRAIN and VALIDATION:

    mechanism arm  : transform( corrected-baseline deformed image )

seed = 42 (fixed). NO TEST split is ever opened (evaluate_test_after_training=False).

Recipe fidelity (identical to the frozen corrected-baseline attacker protocol):
    - architecture : networks.SiameseNetwork (ResNet-50 backbone, 128-d head)
    - optimizer    : Adam, lr = 1e-4
    - batch size   : 16
    - early stopping : patience = 5 (min-delta 0) on validation loss
    - checkpoint rule : lowest validation loss (deepcopy of best net)
    - max epochs   : 100 (same cap)
    - pair files   : image_pairs_training_10000.txt / image_pairs_validation_2000.txt
    - deformation  : corrected generator, transform_mode=corrected, mu=0.01, lambda=0
    - determinism  : utils.seed_all(seed) at construction (same as AgentSiameseNetwork)

Persists band-style diagnostics JSON per arm. One restart per mechanism.

Usage:
    python research_agent/diag_4b_adaptive.py --transform border [--seed 42] [--max_epochs 100]
    python research_agent/diag_4b_adaptive.py --transform intensity
"""

import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from networks.SiameseNetwork import SiameseNetwork
from utils import utils
from datasets.SiameseDataset import SiameseDataset
from utils.EarlyStopping import EarlyStopping
from utils.GaussianSmoothing import GaussianSmoothing
from research_agent import caa_transforms

IMAGE_PATH = '/home/minhtt/datasets/nih/images/'
GENERATOR = 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth'
PAIR_TRAIN = 'image_pairs/image_pairs_training_10000.txt'
PAIR_VAL = 'image_pairs/image_pairs_validation_2000.txt'


def build_deform(device, mu):
    generator = utils.load_flow_generator(GENERATOR).to(device)
    generator.eval()
    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)

    def deform(vals):
        return utils.deform(vals, generator, grid_identity, gauss, mu, 0.0, 'corrected')
    return deform


def get_mechanism(name):
    """Return (key, human label, transform composition over deformed images)."""
    if name == 'border':
        return 'H3_border', 'border_normalize(BW=4)', lambda x: caa_transforms.border_normalize(x)
    if name == 'intensity':
        return 'H2_intensity', 'intensity_normalize(p1/p99)', lambda x: caa_transforms.intensity_normalize(x, 0.0, 1.0)
    raise ValueError(name)


def run_mechanism(name, seed, device, out_dir, max_epochs=100):
    utils.seed_all(seed)
    key, label, transform_fn = get_mechanism(name)

    exp = 'diag_4b_arm_%s_seed%d' % (name, seed)
    save_path = os.path.join(out_dir, exp)
    os.makedirs(save_path, exist_ok=True)

    net = SiameseNetwork().cuda()
    best_net = copy.deepcopy(net)
    loss_fn = nn.BCEWithLogitsLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    es = EarlyStopping(mode='min', min_delta=0, patience=5)
    best_loss = 1e6

    train_ds = SiameseDataset(phase='training', n_channels=1, image_size=256, image_path=IMAGE_PATH)
    val_ds = SiameseDataset(phase='validation', n_channels=1, image_size=256, image_path=IMAGE_PATH)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    deform = build_deform(device, 0.01)

    def apply(t):
        return transform_fn(deform(t))

    val_auc_history = []
    val_loss_history = []
    best_val_auc = -1.0
    best_val_auc_epoch = -1
    best_val_loss_epoch = -1
    termination = 'epoch_cap'
    completed = 0

    for epoch in range(max_epochs):
        net.train()
        run_loss = 0.0
        for i, (inputs1, inputs2, labels) in enumerate(train_loader):
            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()
            inputs1 = apply(inputs1)
            inputs2 = apply(inputs2)
            inputs1, inputs2 = inputs1.expand(-1, 3, -1, -1), inputs2.expand(-1, 3, -1, -1)
            inputs1, inputs2 = normalize(inputs1), normalize(inputs2)
            optimizer.zero_grad()
            outputs = net(inputs1, inputs2).squeeze()
            labels = labels.type_as(outputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()

        train_loss = run_loss / len(train_loader)

        net.eval()
        v_loss = 0.0
        y_true = []
        y_scores = []
        with torch.no_grad():
            for inputs1, inputs2, labels in val_loader:
                inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()
                inputs1 = apply(inputs1)
                inputs2 = apply(inputs2)
                inputs1, inputs2 = inputs1.expand(-1, 3, -1, -1), inputs2.expand(-1, 3, -1, -1)
                inputs1, inputs2 = normalize(inputs1), normalize(inputs2)
                outputs = net(inputs1, inputs2).squeeze()
                labels = labels.type_as(outputs)
                v_loss += loss_fn(outputs, labels).item()
                y_true.append(labels.cpu())
                y_scores.append(outputs.cpu())
        v_loss = v_loss / len(val_loader)
        y_true = torch.cat(y_true).numpy()
        y_scores = torch.cat(y_scores).numpy()
        from sklearn.metrics import roc_auc_score
        val_auc = float(roc_auc_score(y_true, y_scores))
        val_auc_history.append(val_auc)
        val_loss_history.append(v_loss)
        completed += 1

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_auc_epoch = epoch

        if v_loss < best_loss:
            best_loss = v_loss
            best_net = copy.deepcopy(net)
            best_val_loss_epoch = epoch

        print('[%s] epoch %d  train_loss %.4f  val_loss %.4f  val_auc %.5f' % (
            key, epoch, train_loss, v_loss, val_auc), flush=True)

        if es.step(v_loss):
            termination = 'early_stopping'
            break

    torch.save(best_net.state_dict(), os.path.join(save_path, exp + '_best_network.pth'))
    record = {
        'mechanism': name,
        'mechanism_label': label,
        'seed': seed,
        'epochs_completed': completed,
        'termination_reason': termination,
        'best_validation_loss': float(best_loss),
        'best_validation_loss_epoch': best_val_loss_epoch,
        'best_validation_auc': best_val_auc,
        'best_validation_auc_epoch': best_val_auc_epoch,
        'validation_auc_per_epoch': val_auc_history,
        'validation_loss_per_epoch': val_loss_history,
        'mu': 0.01,
        'transform_mode': 'corrected',
        'stochastic_lambda': 0.0,
        'generator_checkpoint': GENERATOR,
        'pair_train_path': PAIR_TRAIN,
        'pair_validation_path': PAIR_VAL,
        'architecture': 'SiameseNetwork(ResNet-50, 128-d)',
        'optimizer': 'Adam lr=1e-4',
        'batch_size': 16,
        'early_stopping': 'patience=5 min-delta=0',
        'checkpoint_rule': 'lowest validation loss',
        'test_touched': False,
    }
    with open(os.path.join(save_path, 'mechanism_diagnostics.json'), 'w') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


def main():
    parser = argparse.ArgumentParser('STEP 4B adaptive CAA mechanism diagnostic')
    parser.add_argument('--transform', choices=['border', 'intensity'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--out', default='research_agent/05B_artifacts/adaptive')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rec = run_mechanism(args.transform, args.seed, device, args.out, max_epochs=args.max_epochs)
    print('=== %s ARM DONE: best_val_auc=%.5f (epoch %d) ===' % (rec['mechanism'], rec['best_validation_auc'], rec['best_validation_auc_epoch']))


if __name__ == '__main__':
    main()