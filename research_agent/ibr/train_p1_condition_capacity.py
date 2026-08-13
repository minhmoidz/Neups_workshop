"""STEP 7B — P1 capacity-limited anatomy condition experiment driver.

Reuses the STEP 7A pairwise condition-attacker design (same architecture class,
order-invariant |e1-e2|, BCE, Adam lr=1e-4, patience 5, min-val-BCE ckpt).
Anatomy branch reads the frozen P1 coarse cache (3x64x64 binary).

Arms:
    ARM A — P1 coarse anatomy only
    ARM B — pathology only (REUSES STEP 7A records; not retrained here)
    ARM C — P1 coarse anatomy + pathology (PRIMARY)

TRAIN + VALIDATION only. No TEST. No resolution/threshold sweep.
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from research_agent.ibr.condition_attacker import build_model

CACHE_DIR = 'research_agent/ibr_s1_condition_cache/'
P1_CACHE_DIR = 'research_agent/ibr_s1_condition_p1_cache/'
PAIR_FILES = {
    'train': 'image_pairs/image_pairs_training_10000.txt',
    'val': 'image_pairs/image_pairs_validation_2000.txt',
}
LABELS_CSV = 'chexnet/nih_labels.csv'
OUT_DIR = 'research_agent/ibr_s1_condition_p1_capacity/'
PRED_LABEL_ORDER = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
                    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
MAX_EPOCHS = 60
PATIENCE = 5
LR = 1e-4
BS = 64


def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()


def load_p1_cache():
    import pandas as pd
    df = pd.read_csv(LABELS_CSV).set_index('Image Index')
    labels_by_image = {}
    for _, row in df.iterrows():
        labels_by_image[str(row.name)] = row[PRED_LABEL_ORDER].values.astype(np.float32)
    cache = {}
    for fold in ('train', 'val'):
        p1 = np.load(os.path.join(P1_CACHE_DIR, '%s_p1.npy' % fold))
        with open(os.path.join(CACHE_DIR, '%s_images.json' % fold)) as f:
            images = json.load(f)
        assert p1.shape[0] == len(images)
        cache[fold] = {'p1': p1, 'images': images}
    pairs = {}
    for fold in ('train', 'val'):
        pairs[fold] = np.loadtxt(PAIR_FILES[fold], dtype=str, delimiter='\t')
    return cache, labels_by_image, pairs


class P1PairDataset(Dataset):
    """Pair row -> (m1, m2, y1, y2, label) with m = P1 coarse 3x64x64 uint8."""

    def __init__(self, fold, cache, labels_by_image, pairs):
        self.fold = fold
        self.cache = cache[fold]
        self.labels = labels_by_image
        self.rows = pairs[fold]
        self.idx = {img: i for i, img in enumerate(self.cache['images'])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a, b, lab = self.rows[i]
        m1 = self.cache['p1'][self.idx[a]].astype(np.float32)   # (3,64,64) binary
        m2 = self.cache['p1'][self.idx[b]].astype(np.float32)
        y1 = self.labels[a]
        y2 = self.labels[b]
        return m1, m2, y1, y2, float(lab)


def eval_auc(model, loader, device):
    model.eval()
    logits = []
    labs = []
    with torch.no_grad():
        for m1, m2, y1, y2, lab in loader:
            m1 = m1.to(device); m2 = m2.to(device)
            y1 = y1.to(device); y2 = y2.to(device)
            logits.append(model(m1, m2, y1, y2).cpu())
            labs.append(lab)
    l = torch.cat(logits).flatten().numpy()
    y = torch.cat(labs).numpy()
    return float(roc_auc_score(y, l))


def run_arm(arm, seed, device, cache, labels_by_image, pairs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(arm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tr = P1PairDataset('train', cache, labels_by_image, pairs)
    va = P1PairDataset('val', cache, labels_by_image, pairs)
    tr_dl = DataLoader(tr, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True)
    va_dl = DataLoader(va, batch_size=BS, shuffle=False, num_workers=8, pin_memory=True)

    best_loss = float('inf')
    best_auc = 0.0
    best_epoch = 0
    patience = 0
    term = 'max_epochs'
    t0 = time.time()

    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        run_loss = 0.0
        nb = 0
        for m1, m2, y1, y2, lab in tr_dl:
            m1 = m1.to(device); m2 = m2.to(device)
            y1 = y1.to(device); y2 = y2.to(device)
            lab = lab.to(device).unsqueeze(1)
            opt.zero_grad()
            logit = model(m1, m2, y1, y2)
            loss = F.binary_cross_entropy_with_logits(logit, lab)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            nb += 1
        tr_loss = run_loss / max(nb, 1)
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vn = 0
            for m1, m2, y1, y2, lab in va_dl:
                m1 = m1.to(device); m2 = m2.to(device)
                y1 = y1.to(device); y2 = y2.to(device)
                lab = lab.to(device).unsqueeze(1)
                vloss += F.binary_cross_entropy_with_logits(model(m1, m2, y1, y2), lab).item()
                vn += 1
            va_loss = vloss / max(vn, 1)
        va_auc = eval_auc(model, va_dl, device)

        improved = va_loss < best_loss - 1e-6
        if improved:
            best_loss = va_loss
            best_auc = va_auc
            best_epoch = ep
            patience = 0
            torch.save({'arm': arm, 'seed': seed, 'epoch': ep, 'model': model.state_dict(),
                        'best_val_loss': best_loss, 'best_val_auc': best_auc},
                       os.path.join(OUT_DIR, 'p1_arm%s_seed%d_best.pth' % (arm, seed)))
        else:
            patience += 1
        print('  p1 arm %s seed %d ep %02d tr %.4f va %.4f va_auc %.4f best %.4f@%d pat %d (%.0fs)'
              % (arm, seed, ep, tr_loss, va_loss, va_auc, best_loss, best_epoch, patience,
                 time.time() - t0))
        if patience >= PATIENCE:
            term = 'early_stopping_patience_%d' % PATIENCE
            break

    return {'arm': arm, 'seed': seed, 'best_val_loss': best_loss, 'best_val_auc': best_auc,
            'best_epoch': best_epoch, 'termination': term, 'num_epochs_run': ep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='A,C')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    arms = args.arms.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    assert 'C' in arms, 'JOINT (ARM C) mandatory'

    os.makedirs(OUT_DIR, exist_ok=True)
    cache, labels_by_image, pairs = load_p1_cache()
    device = args.device

    results = []
    for arm in arms:
        print('=== P1 ARM %s ===' % arm)
        for seed in seeds:
            r = run_arm(arm, seed, device, cache, labels_by_image, pairs)
            results.append(r)

    # ARM B (pathology only): reuse exact STEP 7A records (identical inputs/code path)
    with open('research_agent/ibr_s1_condition_capacity/results.json') as f:
        step7a = json.load(f)
    b_records = [r for r in step7a if r['arm'] == 'B']
    for r in b_records:
        r = dict(r)
        r['reused_from_7A'] = True
        results.append(r)

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('RESULTS SAVED to %s' % OUT_DIR)
    for r in results:
        print(r)


if __name__ == '__main__':
    main()