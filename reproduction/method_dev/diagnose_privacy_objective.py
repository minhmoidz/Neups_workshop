"""Diagnosis: the PriCheXy-Net privacy objective is invariant to the ROC AUC
it is supposed to improve.

WHY THIS EXISTS
---------------
The project spent substantial GPU budget on Direction A (periodic best-response
critic refresh) and Direction B (k:1 critic/generator update ratio), both of
which assume the SAME premise: that the train-time verifier is too WEAK, so the
generator only learns to fool a weak adversary. That premise was recorded as
`WEAK_CRITIC_HYPOTHESIS_STATUS: UNTESTED` in
`reproduction/reports/P0_P1_PREEXPERIMENT_PROTOCOL_REVIEW_2026-08-21.md` §14.

This script tests it, and finds the opposite. The co-adapted training verifier
is the STRONGEST re-identifier measured on its own generator. The critic is not
weak; the OBJECTIVE is misspecified.

THE ARGUMENT
------------
Upstream's privacy term (utils/VerificationLoss.py + utils/utils.py::train,
both byte-identical to upstream commit 29245d1) is

    L_priv = -log(1 - sigmoid(z)) = softplus(z),   z = verifier logit

Minimizing it drives EVERY logit toward -inf, i.e. it asks the verifier to say
"different patient" for every pair. ROC AUC depends only on the ORDER of z
between positive and negative pairs, and a uniform shift of z preserves order
exactly. So L_priv can be driven to 0 while AUC is untouched. Section D1 shows
this numerically; D4 shows it happening in real trained checkpoints.

STRICTLY VALIDATION-ONLY. This script never constructs a TEST loader and never
trains anything: it is pure measurement over existing artifacts.

Output: <out_dir>/privacy_objective_diagnosis.json plus a printed summary.
Every input checkpoint and pair file is SHA-256'd into the JSON so the result
is bound to the exact bytes it was computed from.
"""
import hashlib
import json
import os
import statistics as st
import subprocess
import sys
from collections import Counter

import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from networks.SiameseNetwork import SiameseNetwork  # noqa: E402
from networks.UNet_PriCheXyNet import UNet  # noqa: E402
from networks.UNet_AttentionPriCheXyNet import UNetAtt  # noqa: E402
from utils.GaussianSmoothing import GaussianSmoothing  # noqa: E402

MU = 0.01
IMAGE_SIZE = 256
BATCH = 16
IMAGE_ROOT = '/home/minhtt/datasets/nih/images/'
VAL_PAIRS = os.path.join(ROOT, 'image_pairs', 'image_pairs_validation_2000.txt')
TRAIN_PAIRS = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
OUT_DIR = os.path.join(ROOT, 'reproduction', 'method_dev', 'privacy_objective_diagnosis')

IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
LOAD_TF = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.Grayscale(1), T.ToTensor()])

# Fixed generators (plain U-Net) used for the deformation-magnitude section.
PLAIN_GENERATORS = {
    'U_PUBLISHED': 'networks/generator_lowest_total_loss_mu_0.01.pth',
    'I_M2_pre_adversarial': 'networks/pretrained_generator_prichexy_net.pth',
    'D_BDEV_certified': 'research_runs/M2_S1/B_dev/seed_42/generator_best_method_neutral.pth',
}

# (generator, its OWN co-adapted verifier) pairs. The certified B_dev run did
# not persist its verifier, so only the V2 runs can be measured this way; both
# were trained with the stock upstream privacy objective.
COADAPTED_PAIRS = [
    ('v2_Uinit_selected', 'archive/v2_attention_feat1_Uinit_run1',
     'generator_lowest_total_loss.pth', 'ver_model_trained_lowest_total_loss.pth'),
    ('v2_Uinit_epoch250', 'archive/v2_attention_feat1_Uinit_run1',
     'generator_latest.pth', 'ver_model_trained_latest.pth'),
    ('v2_run1_selected', 'archive/v2_attention_feat1_run1',
     'generator_lowest_total_loss.pth', 'ver_model_trained_lowest_total_loss.pth'),
    ('v2_run1_epoch250', 'archive/v2_attention_feat1_run1',
     'generator_latest.pth', 'ver_model_trained_latest.pth'),
]

# Fresh-attacker references, read from the sealed P0 manifests rather than
# retyped, so this file cannot drift from the governed numbers.
P0_FRESH_ATTACKER_DIRS = {
    'v2_Uinit_selected': 'reproduction/p0_bridge/runs_v2diag/V2_UINIT',
    'v2_Uinit_epoch250': None,   # generator_latest was never run through P0
    'v2_run1_selected': None,
    'v2_run1_epoch250': None,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_pairs(path):
    out = []
    for line in open(path):
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            out.append((parts[0], parts[1], float(parts[2])))
    return out


# --------------------------------------------------------------------------
# D1 -- the objective is exactly AUC-invariant (pure numerics, no data needed)
# --------------------------------------------------------------------------
def d1_auc_invariance():
    """Shift a fixed, genuinely separable logit vector and watch L_priv collapse
    to zero while ROC AUC does not move at all."""
    g = torch.Generator().manual_seed(0)
    logits = torch.cat([torch.randn(500, generator=g) + 1.5,
                        torch.randn(500, generator=g) - 1.5])
    y = [1] * 500 + [0] * 500
    rows = []
    for shift in (0.0, -2.0, -5.0, -10.0, -20.0):
        z = logits + shift
        rows.append({
            'logit_shift': shift,
            'privacy_term_softplus_mean': float(torch.nn.functional.softplus(z).mean()),
            'ver_loss_sigmoid_mean': float(torch.sigmoid(z).mean()),
            'roc_auc': float(roc_auc_score(y, z.numpy())),
        })
    aucs = {round(r['roc_auc'], 12) for r in rows}
    return {
        'rows': rows,
        'auc_is_constant_across_all_shifts': len(aucs) == 1,
        'privacy_term_range': [rows[0]['privacy_term_softplus_mean'],
                               rows[-1]['privacy_term_softplus_mean']],
    }


# --------------------------------------------------------------------------
# D2 -- pair-pool audit: rules out memorization and base-rate explanations
# --------------------------------------------------------------------------
def d2_pair_pool_audit():
    def patients(path):
        s = set()
        for a, b, _ in read_pairs(path):
            s.add(a.split('_')[0])
            s.add(b.split('_')[0])
        return s
    tr, va = patients(TRAIN_PAIRS), patients(VAL_PAIRS)
    labels = Counter(lbl for _, _, lbl in read_pairs(VAL_PAIRS))
    n = sum(labels.values())
    return {
        'train_patients': len(tr),
        'val_patients': len(va),
        'patient_overlap_train_val': len(tr & va),
        'val_positive_rate': labels.get(1.0, 0) / n,
        'val_n_pairs': n,
        'train_pairs_sha256': sha256_file(TRAIN_PAIRS),
        'val_pairs_sha256': sha256_file(VAL_PAIRS),
    }


# --------------------------------------------------------------------------
# shared image / anonymization helpers
# --------------------------------------------------------------------------
def _grid_components(device):
    d = torch.linspace(-1, 1, IMAGE_SIZE)
    mx, my = torch.meshgrid((d, d), indexing='ij')
    gid = torch.stack((my, mx), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    return gid, GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)


def _load_batch(rows, device):
    x1 = torch.stack([LOAD_TF(Image.open(IMAGE_ROOT + r[0]).convert('L')) for r in rows])
    x2 = torch.stack([LOAD_TF(Image.open(IMAGE_ROOT + r[1]).convert('L')) for r in rows])
    return x1.to(device), x2.to(device)


def _anonymize(gen, x, gid, gauss):
    flow = gen(x)
    grid = gid - MU * flow
    grid = gauss(grid)
    grid = grid.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(x, grid, padding_mode='border',
                                           align_corners=True), flow


# --------------------------------------------------------------------------
# D3 -- deformation magnitude: is the generator even deforming?
# --------------------------------------------------------------------------
def d3_deformation(device, n_images=24):
    gid, gauss = _grid_components(device)
    rows = read_pairs(VAL_PAIRS)[:n_images]
    out = {}
    for name, rel in PLAIN_GENERATORS.items():
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            out[name] = {'error': 'missing', 'path': rel}
            continue
        gen = UNet(1, 2, 32).to(device)
        gen.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        gen.eval()
        with torch.no_grad():
            x1, _ = _load_batch(rows, device)
            anon, flow = _anonymize(gen, x1, gid, gauss)
            out[name] = {
                'checkpoint': rel,
                'sha256': sha256_file(path),
                'mean_abs_flow': float(flow.abs().mean()),
                'max_displacement_px': float((MU * flow).abs().max()) * (IMAGE_SIZE / 2),
                'mean_abs_pixel_change': float((anon - x1).abs().mean()),
                'n_images': n_images,
            }
        del gen
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# D4 -- the headline: training metric vs the AUC it is supposed to represent
# --------------------------------------------------------------------------
def d4_objective_vs_auc(device):
    gid, gauss = _grid_components(device)
    rows = read_pairs(VAL_PAIRS)
    out = {}
    for tag, run, gck, vck in COADAPTED_PAIRS:
        gpath = os.path.join(ROOT, run, gck)
        vpath = os.path.join(ROOT, run, vck)
        if not (os.path.exists(gpath) and os.path.exists(vpath)):
            out[tag] = {'error': 'missing', 'generator': gpath, 'verifier': vpath}
            continue
        gen = UNetAtt(1, 2, 32).to(device)
        gen.load_state_dict(torch.load(gpath, map_location=device, weights_only=False))
        gen.eval()
        ver = SiameseNetwork().to(device)
        ver.load_state_dict(torch.load(vpath, map_location=device, weights_only=False))
        ver.eval()

        scores, labels = [], []
        with torch.no_grad():
            for i in range(0, len(rows), BATCH):
                chunk = rows[i:i + BATCH]
                x1, x2 = _load_batch(chunk, device)
                anon, _ = _anonymize(gen, x1, gid, gauss)
                z = ver(IMAGENET_NORM(anon.expand(-1, 3, -1, -1)),
                        IMAGENET_NORM(x2.expand(-1, 3, -1, -1))).squeeze(-1)
                scores += [float(v) for v in torch.sigmoid(z).float().cpu()]
                labels += [r[2] for r in chunk]
        out[tag] = {
            'run': run,
            'generator_checkpoint': gck,
            'generator_sha256': sha256_file(gpath),
            'verifier_checkpoint': vck,
            'verifier_sha256': sha256_file(vpath),
            # This is exactly the quantity the training loop logs as ver_loss.
            'ver_loss_mean_sigmoid': st.mean(scores),
            # This is what the paper and the S1 gate actually report as privacy.
            'roc_auc': float(roc_auc_score(labels, scores)),
            'n_pairs': len(labels),
            'geometry': 'anon(x1) vs real(x2)',
        }
        del gen, ver
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# D5 -- best-response gap, using the sealed P0 fresh-attacker manifests
# --------------------------------------------------------------------------
def d5_best_response_gap(d4):
    out = {}
    for tag, rel in P0_FRESH_ATTACKER_DIRS.items():
        if rel is None or tag not in d4 or 'roc_auc' not in d4[tag]:
            continue
        base = os.path.join(ROOT, rel)
        if not os.path.isdir(base):
            continue
        fresh = []
        for seed in sorted(os.listdir(base)):
            man = os.path.join(base, seed, 'run_manifest.json')
            if os.path.exists(man):
                fresh.append(json.load(open(man))['raw_roc_auc'])
        if not fresh:
            continue
        coad = d4[tag]['roc_auc']
        out[tag] = {
            'coadapted_training_verifier_auc': coad,
            'fresh_adaptive_attacker_auc_mean': st.mean(fresh),
            'fresh_adaptive_attacker_n_seeds': len(fresh),
            'best_response_gap_fresh_minus_coadapted': st.mean(fresh) - coad,
            'weak_critic_hypothesis_supported': st.mean(fresh) > coad,
            'p0_source_dir': rel,
        }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def git_head():
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=ROOT).decode().strip()
        except Exception:
            return None

    print('Running privacy-objective diagnosis on %s ...' % device)
    d1 = d1_auc_invariance()
    d2 = d2_pair_pool_audit()
    d3 = d3_deformation(device)
    d4 = d4_objective_vs_auc(device)
    d5 = d5_best_response_gap(d4)

    result = {
        'schema': 'PRIVACY_OBJECTIVE_DIAGNOSIS_V1',
        'method_uncertified': True,
        'test_split_accessed': False,
        'git_head': git_head(),
        'torch': torch.__version__,
        'device': str(device),
        'objective_under_test': 'L_priv = -log(1 - sigmoid(z)) = softplus(z)',
        'objective_source_files': {
            p: sha256_file(os.path.join(ROOT, p))
            for p in ('utils/VerificationLoss.py', 'utils/utils.py')
        },
        'D1_auc_invariance': d1,
        'D2_pair_pool_audit': d2,
        'D3_deformation_magnitude': d3,
        'D4_objective_vs_auc': d4,
        'D5_best_response_gap': d5,
    }
    out_path = os.path.join(OUT_DIR, 'privacy_objective_diagnosis.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    # ---- human-readable summary ----
    print('\nD1  objective is AUC-invariant: %s  (L_priv %.4f -> %.6f)'
          % (d1['auc_is_constant_across_all_shifts'], *d1['privacy_term_range']))
    print('D2  train/val patient overlap = %d ; val positive rate = %.3f'
          % (d2['patient_overlap_train_val'], d2['val_positive_rate']))
    print('\nD3  deformation magnitude')
    print('    %-26s %12s %12s' % ('generator', 'mean|flow|', 'mean|dpix|'))
    for k, v in d3.items():
        if 'error' not in v:
            print('    %-26s %12.5f %12.6f'
                  % (k, v['mean_abs_flow'], v['mean_abs_pixel_change']))
    print('\nD4  training metric vs the AUC it stands for')
    print('    %-22s %22s %10s' % ('checkpoint', 'ver_loss (logged)', 'ROC AUC'))
    for k, v in d4.items():
        if 'error' not in v:
            print('    %-22s %22.6f %10.4f'
                  % (k, v['ver_loss_mean_sigmoid'], v['roc_auc']))
    print('\nD5  best-response gap (fresh adaptive attacker - co-adapted critic)')
    for k, v in d5.items():
        print('    %-22s coadapted %.4f  fresh %.4f (n=%d)  gap %+.4f  weak-critic supported: %s'
              % (k, v['coadapted_training_verifier_auc'],
                 v['fresh_adaptive_attacker_auc_mean'],
                 v['fresh_adaptive_attacker_n_seeds'],
                 v['best_response_gap_fresh_minus_coadapted'],
                 v['weak_critic_hypothesis_supported']))
    print('\nWrote %s' % out_path)


if __name__ == '__main__':
    main()
