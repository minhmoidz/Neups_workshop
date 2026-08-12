"""STEP 4A PART E - utility-by-band on VALIDATION (frozen classifier + frozen segmenter).

For the three band variants of the corrected-baseline anonymized VALIDATION images,
    original   : corrected-baseline deformed image
    low_pass   : LP( deformed image )
    high_pass  : HP( deformed image )
measure utility with the SAME frozen models used in STEP 3C:

  (a) Classification: DenseNet-121 (networks/pretrained_classifier.pth), 14-label
      multi-label AUC (mean AUC-14). Trained on raw images; frozen at eval.
  (b) Segmentation  : UNetSeg (archive/train_seg_unet/best.pth, init_features from
      checkpoint), Dice / IoU / HD95 over Left Lung, Right Lung, Heart.

Pipeline fidelity (mirrors eval_classifier.py / eval_seg.py):
  - classification images: CXRDataset fold='val' with the flow_field path
    (Resize 256 -> ToTensor, 1 channel), then corrected deform, then band variant,
    then expand to 3 channels + Resize 224 + ImageNet normalization (as eval_model).
  - segmentation images: SegDataset fold='val' (1ch 256, normalized to [-1,1]),
    corrected deform, band variant, then UNetSeg (prob > 0.5).
  - Band transform is inserted ONLY at the deformed-image stage (the mechanism under
    test). No model is trained here; both models are frozen. VALIDATION ONLY.

Output: research_agent/05A_artifacts/utility_by_band.json
"""

import argparse
import json
import os

import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from utils import utils
from utils.GaussianSmoothing import GaussianSmoothing
from research_agent import band

LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
          'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
NAMES = ['Left Lung', 'Right Lung', 'Heart']

OUT = 'research_agent/05A_artifacts/utility_by_band.json'


def build_deform_fn(mu=0.01, device='cuda'):
    gen = utils.load_flow_generator('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth').to(device)
    gen.eval()
    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)

    def fn(images):
        return utils.deform(images, gen, grid_identity, gauss, mu, 0.0, 'corrected')

    return fn


def eval_classification(deform_fn, band_fn, device):
    from chexnet.cxr_dataset import CXRDataset
    checkpoint = torch.load('networks/pretrained_classifier.pth', weights_only=False, map_location='cpu')
    model = checkpoint['model'].to(device)
    model.eval()

    ds = CXRDataset(path_to_images='/home/minhtt/datasets/nih/images/', fold='val',
                    perturbation_type='flow_field')
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)

    resizing = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    y_true = []
    y_scores = []
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            inputs = deform_fn(inputs)
            if band_fn is not None:
                inputs = band_fn(inputs)
            inputs = inputs.expand(-1, 3, -1, -1)
            inputs = resizing(inputs)
            inputs = normalize(inputs)
            outputs = torch.sigmoid(model(inputs))
            y_scores.append(outputs.cpu().numpy())
            y_true.append(labels.numpy())

    y_scores = np.concatenate(y_scores, 0)
    y_true = np.concatenate(y_true, 0)
    auc_by_label = {}
    for i, L in enumerate(LABELS):
        try:
            auc_by_label[L] = float(roc_auc_score(y_true[:, i].astype(int), y_scores[:, i]))
        except ValueError:
            auc_by_label[L] = None
    mean_auc_14 = float(np.nanmean([a for a in auc_by_label.values() if a is not None]))
    return {'mean_auc_14': mean_auc_14, 'auc_by_label': auc_by_label, 'n_cases': int(len(y_true))}


def hd95(pred, target):
    from scipy.ndimage import distance_transform_edt
    if pred.sum() == 0 or target.sum() == 0:
        return float('nan')
    d_pred = distance_transform_edt(pred == 0)
    d_target = distance_transform_edt(target == 0)
    fwd = d_target[pred > 0]
    bwd = d_pred[target > 0]
    return float(max(np.percentile(fwd, 95), np.percentile(bwd, 95)))


def eval_segmentation(deform_fn, band_fn, device):
    from chexnet.seg_dataset import SegDataset
    from networks.UNetSeg import UNetSeg
    chk = torch.load('archive/train_seg_unet/best.pth', weights_only=False, map_location='cpu')
    model = UNetSeg(in_channels=1, out_channels=3,
                    init_features=32 if 'init_features' not in chk else chk['init_features']).to(device)
    model.load_state_dict(chk['model'])
    model.eval()

    ds = SegDataset('val', '/home/minhtt/datasets/nih/images', subsample=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)

    sums = {'dice': np.zeros(3), 'iou': np.zeros(3), 'hd95': np.zeros(3)}
    n_cases = 0
    with torch.no_grad():
        for img, mask, _ in loader:
            img = img.to(device)
            img = deform_fn(img)
            if band_fn is not None:
                img = band_fn(img)
            pred = (model(img) > 0.5).detach().cpu().numpy()
            mask_np = mask.numpy()
            for b in range(img.size(0)):
                for c in range(3):
                    p = pred[b, c]
                    t = mask_np[b, c]
                    inter = (t * p).sum()
                    union = ((t + p) > 0).sum()
                    dice = (2 * inter + 1e-7) / (t.sum() + p.sum() + 1e-7)
                    iou = (inter + 1e-7) / (union + 1e-7)
                    hd = hd95(p, t)
                    sums['dice'][c] += dice
                    sums['iou'][c] += iou
                    if not np.isnan(hd):
                        sums['hd95'][c] += hd
                n_cases += 1

    return {
        'n_cases': n_cases,
        'per_structure': {
            n: {
                'dice': float(sums['dice'][c] / n_cases),
                'iou': float(sums['iou'][c] / n_cases),
                'hd95': float(sums['hd95'][c] / n_cases),
            } for c, n in enumerate(NAMES)
        },
        'mean_dice': float(sums['dice'].mean() / n_cases),
        'mean_iou': float(sums['iou'].mean() / n_cases),
        'mean_hd95': float(sums['hd95'].mean() / n_cases),
    }


def main():
    parser = argparse.ArgumentParser('STEP 4A PART E utility-by-band (VALIDATION)')
    parser.add_argument('--out', default=OUT)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deform_fn = build_deform_fn(mu=0.01, device=device)

    band_fns = {
        'original': None,
        'low_pass': lambda t: band.low_pass(t),
        'high_pass': lambda t: band.high_pass(t),
    }

    results = {'transform_mode': 'corrected', 'mu': 0.01, 'stochastic_lambda': 0.0, 'bands': {}}
    for name, fn in band_fns.items():
        print('--- classification band =', name)
        clf = eval_classification(deform_fn, fn, device)
        print('    mean_auc_14 = %.4f (n=%d)' % (clf['mean_auc_14'], clf['n_cases']))
        print('--- segmentation band =', name)
        seg = eval_segmentation(deform_fn, fn, device)
        print('    dice=%.4f iou=%.4f hd95=%.3f (n=%d)' % (seg['mean_dice'], seg['mean_iou'], seg['mean_hd95'], seg['n_cases']))
        results['bands'][name] = {'classification': clf, 'segmentation': seg}

    results['band_lowpass_sigma'] = band.SIGMA_LP
    results['_label'] = 'UTILITY-BY-BAND on VALIDATION (frozen classifier + frozen segmenter; no privacy number)'
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print('wrote', args.out)


if __name__ == '__main__':
    main()