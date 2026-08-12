"""STEP 4B PART 3 — VALIDATION utility sanity for the two CAA diagnostic transforms.

For each condition of the corrected-baseline anonymized VALIDATION images,
    reference  : corrected-baseline deformed image (no extra transform)
    border     : H3 border_normalize(BW=4) applied to deformed image
    intensity  : H2 intensity_normalize(p1/p99) applied to deformed image
measure utility with the SAME frozen models as STEP 3C:

  (a) Classification DenseNet-121 (networks/pretrained_classifier.pth): mean AUC-14.
  (b) Segmentation UNetSeg (archive/train_seg_unet/best.pth): mean Dice.

Pipeline fidelity: classification uses the CXRDataset flow_field path (Resize 256 ->
1ch [0,1]) -> corrected deform -> transform -> expand 3ch -> Resize 224 -> ImageNet
normalization (as eval_model); segmentation uses SegDataset 'val' (1ch 256 in [-1,1])
-> corrected deform -> transform -> UNetSeg (prob > 0.5).

These are DEVELOPMENT diagnostics; no TEST non-inferiority statistics are applied.
Gross collapse flag uses the frozen screening gates: classification < 0.765 or Dice < 0.930.

Output: research_agent/05B_artifacts/utility_sanity.json
"""

import json
import os

import numpy as np
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from utils import utils
from utils.GaussianSmoothing import GaussianSmoothing
from research_agent import caa_transforms

LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
          'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
NAMES = ['Left Lung', 'Right Lung', 'Heart']
OUT = 'research_agent/05B_artifacts/utility_sanity.json'


def build_deform_fn(device):
    gen = utils.load_flow_generator('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth').to(device)
    gen.eval()
    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)

    def fn(images):
        return utils.deform(images, gen, grid_identity, gauss, 0.01, 0.0, 'corrected')
    return fn


def hd95(pred, target):
    from scipy.ndimage import distance_transform_edt
    if pred.sum() == 0 or target.sum() == 0:
        return float('nan')
    d_pred = distance_transform_edt(pred == 0)
    d_target = distance_transform_edt(target == 0)
    fwd = d_target[pred > 0]
    bwd = d_pred[target > 0]
    return float(max(np.percentile(fwd, 95), np.percentile(bwd, 95)))


def eval_classification(transform_name, device):
    from chexnet.cxr_dataset import CXRDataset
    checkpoint = torch.load('networks/pretrained_classifier.pth', weights_only=False, map_location='cpu')
    model = checkpoint['model'].to(device)
    model.eval()

    ds = CXRDataset(path_to_images='/home/minhtt/datasets/nih/images/', fold='val', perturbation_type='flow_field')
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)

    resizing = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    deform_fn = build_deform_fn(device)

    def compose(t):
        d = deform_fn(t)
        if transform_name == 'border':
            return caa_transforms.border_normalize(d)
        if transform_name == 'intensity':
            return caa_transforms.intensity_normalize(d, 0.0, 1.0)
        return d

    y_true = []
    y_scores = []
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device)
            inputs = compose(inputs)
            inputs = inputs.expand(-1, 3, -1, -1)
            inputs = resizing(inputs)
            inputs = normalize(inputs)
            outputs = torch.sigmoid(model(inputs))
            y_scores.append(outputs.cpu().numpy())
            y_true.append(labels.numpy())
    y_scores = np.concatenate(y_scores, 0)
    y_true = np.concatenate(y_true, 0)
    aucs = []
    for i, L in enumerate(LABELS):
        try:
            aucs.append(roc_auc_score(y_true[:, i].astype(int), y_scores[:, i]))
        except ValueError:
            aucs.append(float('nan'))
    return {'mean_auc_14': float(np.nanmean(aucs)), 'n_cases': int(len(y_true))}


def eval_segmentation(transform_name, device):
    from chexnet.seg_dataset import SegDataset
    from networks.UNetSeg import UNetSeg
    chk = torch.load('archive/train_seg_unet/best.pth', weights_only=False, map_location='cpu')
    model = UNetSeg(in_channels=1, out_channels=3,
                    init_features=32 if 'init_features' not in chk else chk['init_features']).to(device)
    model.load_state_dict(chk['model'])
    model.eval()

    ds = SegDataset('val', '/home/minhtt/datasets/nih/images', subsample=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=8)
    deform_fn = build_deform_fn(device)

    def compose(t):
        d = deform_fn(t)
        if transform_name == 'border':
            return caa_transforms.border_normalize(d)
        if transform_name == 'intensity':
            return caa_transforms.intensity_normalize(d, -1.0, 1.0)
        return d

    sums = {'dice': np.zeros(3), 'hd95': 0.0}
    n_cases = 0
    with torch.no_grad():
        for img, mask, _ in loader:
            img = img.to(device)
            img = compose(img)
            pred = (model(img) > 0.5).detach().cpu().numpy()
            mask_np = mask.numpy()
            for b in range(img.size(0)):
                for c in range(3):
                    p = pred[b, c]
                    t = mask_np[b, c]
                    inter = (t * p).sum()
                    sums['dice'][c] += (2 * inter + 1e-7) / (t.sum() + p.sum() + 1e-7)
                n_cases += 1
    return {'mean_dice': float(sums['dice'].mean() / n_cases), 'n_cases': n_cases}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {'transform_mode': 'corrected', 'mu': 0.01, '_label': 'DEVELOPMENT / MECHANISM DIAGNOSTIC (NOT A PAPER PRIVACY ESTIMATE)'}
    for name in ['reference', 'border', 'intensity']:
        clf = eval_classification(name, device)
        seg = eval_segmentation(name, device)
        results[name] = {
            'classification_mean_auc14': clf['mean_auc_14'],
            'classification_n_cases': clf['n_cases'],
            'segmentation_mean_dice': seg['mean_dice'],
            'segmentation_n_cases': seg['n_cases'],
            'gross_collapse': {
                'classification_below_0765': clf['mean_auc_14'] < 0.765,
                'dice_below_0930': seg['mean_dice'] < 0.930,
            },
        }
        print('%-12s  class AUC14 = %.4f   seg Dice = %.4f' % (name, clf['mean_auc_14'], seg['mean_dice']))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print('wrote', OUT)


if __name__ == '__main__':
    main()