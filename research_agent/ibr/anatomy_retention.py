"""STEP 7B — anatomy-retention diagnostic (NOT the final segmentation metric).

Upsamples M64 back to 256x256 with NEAREST-NEIGHBOR ONLY, compares against the
original teacher hard masks (M = 1[p>=0.5]). Reports per structure (Left Lung,
Right Lung, Heart) on the validation split: Dice, IoU, and HD95 (boundary-based,
computed via a cheap contour-distance approximation on the 4x4 grid geometry).

Diagnostic only; never used to tune P1 or select anything.
"""

import json

import numpy as np
from scipy import ndimage

from research_agent.ibr.build_p1_cache import p1_transform

CACHE_DIR = 'research_agent/ibr_s1_condition_cache/'
P1_CACHE_DIR = 'research_agent/ibr_s1_condition_p1_cache/'
STRUCTS = ['Left Lung', 'Right Lung', 'Heart']


def _hd95(mask_a, mask_b):
    """Hausdorff 95th percentile between two 2D binary masks (approx, grid-based).

    Uses 4-connectivity distance transforms via ndimage.distance_transform_edt.
    """
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float('nan')
    da = ndimage.distance_transform_edt(mask_b == 0)
    db = ndimage.distance_transform_edt(mask_a == 0)
    d_a_to_b = da[mask_a]
    d_b_to_a = db[mask_b]
    return float(max(np.percentile(d_a_to_b, 95), np.percentile(d_b_to_a, 95)))


def main():
    maps = np.load('%sval_maps.npy' % CACHE_DIR)
    M64 = np.load('%sval_p1.npy' % P1_CACHE_DIR)
    assert maps.shape[0] == M64.shape[0]
    N = maps.shape[0]

    M = (maps >= 0.5).astype(np.uint8)
    # nearest-neighbor upsample 64 -> 256: repeat each cell 4x in each dim
    M64_up = np.repeat(np.repeat(M64, 4, axis=2), 4, axis=3)

    stats = {}
    for c in range(3):
        dice, iou, hd = [], [], []
        for i in range(N):
            a = M[i, c].astype(bool)
            b = M64_up[i, c].astype(bool)
            inter = np.logical_and(a, b).sum()
            uni = np.logical_or(a, b).sum()
            if a.sum() + b.sum() == 0:
                continue
            dice.append(2 * inter / (a.sum() + b.sum() + 1e-8))
            iou.append(inter / (uni + 1e-8))
            hd.append(_hd95(a, b))
        stats[STRUCTS[c]] = {
            'dice_mean': float(np.mean(dice)), 'dice_sd': float(np.std(dice, ddof=1)),
            'iou_mean': float(np.mean(iou)), 'iou_sd': float(np.std(iou, ddof=1)),
            'hd95_mean_px': float(np.mean(hd)), 'hd95_sd_px': float(np.std(hd, ddof=1)),
        }
        print('%s: Dice %.4f IoU %.4f HD95 %.2f px' % (STRUCTS[c], stats[STRUCTS[c]]['dice_mean'],
                                                       stats[STRUCTS[c]]['iou_mean'],
                                                       stats[STRUCTS[c]]['hd95_mean_px']))
    mean_dice = np.mean([stats[s]['dice_mean'] for s in STRUCTS])
    mean_iou = np.mean([stats[s]['iou_mean'] for s in STRUCTS])
    mean_hd = np.mean([stats[s]['hd95_mean_px'] for s in STRUCTS])
    print('MEAN across structures: Dice %.4f IoU %.4f HD95 %.2f px' % (mean_dice, mean_iou, mean_hd))
    out = {'val_n': N, 'per_structure': stats,
           'mean_dice': float(mean_dice), 'mean_iou': float(mean_iou),
           'mean_hd95_px': float(mean_hd),
           'method': 'M64 upsampled to 256 via nearest-neighbor, vs original teacher hard masks'}
    with open('research_agent/ibr_s1_condition_p1_cache/anatomy_retention.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('SAVED anatomy_retention.json')


if __name__ == '__main__':
    main()