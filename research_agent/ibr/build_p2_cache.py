"""STEP 7C — P2 Utility-Constrained Atlas Normalization cache builder.

Frozen P2 transform protocol:
  1. Load P1 3x64x64 binary masks from research_agent/ibr_s1_condition_p1_cache/.
  2. Convert each TRAIN binary mask into a signed Euclidean distance field:
     D_src = edt(M) - edt(1 - M) (positive inside, negative outside).
  3. Compute TRAIN-ONLY population atlas: D_atlas = mean_{TRAIN}(D_src) per structure.
  4. Search for largest lambda in [0, 1] satisfying TRAIN retention constraints:
       - TRAIN mean macro Dice >= 0.940
       - TRAIN mean Dice for EACH structure >= 0.930
     where M_lambda = 1[(1 - lambda) * D_src + lambda * D_atlas >= 0].
  5. Freeze lambda*.
  6. Apply frozen lambda* to TRAIN and VALIDATION masks -> train_p2.npy, val_p2.npy (3x64x64 uint8 binary).
  7. Evaluate VALIDATION retention vs original teacher hard masks (upsampled nearest-neighbor 64 -> 256).
  8. Write p2_meta.json and anatomy_retention.json in research_agent/ibr_s1_condition_p2_cache/.
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage

P1_CACHE_DIR = 'research_agent/ibr_s1_condition_p1_cache/'
P2_CACHE_DIR = 'research_agent/ibr_s1_condition_p2_cache/'
STEP7A_CACHE_DIR = 'research_agent/ibr_s1_condition_cache/'
SEG_SHA256_FULL = '2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0'
STRUCTS = ['Left Lung', 'Right Lung', 'Heart']


def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()


def compute_signed_distance_batch(masks_bool):
    """masks_bool: (N, C, H, W) boolean numpy array.

    Returns: (N, C, H, W) float32 signed distance fields.
    Convention: positive inside, negative outside.
    """
    N, C, H, W = masks_bool.shape
    sd = np.zeros((N, C, H, W), dtype=np.float32)
    for i in range(N):
        for c in range(C):
            m = masks_bool[i, c]
            if not m.any():
                sd[i, c] = -ndimage.distance_transform_edt(~m).astype(np.float32)
            elif m.all():
                sd[i, c] = ndimage.distance_transform_edt(m).astype(np.float32)
            else:
                pos = ndimage.distance_transform_edt(m)
                neg = ndimage.distance_transform_edt(~m)
                sd[i, c] = (pos - neg).astype(np.float32)
    return sd


def compute_dice_fast(m_true_bool, m_pred_bool):
    """m_true_bool, m_pred_bool: (N, C, H, W) boolean.

    Returns: macro_dice (float), per_structure_dices (list of 3 floats).
    """
    per_struct = []
    for c in range(3):
        a = m_true_bool[:, c]
        b = m_pred_bool[:, c]
        inter = (a & b).sum(axis=(1, 2))
        total = a.sum(axis=(1, 2)) + b.sum(axis=(1, 2))
        dice_c = float((2.0 * inter / np.maximum(total, 1e-8)).mean())
        per_struct.append(dice_c)
    macro = float(np.mean(per_struct))
    return macro, per_struct


def calibrate_lambda_on_train(train_sd, train_p1_bool, atlas):
    """Bisection search to find the largest lambda in [0, 1] satisfying TRAIN retention:

    - macro Dice >= 0.940
    - per-structure Dice >= 0.930
    """
    low = 0.0
    high = 1.0
    best_lam = 0.0
    best_stats = None
    curve = []

    # Dense grid check for logging curve
    for lam_test in np.linspace(0.0, 1.0, 11):
        d_lam = (1.0 - lam_test) * train_sd + lam_test * atlas[None, :]
        m_lam = (d_lam >= 0)
        macro, per_s = compute_dice_fast(train_p1_bool, m_lam)
        valid = (macro >= 0.940) and all(d >= 0.930 for d in per_s)
        curve.append({'lambda': round(float(lam_test), 2), 'macro_dice': round(macro, 5),
                      'per_structure_dice': [round(d, 5) for d in per_s], 'valid': valid})

    # High-precision bisection (16 iterations)
    for _ in range(16):
        mid = (low + high) / 2.0
        d_lam = (1.0 - mid) * train_sd + mid * atlas[None, :]
        m_lam = (d_lam >= 0)
        macro, per_s = compute_dice_fast(train_p1_bool, m_lam)
        valid = (macro >= 0.940) and all(d >= 0.930 for d in per_s)
        if valid:
            best_lam = mid
            best_stats = (macro, per_s)
            low = mid  # try larger lambda
        else:
            high = mid  # lambda too large

    frozen_lambda = round(float(best_lam), 4)
    print('FROZEN lambda* calibrated on TRAIN:', frozen_lambda)
    print('TRAIN retention at lambda*: macro Dice=%.5f, per-structure=%s'
          % (best_stats[0], [round(x, 5) for x in best_stats[1]]))
    return frozen_lambda, best_stats, curve


def hd95(mask_a, mask_b):
    """Hausdorff 95th percentile distance between two 2D binary masks."""
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float('nan')
    da = ndimage.distance_transform_edt(mask_b == 0)
    db = ndimage.distance_transform_edt(mask_a == 0)
    d_a_to_b = da[mask_a]
    d_b_to_a = db[mask_b]
    return float(max(np.percentile(d_a_to_b, 95), np.percentile(d_b_to_a, 95)))


def evaluate_validation_retention(val_p2):
    """Compare val_p2 upsampled 64->256 nearest-neighbor against original teacher hard masks."""
    teacher_maps = np.load(os.path.join(STEP7A_CACHE_DIR, 'val_maps.npy'))
    N = teacher_maps.shape[0]
    assert N == val_p2.shape[0]

    M_teacher = (teacher_maps >= 0.5).astype(np.uint8)
    val_p2_up = np.repeat(np.repeat(val_p2, 4, axis=2), 4, axis=3)

    stats = {}
    for c in range(3):
        dice, iou, hd = [], [], []
        for i in range(N):
            a = M_teacher[i, c].astype(bool)
            b = val_p2_up[i, c].astype(bool)
            inter = np.logical_and(a, b).sum()
            uni = np.logical_or(a, b).sum()
            if a.sum() + b.sum() == 0:
                continue
            dice.append(2.0 * inter / (a.sum() + b.sum() + 1e-8))
            iou.append(inter / (uni + 1e-8))
            hd.append(hd95(a, b))
        stats[STRUCTS[c]] = {
            'dice_mean': float(np.mean(dice)), 'dice_sd': float(np.std(dice, ddof=1)),
            'iou_mean': float(np.mean(iou)), 'iou_sd': float(np.std(iou, ddof=1)),
            'hd95_mean_px': float(np.mean(hd)), 'hd95_sd_px': float(np.std(hd, ddof=1)),
        }
        print('%s: Val Dice %.4f IoU %.4f HD95 %.2f px'
              % (STRUCTS[c], stats[STRUCTS[c]]['dice_mean'],
                 stats[STRUCTS[c]]['iou_mean'],
                 stats[STRUCTS[c]]['hd95_mean_px']))

    mean_dice = float(np.mean([stats[s]['dice_mean'] for s in STRUCTS]))
    mean_iou = float(np.mean([stats[s]['iou_mean'] for s in STRUCTS]))
    mean_hd = float(np.mean([stats[s]['hd95_mean_px'] for s in STRUCTS]))

    print('VALIDATION MEAN retention: Dice %.4f IoU %.4f HD95 %.2f px'
          % (mean_dice, mean_iou, mean_hd))

    res = {
        'val_n': N,
        'per_structure': stats,
        'mean_dice': mean_dice,
        'mean_iou': mean_iou,
        'mean_hd95_px': mean_hd,
        'passed_retention_target': bool(mean_dice >= 0.930),
        'method': 'val_p2 upsampled 64->256 via nearest-neighbor vs original teacher hard masks'
    }
    return res


def main():
    os.makedirs(P2_CACHE_DIR, exist_ok=True)
    t0 = time.time()

    print('Loading P1 binary masks...')
    train_p1 = np.load(os.path.join(P1_CACHE_DIR, 'train_p1.npy'))  # (18089, 3, 64, 64) uint8
    val_p1 = np.load(os.path.join(P1_CACHE_DIR, 'val_p1.npy'))      # (3484, 3, 64, 64) uint8

    train_p1_bool = (train_p1 > 0)
    val_p1_bool = (val_p1 > 0)

    print('Computing signed distance fields for TRAIN (18,089 images)...')
    train_sd = compute_signed_distance_batch(train_p1_bool)

    print('Computing TRAIN population atlas D_atlas...')
    atlas = train_sd.mean(axis=0)  # (3, 64, 64) float32
    np.save(os.path.join(P2_CACHE_DIR, 'train_atlas.npy'), atlas)
    atlas_sha = hashlib.sha256(open(os.path.join(P2_CACHE_DIR, 'train_atlas.npy'), 'rb').read()).hexdigest()

    print('Calibrating lambda* on TRAIN retention...')
    frozen_lambda, train_ret_stats, curve = calibrate_lambda_on_train(train_sd, train_p1_bool, atlas)

    print('Applying P2 transform (lambda* = %.4f) to TRAIN...' % frozen_lambda)
    train_d_lam = (1.0 - frozen_lambda) * train_sd + frozen_lambda * atlas[None, :]
    train_p2 = (train_d_lam >= 0).astype(np.uint8)
    np.save(os.path.join(P2_CACHE_DIR, 'train_p2.npy'), train_p2)

    print('Computing signed distance fields for VALIDATION (3,484 images)...')
    val_sd = compute_signed_distance_batch(val_p1_bool)

    print('Applying P2 transform (frozen lambda* = %.4f) to VALIDATION...' % frozen_lambda)
    val_d_lam = (1.0 - frozen_lambda) * val_sd + frozen_lambda * atlas[None, :]
    val_p2 = (val_d_lam >= 0).astype(np.uint8)
    np.save(os.path.join(P2_CACHE_DIR, 'val_p2.npy'), val_p2)

    print('Evaluating VALIDATION anatomy retention vs original teacher hard masks...')
    val_ret_res = evaluate_validation_retention(val_p2)
    with open(os.path.join(P2_CACHE_DIR, 'anatomy_retention.json'), 'w') as f:
        json.dump(val_ret_res, f, indent=2)

    meta = {
        'step': '7C',
        'p2_transform': {
            'description': 'D_lambda = (1 - lambda) * D_src + lambda * D_atlas; M_lambda = 1[D_lambda >= 0]',
            'signed_distance_convention': 'positive inside, negative outside',
            'train_atlas_file': 'train_atlas.npy',
            'train_atlas_sha256': atlas_sha,
            'frozen_lambda': frozen_lambda,
            'calibration': {
                'train_macro_dice_target': 0.940,
                'train_per_structure_dice_target': 0.930,
                'train_macro_dice_achieved': train_ret_stats[0],
                'train_per_structure_dice_achieved': train_ret_stats[1],
                'search_curve': curve
            }
        },
        'source_teacher': {
            'path': 'archive/train_seg_unet/best.pth',
            'sha256_full': SEG_SHA256_FULL,
        },
        'train_p2': {'file': 'train_p2.npy', 'shape': list(train_p2.shape), 'dtype': str(train_p2.dtype)},
        'val_p2': {'file': 'val_p2.npy', 'shape': list(val_p2.shape), 'dtype': str(val_p2.dtype)},
        'validation_retention': val_ret_res,
        'built_at': datetime.now(timezone.utc).isoformat(),
        'commit': git_head()
    }

    with open(os.path.join(P2_CACHE_DIR, 'p2_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('P2 CACHE BUILD COMPLETE in %.2fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
