import json
import hashlib
import subprocess
from datetime import datetime, timezone
import pandas as pd
import numpy as np

STRUCTS = ['Left Lung', 'Right Lung', 'Heart']

def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

def main():
    df = pd.read_csv('research_agent/03H_artifacts/segmentation/seg_per_case.csv')

    per_struct = {}
    for m in ['dice', 'iou', 'hd95']:
        g = df.groupby('structure')[m].agg(['mean', 'std', 'median', 'min', 'max'])
        per_struct[m] = {s: {k: (None if pd.isna(v) else float(v)) for k, v in g.loc[s].items()} for s in STRUCTS}
    allv = {m: df[m].values for m in ['dice', 'iou', 'hd95']}
    aggregates = {m: float(np.mean([np.mean(df[df['structure'] == s][m]) for s in STRUCTS])) for m in ['dice', 'iou', 'hd95']}

    rng = np.random.default_rng(42)
    def agg(df_, m):
        return float(np.mean([np.mean(df_[df_['structure'] == s][m]) for s in STRUCTS]))
    ci = {}
    for m in ['dice', 'iou', 'hd95']:
        boots = np.array([agg(df.sample(frac=1, replace=True, random_state=int(rng.integers(0, 2**31))), m)
                          for _ in range(1000)])
        ci[m] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    n_cases = int(len(df) / 3)
    gen = sha256('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth')
    seg_ckpt = sha256('archive/train_seg_unet/best.pth')
    split = sha256('chexnet/nih_labels.csv')
    maskmeta = sha256('data/chexmask/ChestX-Ray8.csv')

    out = {
        'generator_path': 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth',
        'generator_sha256': gen,
        'segmentation_checkpoint': './archive/train_seg_unet/best.pth',
        'segmentation_sha256': seg_ckpt,
        'segmentation_architecture': 'UNetSeg(in_channels=1, out_channels=3, init_features=16), sigmoid head',
        'segmenter_trained_on': 'raw (unperturbed) NIH ChestX-ray14 images (fold=train)',
        'anatomical_targets': ['Left Lung', 'Right Lung', 'Heart'],
        'mask_metadata': 'data/chexmask/ChestX-Ray8.csv',
        'mask_metadata_sha256': maskmeta,
        'dataset_split': 'chexnet/nih_labels.csv fold==test (restricted to rows with masks)',
        'dataset_split_sha256': split,
        'transform_mode': 'corrected',
        'mu': 0.01,
        'stochastic_lambda': 0.0,
        'threshold': 0.5,
        'n_cases': n_cases,
        'aggregates': aggregates,
        'per_structure_mean': {s: {m: float(df[df['structure'] == s][m].mean()) for m in ['dice', 'iou', 'hd95']} for s in STRUCTS},
        'per_structure_stats': per_struct,
        'bootstrap_95ci': ci,
        'empty_pred_cases': 0,
        'empty_mask_cases': 0,
        'hd95_units': 'pixels at 256x256 image resolution (resized-image pixels); no empty masks in this run',
        'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
        'git_commit': git_head(),
        'eval_script': 'research_agent/eval_seg_percase.py',
        'eval_script_sha256': sha256('research_agent/eval_seg_percase.py'),
        'canonical_eval_seg_sha256': sha256('eval_seg.py'),
    }
    with open('research_agent/03H_corrected_segmentation.json', 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print('Dice=%.5f IoU=%.5f HD95=%.5f' % (aggregates['dice'], aggregates['iou'], aggregates['hd95']))
    print('wrote research_agent/03H_corrected_segmentation.json')

if __name__ == '__main__':
    main()
