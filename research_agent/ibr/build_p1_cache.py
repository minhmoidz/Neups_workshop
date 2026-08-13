"""STEP 7B — P1 capacity-limited anatomy condition.

Frozen P1 transform (no sweep):
    1. teacher prob map (256x256) -> hard mask  M = 1[p >= 0.5]
    2. partition into non-overlapping 4x4 blocks
    3. q = mean(block)
    4. M64 = 1[q >= 0.5]          -> shape 3x64x64, binary

Reads the exact STEP 7A cache (teacher SHA and source-ID provenance identical).
Writes P1 coarse cache (3x64x64 uint8 binary) for TRAIN/VAL. Never touches TEST.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

CACHE_DIR = 'research_agent/ibr_s1_condition_cache/'
P1_CACHE_DIR = 'research_agent/ibr_s1_condition_p1_cache/'
SEG_SHA256_FULL = '2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0'


def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()


def p1_transform(maps):
    """maps: (N, 3, 256, 256) prob in [0,1] -> (N, 3, 64, 64) binary uint8.

    Hard mask M = 1[p>=0.5]; block-mean 4x4 (256/64); M64 = 1[q>=0.5].
    """
    N, C, H, W = maps.shape
    assert (H, W) == (256, 256)
    M = (maps >= 0.5).astype(np.float32)          # (N,3,256,256) binary
    # reshape to (N,C,64,4,64,4), mean over the 4x4 axes
    q = M.reshape(N, C, 64, 4, 64, 4).mean(axis=(3, 5))  # (N,3,64,64)
    M64 = (q >= 0.5).astype(np.uint8)
    return M64


def build(fold):
    maps = np.load(os.path.join(CACHE_DIR, '%s_maps.npy' % fold))
    with open(os.path.join(CACHE_DIR, '%s_images.json' % fold)) as f:
        images = json.load(f)
    assert maps.shape[0] == len(images)
    M64 = p1_transform(maps)
    assert M64.shape == (len(images), 3, 64, 64)
    assert set(np.unique(M64)).issubset({0, 1})
    os.makedirs(P1_CACHE_DIR, exist_ok=True)
    np.save(os.path.join(P1_CACHE_DIR, '%s_p1.npy' % fold), M64)
    print('%s: P1 cached %d images, shape %s dtype %s, binary=%s'
          % (fold, len(images), M64.shape, M64.dtype,
             set(np.unique(M64)).issubset({0, 1})))
    return images, M64


def provenance_record():
    rec = {
        'step': '7B',
        'parent_cache': CACHE_DIR,
        'source_teacher': {
            'path': 'archive/train_seg_unet/best.pth',
            'sha256_full': SEG_SHA256_FULL,
        },
        'p1_transform': {
            'steps': ['M = 1[p>=0.5]',
                      'non-overlapping 4x4 blocks',
                      'q = mean(block)',
                      'M64 = 1[q>=0.5]'],
            'input_shape': [3, 256, 256],
            'output_shape': [3, 64, 64],
            'dtype': 'uint8',
            'binary': True,
        },
        'no_sweep': True,
        'no_test': True,
        'built_at': datetime.now(timezone.utc).isoformat(),
        'commit': git_head(),
    }
    for fold in ('train', 'val'):
        maps = np.load(os.path.join(CACHE_DIR, '%s_maps.npy' % fold))
        rec[fold] = {'n_images': maps.shape[0],
                     'file': '%s_p1.npy' % fold}
    os.makedirs(P1_CACHE_DIR, exist_ok=True)
    with open(os.path.join(P1_CACHE_DIR, 'p1_meta.json'), 'w') as f:
        json.dump(rec, f, indent=2)
    return rec


def main():
    rec = provenance_record()
    print('P1 provenance:', json.dumps(rec, indent=2))
    for fold in ('train', 'val'):
        images, M64 = build(fold)
        assert len(images) == M64.shape[0]
    print('P1 CACHE BUILD DONE')


if __name__ == '__main__':
    main()