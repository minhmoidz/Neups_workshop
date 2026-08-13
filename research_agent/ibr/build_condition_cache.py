"""STEP 7A — build condition cache (anatomy maps for TRAIN/VAL pair-file images).

Caches the 3 soft probability maps (Left Lung, Right Lung, Heart) from the frozen
segmentation teacher for every unique image referenced by the frozen TRAIN and
VALIDATION pair files.

Provenance recorded: source image identifier, fold, teacher SHA, map shape, dtype,
build commit, image base path. TEST images are never touched.
"""

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from research_agent.ibr.frozen_models import load_frozen_segmentation_teacher

IMAGE_PATH = '/home/minhtt/datasets/nih/images/'
PAIR_FILES = {
    'train': 'image_pairs/image_pairs_training_10000.txt',
    'val': 'image_pairs/image_pairs_validation_2000.txt',
}
CACHE_DIR = 'research_agent/ibr_s1_condition_cache/'
SEG_SHA256_FULL = '2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0'


def git_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()


def load_x_image(path):
    from PIL import Image
    from torchvision import transforms
    t = transforms.Compose([transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda im: im * 2 - 1)])
    with Image.open(path).convert('L') as im:
        return t(im)


def unique_images(fold):
    rows = np.loadtxt(PAIR_FILES[fold], dtype=str, delimiter='\t')
    imgs = sorted(set(rows[:, 0]) | set(rows[:, 1]))
    for i in imgs:
        assert '/' not in i and not i.startswith('.'), i
    return imgs


class ImageMapDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return load_x_image(IMAGE_PATH + self.images[i])


def build(fold):
    images = unique_images(fold)
    teacher, meta = load_frozen_segmentation_teacher('cuda')
    assert meta['sha256'].startswith('2dfdcf9b'), meta['sha256']
    teacher = teacher.to('cuda')
    teacher.eval()

    dl = DataLoader(ImageMapDataset(images), batch_size=16, shuffle=False,
                    num_workers=8, pin_memory=True)
    maps = np.zeros((len(images), 3, 256, 256), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            xb = batch.to('cuda')
            m = teacher(xb).cpu().numpy()          # (B,3,256,256) sigmoid in [0,1]
            s = bi * 16
            maps[s:s + len(xb)] = m.astype(np.float16)
            if (bi + 1) % 200 == 0:
                print('  %s batch %d/%d (%.0fs)' % (fold, bi + 1, len(dl), time.time() - t0))
    assert not np.isnan(maps).any() and not np.isinf(maps).any()

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, '%s_maps.npy' % fold), maps)
    with open(os.path.join(CACHE_DIR, '%s_images.json' % fold), 'w') as f:
        json.dump(images, f)
    print('%s: cached %d images, shape %s dtype %s (%.0fs)'
          % (fold, len(images), maps.shape, maps.dtype, time.time() - t0))
    return images, maps


def provenance_record():
    rec = {
        'step': '7A',
        'teacher': {
            'path': 'archive/train_seg_unet/best.pth',
            'sha256_full': SEG_SHA256_FULL,
            'prefix_assert': '2dfdcf9b',
        },
        'image_base_path': IMAGE_PATH,
        'pair_files': PAIR_FILES,
        'map_shape': [3, 256, 256],
        'dtype': 'float16',
        'channel_order': ['Left Lung', 'Right Lung', 'Heart'],
        'built_at': datetime.now(timezone.utc).isoformat(),
        'commit': git_head(),
        'test_accessed': False,
    }
    for fold in ('train', 'val'):
        images = unique_images(fold)
        rec[fold] = {'n_images': len(images), 'files': '%s_maps.npy / %s_images.json' % (fold, fold)}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, 'cache_meta.json'), 'w') as f:
        json.dump(rec, f, indent=2)
    return rec


def main():
    assert os.path.exists(IMAGE_PATH), IMAGE_PATH
    rec = provenance_record()
    print('provenance:', json.dumps(rec, indent=2))
    for fold in ('train', 'val'):
        images, maps = build(fold)
        assert len(images) == maps.shape[0]
    print('CACHE BUILD DONE')


if __name__ == '__main__':
    main()