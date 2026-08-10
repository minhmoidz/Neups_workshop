"""Top-1 / Top-5 identification rate (T6) -- the real-world N:1 linkage measure missing from the paper.

Attacker scenario:
  * gallery  = N real scans (one per patient) that identify the person (e.g. from a hospital PACS).
  * probe    = an anonymized release image of the *same* patient (a second scan, deformed).
  * attacker  embeds everything with a frozen, independent ImageNet ResNet-50 and looks up the
    gallery item most similar to each probe. Success = the gallery patient matches the probe patient.

This is measured on top of the same deformation pipeline used everywhere else (utils.deform), so
values are directly comparable with the verification-AUC table.

Usage:
    python eval_top1.py --checkpoint ./archive/.../generator_lowest_total_loss.pth --mu 0.02 \
        --n_patients 500 --image_path /home/minhtt/datasets/nih/images/
Omit --checkpoint to measure the original (unperturbed) upper bound.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image

from utils import utils
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_PriCheXyNet import UNet


def build_extractor():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    return model.cuda().eval()


def embed(extractor, images, mean, std):
    images = images.expand(-1, 3, -1, -1)
    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    return F.normalize(extractor((images - mean) / std), dim=1)


def load_rgb(path):
    img = Image.open(path).convert('L')
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1,256,256) [0,1]


def main():
    parser = argparse.ArgumentParser('Top-1/5 identification (T6)')
    parser.add_argument('--checkpoint', default=None, help='Generator checkpoint; omit for real images.')
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--n_patients', type=int, default=500)
    parser.add_argument('--image_path', default='/home/minhtt/datasets/nih/images/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    utils.seed_all(args.seed)
    extractor = build_extractor()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    gauss = None
    grid_identity = None
    generator = None
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
        generator = UNet(1, state['conv.weight'].shape[0], 32).cuda()
        generator.load_state_dict(state)
        generator.eval()
        gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
        axis = torch.linspace(-1, 1, 256)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing='ij')
        grid_identity = torch.stack([grid_x, grid_y], 0).unsqueeze(0).cuda()

    # --- pick n_patients with >=2 test images; gallery image != probe image ---
    df = pd.read_csv('chexnet/nih_labels.csv')
    df = df[df['fold'] == 'test']
    sizes = df.groupby('Patient ID').size()
    eligible = sizes[sizes >= 2]
    rng = np.random.RandomState(args.seed)
    patients = list(eligible.sample(min(args.n_patients, len(eligible)), random_state=args.seed).index)

    gallery_fnames, probe_fnames, probe_pid = [], [], []
    for pid in patients:
        imgs = df[df['Patient ID'] == pid]['Image Index'].tolist()
        rng.shuffle(imgs)
        gallery_fnames.append(imgs[0])
        probe_fnames.append(imgs[1])
        probe_pid.append(pid)

    def embed_batch(fnames):
        vecs = []
        with torch.no_grad():
            for i in range(0, len(fnames), args.batch_size):
                chunk = fnames[i:i + args.batch_size]
                images = torch.stack([load_rgb(os.path.join(args.image_path, f)) for f in chunk])
                images = images.cuda()
                if generator is not None:
                    images = utils.deform(images, generator, grid_identity.expand(images.shape[0], -1, -1, -1),
                                          gauss, args.mu, 0.0)
                vecs.append(embed(extractor, images, mean, std))
        return torch.cat(vecs)

    gallery_vec = embed_batch(gallery_fnames)   # (N, D)
    probe_vec = embed_batch(probe_fnames)       # (N, D)

    sims = probe_vec @ gallery_vec.T            # (N, N), row r sorted by identity
    ranks = np.argsort(-sims.cpu().numpy(), axis=1)
    top1 = (ranks[:, 0] == np.arange(len(patients))).mean()
    top5 = (np.arange(len(patients))[:, None] == ranks[:, :5]).any(axis=1).mean()

    # mean reciprocal rank (MRR) for completeness
    mrr = np.mean([1.0 / (np.where(r == i)[0][0] + 1) for i, r in enumerate(ranks)])

    print('n_patients: %d' % len(patients))
    print('TOP1_ACC: %.4f' % top1)
    print('TOP5_ACC: %.4f' % top5)
    print('MRR: %.4f' % mrr)
    print('generator: %s | mu: %s' % (args.checkpoint, args.mu))


if __name__ == '__main__':
    main()