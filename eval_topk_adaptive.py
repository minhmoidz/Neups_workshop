"""Top-1 / Top-5 / MRR identification with the REPRESENTATIVE adaptive attacker (STEP 2B Part 10).

Canonical arm configuration (frozen protocol):
    N = 500 patients, fixed selection seed 42
    CLEAN gallery, ANONYMIZED probe
    gallery/probe images prefer different follow-up numbers where available.

The frozen gallery/probe list (topk_frozen_list.csv) is built ONCE per arm by
adaptive_reid.topk and reused by every arm. Evaluation uses the representative adaptive
attacker's branch embedding (forward output of a SiameseNetwork) for both gallery and
probe. Frozen ImageNet feature evaluation is available separately as an explicitly named
proxy mode (--proxy_resnet).

NOTE: headline Top-k results are NOT computed during STEP 2B (infrastructure only);
this script is the evaluation entry point used after representative selection.
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

from networks.SiameseNetwork import SiameseNetwork
from networks.UNet_PriCheXyNet import UNet

from adaptive_reid import topk


def load_rgb(path):
    img = Image.open(path).convert('L')
    img = img.resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1,256,256) [0,1]


def build_attacker(checkpoint_path):
    net = SiameseNetwork().cuda()
    state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    net.load_state_dict(state)
    net.eval()
    return net


def build_generator(checkpoint_path):
    state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    gen = UNet(1, state['conv.weight'].shape[0], 32).cuda()
    gen.load_state_dict(state)
    gen.eval()
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
    axis = torch.linspace(-1, 1, 256)
    gy, gx = torch.meshgrid(axis, axis, indexing='ij')
    grid_identity = torch.stack([gx, gy], 0).unsqueeze(0).cuda()
    return gen, gauss, grid_identity


def embed_attacker(attacker, images, generator, gauss, grid_identity, mu, transform_mode):
    if generator is not None:
        images = utils.deform(images, generator, grid_identity.expand(images.shape[0], -1, -1, -1),
                              gauss, mu, 0.0, transform_mode)
    images = images.expand(-1, 3, -1, -1)
    norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    images = norm(images)
    with torch.no_grad():
        feats = attacker.forward_once(images)  # 128-d branch embedding
    return F.normalize(feats, dim=1)


def embed_generator_curve(extractor, images, mean, std):
    images = images.expand(-1, 3, -1, -1)
    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    with torch.no_grad():
        feats = extractor((images - mean) / std)
    return F.normalize(feats, dim=1)


def compute_ranks(sims):
    ranks = np.argsort(-sims.cpu().numpy(), axis=1)
    n = ranks.shape[0]
    idx = np.arange(n)
    top1 = (ranks[:, 0] == idx).mean()
    top5 = (idx[:, None] == ranks[:, :5]).any(axis=1).mean()
    mrr = np.mean([1.0 / (np.where(r == i)[0][0] + 1) for i, r in enumerate(ranks)])
    return top1, top5, mrr


def main():
    parser = argparse.ArgumentParser('Top-k identification (representative adaptive attacker)')
    parser.add_argument('--frozen_list', required=True, help='path to topk_frozen_list.csv')
    parser.add_argument('--attacker_checkpoint', required=True,
                        help='representative adaptive attacker (SiameseNetwork) checkpoint')
    parser.add_argument('--generator_checkpoint', default=None,
                        help='anonymizer generator; omit for identity (no deform)')
    parser.add_argument('--transform_mode', default='legacy', choices=['legacy', 'corrected'])
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--image_path', default='/home/minhtt/datasets/nih/images/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--proxy_resnet', action='store_true',
                        help='EXPLICIT degraded mode: use frozen ImageNet ResNet-50 instead of '
                             'the adaptive attacker (proxy only, never the headline estimator).')
    parser.add_argument('--seed', type=int, default=topk.TOPK_SELECTION_SEED)
    args = parser.parse_args()

    utils.seed_all(args.seed)
    df = topk.load_frozen_topk_list(args.frozen_list)

    attacker = None
    generator = None
    gauss = None
    grid_identity = None
    if args.proxy_resnet:
        extractor = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        extractor.fc = torch.nn.Identity()
        extractor = extractor.cuda().eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    else:
        attacker = build_attacker(args.attacker_checkpoint)
    if args.generator_checkpoint:
        generator, gauss, grid_identity = build_generator(args.generator_checkpoint)

    def embed_fnames(fnames):
        vecs = []
        for i in range(0, len(fnames), args.batch_size):
            chunk = fnames[i:i + args.batch_size]
            images = torch.stack([load_rgb(os.path.join(args.image_path, f)) for f in chunk]).cuda()
            if args.proxy_resnet:
                vecs.append(embed_generator_curve(extractor, images, mean, std))
            else:
                vecs.append(embed_attacker(attacker, images, generator, gauss, grid_identity,
                                           args.mu, args.transform_mode))
        return torch.cat(vecs)

    gallery = df['gallery_image'].tolist()
    probe = df['probe_image'].tolist()

    gallery_vec = embed_fnames(gallery)
    probe_vec = embed_fnames(probe)

    sims = probe_vec @ gallery_vec.T
    top1, top5, mrr = compute_ranks(sims)

    print('n_patients: %d' % len(df))
    print('embedding: %s' % ('frozen-ImageNet-ResNet50 (PROXY)' if args.proxy_resnet
                             else 'representative adaptive attacker'))
    print('transform_mode: %s | mu: %s | gallery: CLEAN | probe: ANONYMIZED(%s)' %
          (args.transform_mode, args.mu, 'yes' if args.generator_checkpoint else 'no'))
    print('TOP1_ACC: %.4f' % top1)
    print('TOP5_ACC: %.4f' % top5)
    print('MRR: %.4f' % mrr)


if __name__ == '__main__':
    main()