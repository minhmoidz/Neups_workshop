"""Cheap surrogate for the 10-seed re-identification protocol.

The full protocol re-trains a Siamese network from scratch ten times (~10 h per generator), which makes
any kind of fast iteration impossible. This script instead measures how much identity signal survives in
the feature space of a *frozen, independent* ImageNet ResNet-50 and reports the resulting verification AUC.

The extractor matters: both the pre-trained SNN and the pre-trained classifier appear in PriCheXy-Net's
training objective, so a generator can learn to fool them specifically. An ImageNet backbone takes part in
no loss, so it cannot be gamed and gives an unbiased read of the remaining biometric signal.

Only useful once its correlation with the real 10-seed AUC has been checked -- see calibrate_proxy.py.
"""

import argparse

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from utils import utils
from utils.GaussianSmoothing import GaussianSmoothing


def build_extractor():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()

    return model.cuda().eval()


def embed(extractor, images, mean, std):
    """Map single-channel [0,1] images to L2-normalised ImageNet features."""

    images = images.expand(-1, 3, -1, -1)
    images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    return F.normalize(extractor((images - mean) / std), dim=1)


def main():
    parser = argparse.ArgumentParser('Proxy re-identification score')
    parser.add_argument('--checkpoint', default=None, help='Generator checkpoint; omit for real images.')
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--stochastic_lambda', type=float, default=0.0)
    parser.add_argument('--transform_mode', default='legacy', choices=['legacy', 'corrected'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_path', default='/data/images/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    utils.seed_all(args.seed)

    # Explicit operator provenance in the console header (BLOCKER 9).
    args.transform_mode = utils.resolve_transform_mode(args.transform_mode)
    print('[transform_mode] resolved mode for this run: %s' % args.transform_mode)

    extractor = build_extractor()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    if args.checkpoint is not None:
        generator = utils.load_flow_generator(args.checkpoint).eval()
        gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()
        axis = torch.linspace(-1, 1, args.image_size)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing='ij')
        grid_identity = torch.stack([grid_x, grid_y], 0).unsqueeze(0).cuda()
    else:
        generator = None

    loader = utils.get_data_loader(phase='testing', experimental_step='retrainSNN', image_size=args.image_size,
                                  n_channels=1, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                  b=None, m=None, eps=None, image_path=args.image_path)

    scores, targets = [], []
    with torch.no_grad():
        for inputs1, inputs2, labels in loader:
            inputs1, inputs2 = inputs1.cuda(), inputs2.cuda()

            if generator is not None:
                # Only the released image is deformed; it is matched against a real gallery image
                inputs1 = utils.deform(inputs1, generator, grid_identity.expand(inputs1.shape[0], -1, -1, -1),
                                       gauss_filter, args.mu, args.stochastic_lambda, args.transform_mode)

            similarity = (embed(extractor, inputs1, mean, std) * embed(extractor, inputs2, mean, std)).sum(dim=1)
            scores.append(similarity.cpu().numpy())
            targets.append(labels.numpy())

    scores = np.concatenate(scores)
    targets = np.concatenate(targets)

    print('n_pairs: %d' % len(targets))
    print('PROXY_AUC: %.6f' % roc_auc_score(targets, scores))


if __name__ == '__main__':
    main()
