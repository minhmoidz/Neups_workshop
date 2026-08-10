import argparse
import os
import numpy as np
import torch
from torchvision import transforms
from scipy.ndimage import distance_transform_edt

from networks.UNetSeg import UNetSeg
from networks.UNet_PriCheXyNet import UNet
from chexnet.seg_dataset import SegDataset
from utils import utils
from utils.GaussianSmoothing import GaussianSmoothing


def hd95(pred, target):
    """Two-sided 95th percentile Hausdorff distance in pixel units."""
    if pred.sum() == 0 or target.sum() == 0:
        return float('nan')
    d_pred = distance_transform_edt(1 - pred)
    d_target = distance_transform_edt(1 - target)
    fwd = d_target[pred > 0]
    bwd = d_pred[target > 0]
    return max(np.percentile(fwd, 95), np.percentile(bwd, 95))


@torch.no_grad()
def evaluate(model, loader, batch_size, device, deformation_fn=None):
    model.eval()
    dice = np.zeros(3)
    ious = np.zeros(3)
    hds = np.zeros(3)
    hits = 0
    for img, mask, _ in loader:
        img, mask = img.to(device), mask.to(device)
        if deformation_fn is not None:
            img = deformation_fn(img)
        pred = (model(img) > 0.5).detach().cpu().numpy()
        mask = mask.cpu().numpy()
        for b in range(img.size(0)):
            for c in range(3):
                inter = (mask[b, c] * pred[b, c]).sum()
                union = ((mask[b, c] + pred[b, c]) > 0).sum()
                dice[c] += (2 * inter + 1e-7) / (mask[b, c].sum() + pred[b, c].sum() + 1e-7)
                ious[c] += (inter + 1e-7) / (union + 1e-7)
                hds[c] += hd95(pred[b, c], mask[b, c])
        hits += img.size(0)
    model.train()
    return dice / hits, ious / hits, hds / hits


def build_deform_fn(generator_path, mu, stochastic_lambda=0.0, device='cuda', transform_mode='legacy'):
    chk = torch.load(generator_path, weights_only=False, map_location='cpu')
    gen = UNet(1, chk['conv.weight'].shape[0], 32).to(device)
    gen.load_state_dict(chk)
    gen.eval()

    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)

    def fn(images):
        return utils.deform(images, gen, grid_identity, gauss, mu, stochastic_lambda, transform_mode)

    return fn


def main():
    parser = argparse.ArgumentParser('Evaluate U-Net segmenter on original vs anonymized test images')
    parser.add_argument('--image_path', default='/home/minhtt/datasets/nih/images')
    parser.add_argument('--checkpoint', default='./archive/train_seg_unet/best.pth')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--subsample', type=int, default=0, help='cap on test samples (smoke runs)')
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--generator', default=None)
    parser.add_argument('--stochastic_lambda', type=float, default=0.0)
    parser.add_argument('--transform_mode', default='legacy', choices=['legacy', 'corrected'])
    args = parser.parse_args()

    args.transform_mode = utils.resolve_transform_mode(args.transform_mode)
    print('[transform_mode] resolved mode for this run: %s' % args.transform_mode)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chk = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    model = UNetSeg(in_channels=1, out_channels=3, init_features=32 if 'init_features' not in chk else chk['init_features']).to(device)
    model.load_state_dict(chk['model'])
    model.eval()

    ds = SegDataset('test', args.image_path, subsample=args.subsample)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    deform_fn = None
    if args.generator is not None:
        deform_fn = build_deform_fn(args.generator, args.mu, args.stochastic_lambda, device, args.transform_mode)

    dice, ious, hds = evaluate(model, loader, args.batch_size, device, deformation_fn=deform_fn)

    names = ['Left Lung', 'Right Lung', 'Heart']
    print(f'{"structure":<12s} {"Dice":>6s} {"IoU":>6s} {"HD95":>6s}')
    for c, n in enumerate(names):
        print(f'{n:12s} {dice[c]:6.4f} {ious[c]:6.4f} {hds[c]:8.3f}')
    print(f'MEAN      {dice.mean():6.4f} {ious.mean():6.4f} {hds.mean():8.3f}')


if __name__ == '__main__':
    main()