import argparse
import csv
import os

import numpy as np
import torch

from networks.UNetSeg import UNetSeg
from chexnet.seg_dataset import SegDataset
from eval_seg import hd95, build_deform_fn


def main():
    parser = argparse.ArgumentParser('STEP 3C segmentation evaluator with per-case persistence')
    parser.add_argument('--image_path', default='/home/minhtt/datasets/nih/images')
    parser.add_argument('--checkpoint', default='./archive/train_seg_unet/best.pth')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--generator', default='./networks/corrected_baseline/generator_lowest_total_loss_corrected.pth')
    parser.add_argument('--stochastic_lambda', type=float, default=0.0)
    parser.add_argument('--transform_mode', default='corrected', choices=['legacy', 'corrected'])
    parser.add_argument('--out_csv', default='research_agent/03H_artifacts/seg_per_case.csv')
    args = parser.parse_args()

    from utils import utils
    args.transform_mode = utils.resolve_transform_mode(args.transform_mode)
    print('[transform_mode] resolved mode for this run: %s' % args.transform_mode)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    chk = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    model = UNetSeg(in_channels=1, out_channels=3,
                    init_features=32 if 'init_features' not in chk else chk['init_features']).to(device)
    model.load_state_dict(chk['model'])
    model.eval()

    ds = SegDataset('test', args.image_path, subsample=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    deform_fn = None
    if args.generator is not None:
        deform_fn = build_deform_fn(args.generator, args.mu, args.stochastic_lambda, device, args.transform_mode)

    names = ['Left Lung', 'Right Lung', 'Heart']
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []
    n_cases = 0
    n_empty_pred = [0, 0, 0]
    n_empty_mask = [0, 0, 0]
    sums = {'dice': np.zeros(3), 'iou': np.zeros(3), 'hd95': np.zeros(3)}

    with torch.no_grad():
        for idx_b, (img, mask, names_b) in enumerate(loader):
            img = img.to(device)
            mask_np = mask.cpu().numpy()
            if deform_fn is not None:
                img = deform_fn(img)
            pred_np = (model(img) > 0.5).detach().cpu().numpy()
            for b in range(img.size(0)):
                for c in range(3):
                    p = pred_np[b, c]
                    t = mask_np[b, c]
                    inter = (t * p).sum()
                    union = ((t + p) > 0).sum()
                    dice = (2 * inter + 1e-7) / (t.sum() + p.sum() + 1e-7)
                    iou = (inter + 1e-7) / (union + 1e-7)
                    hd = hd95(p, t)
                    sums['dice'][c] += dice
                    sums['iou'][c] += iou
                    sums['hd95'][c] += hd
                    if p.sum() == 0:
                        n_empty_pred[c] += 1
                    if t.sum() == 0:
                        n_empty_mask[c] += 1
                    rows.append([names_b[b], names[c], dice, iou, hd])
                n_cases += 1

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_index', 'structure', 'dice', 'iou', 'hd95'])
        for r in rows:
            w.writerow(r)

    print('n_cases:', n_cases)
    print('empty_pred per structure:', dict(zip(names, n_empty_pred)))
    print('empty_mask per structure:', dict(zip(names, n_empty_mask)))
    print(f'{"structure":<12s} {"Dice":>8s} {"IoU":>8s} {"HD95":>10s}')
    for c, n in enumerate(names):
        print(f'{n:12s} {sums["dice"][c]/n_cases:8.4f} {sums["iou"][c]/n_cases:8.4f} {sums["hd95"][c]/n_cases:10.3f}')
    print(f'{"MEAN":12s} {sums["dice"].mean()/n_cases:8.4f} {sums["iou"].mean()/n_cases:8.4f} {sums["hd95"].mean()/n_cases:10.3f}')


if __name__ == '__main__':
    main()
