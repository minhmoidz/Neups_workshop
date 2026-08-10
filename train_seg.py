import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from networks.UNetSeg import UNetSeg
from chexnet.seg_dataset import SegDataset


def dice_loss(pred, target, eps=1e-7):
    """Soft Dice loss, per-channel averaged (empty target channels count as 1.0)."""
    num = 2 * (pred * target).sum(dim=(2, 3)) + eps
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def dice_score(y_mask, p_mask, eps=1e-7):
    inter = (y_mask * p_mask).sum()
    return (2 * inter + eps) / (y_mask.sum() + p_mask.sum() + eps)


def iou_score(y_mask, p_mask, eps=1e-7):
    inter = (y_mask * p_mask).sum()
    union = (y_mask + p_mask).clamp(max=1).sum()
    return (inter + eps) / (union + eps)


@torch.no_grad()
def evaluate(model, img, mask, device):
    """Per-sample Dice/IoU for one batch. Returns (3,) means across the batch."""
    model.eval()
    with torch.no_grad():
        img = img.to(device)
        mask = mask.to(device)
        out = model(img) > 0.5
        B = img.size(0)
        dice = np.zeros(3)
        ious = np.zeros(3)
        for b in range(B):
            for c in range(3):
                dice[c] += dice_score(mask[b, c], out[b, c]).item()
                ious[c] += iou_score(mask[b, c], out[b, c]).item()
    model.train()
    return dice / B, ious / B


def main():
    parser = argparse.ArgumentParser('Train CheXmask segmenter U-Net')
    parser.add_argument('--image_path', default='/home/minhtt/datasets/nih/images')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--init_features', type=int, default=32)
    parser.add_argument('--save_path', default='./archive/train_seg_unet')
    parser.add_argument('--subsample', type=int, default=0, help='cap on training size for smoke runs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds = SegDataset('train', args.image_path, subsample=args.subsample)
    val_ds = SegDataset('val', args.image_path, subsample=1500)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = UNetSeg(in_channels=1, out_channels=3, init_features=args.init_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.save_path, exist_ok=True)
    best_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        for i, (img, mask, _) in enumerate(train_loader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = 0.5 * criterion(pred, mask) + 0.5 * compute_dice(pred, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            if i % 50 == 0:
                print(f'  epoch {epoch} iter {i} loss {loss.item():.4f}', flush=True)

        dice = np.zeros(3)
        ious = np.zeros(3)
        n_batches = 0
        for img, mask, _ in val_loader:
            d, i = evaluate(model, img, mask, device)
            dice += d
            ious += i
            n_batches += 1
        dice, ious = dice / n_batches, ious / n_batches
        mean_dice = dice.mean()
        print(f'epoch {epoch} | train_loss {run_loss / max(len(train_loader), 1):.4f} | '
              f'val dice LL {dice[0]:.4f} RL {dice[1]:.4f} Heart {dice[2]:.4f} | '
              f'iou LL {ious[0]:.4f} RL {ious[1]:.4f} Heart {ious[2]:.4f}', flush=True)

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'mean_dice': mean_dice,
                        'init_features': args.init_features,
                        'dice': dice.tolist(), 'iou': ious.tolist()},
                       f'{args.save_path}/best.pth')
            print(f'  saved best (mean dice {mean_dice:.4f})', flush=True)

    print(f'DONE best mean dice {best_dice:.4f}')


def compute_dice(pred, target, eps=1e-7):
    inter = (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


if __name__ == '__main__':
    main()