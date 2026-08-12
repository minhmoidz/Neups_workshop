"""STEP 4A PART F - corrected-baseline displacement field spectrum.

Measure the ACTUAL corrected-baseline displacement fields, in pixel units:

    disp_px = gauss_sigma2( mu * flow ) * (H-1)/2      (grid [-1,1] <-> pixels, align_corners=True)

Reported per-image: mean |u|, max |u|, std |u| (in px).
Reported spectrally: 2D radial power spectrum of the applied displacement field,
fraction of displacement energy below vs above a fixed cutoff, and the full PSD profile.

Cutoff: the -3 dB (amplitude half-power) point of the PART A Gaussian low-pass
(sigma_lp = 8 px), i.e.  exp(-2 pi^2 sigma^2 f^2) = 1/2  =>
    f_c = sqrt(ln 2) / (2 pi * sigma_lp)  ~ 0.01655 cycles/px (~4.24 cycles / 256-px image).

Note: this is a descriptive measurement of the frozen corrected baseline only; it
produces no privacy number and uses VALIDATION images only.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from networks.UNet_PriCheXyNet import UNet
from utils import utils
from utils.GaussianSmoothing import GaussianSmoothing
from research_agent import band

PAIR_VAL = 'image_pairs/image_pairs_validation_2000.txt'
IMG_PATH = '/home/minhtt/datasets/nih/images/'
N_PAIRS = 64
OUT = 'research_agent/05A_artifacts/flow_spectrum.json'


def load_images(pairs):
    resize = transforms.Resize((256, 256))
    to_tensor = transforms.ToTensor()
    imgs = []
    seen = set()
    for row in pairs:
        for fn in (row[0], row[1]):
            if fn not in seen:
                seen.add(fn)
                with open(IMG_PATH + fn, 'rb') as f:
                    im = Image.open(f).convert('L')
                imgs.append(to_tensor(resize(im)))
    return torch.stack(imgs)


def radfreqs(n):
    freqs = torch.fft.fftfreq(n, d=1.0)
    fx = freqs.unsqueeze(1).repeat(1, n)
    fy = freqs.unsqueeze(0).repeat(n, 1)
    return torch.sqrt(fx ** 2 + fy ** 2)


@torch.no_grad()
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen = UNet(in_channels=1, out_channels=2, init_features=32)
    gen.load_state_dict(torch.load('networks/corrected_baseline/generator_lowest_total_loss_corrected.pth',
                                   map_location='cpu', weights_only=False))
    gen = gen.to(device).eval()

    d = torch.linspace(-1, 1, 256)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)

    pairs = np.loadtxt(PAIR_VAL, dtype=str)[:N_PAIRS]
    imgs = load_images(pairs).to(device)
    print('images', tuple(imgs.shape))

    flow = gen(imgs)  # (N, 2, 256, 256), in [-1,1]
    budget, _ = utils.compute_budget_map(flow, mu=0.01)
    scaled = budget * flow
    disp_norm = gauss(scaled)  # applied displacement in normalized grid coords
    disp_px = disp_norm * 127.5  # (N,2,256,256)

    stats = []
    per_image = []
    for i in range(disp_px.shape[0]):
        u = disp_px[i]
        mag = u.pow(2).sum(0).sqrt()
        per_image.append({
            'mean_abs_px': float(mag.mean()),
            'max_abs_px': float(mag.max()),
            'std_abs_px': float(mag.std()),
            'rms_px': float(mag.pow(2).mean().sqrt()),
        })
    arr = np.array([[p['mean_abs_px'], p['max_abs_px'], p['std_abs_px'], p['rms_px']] for p in per_image])
    stats = {
        'n_images': len(per_image),
        'mean_abs_px': {'mean': float(arr[:, 0].mean()), 'std': float(arr[:, 0].std()), 'p95': float(np.percentile(arr[:, 0], 95))},
        'max_abs_px': {'mean': float(arr[:, 1].mean()), 'max': float(arr[:, 1].max())},
        'std_abs_px': {'mean': float(arr[:, 2].mean())},
        'rms_px': {'mean': float(arr[:, 3].mean())},
    }

    n = 256
    fy = torch.fft.fftfreq(n, d=1.0).view(-1, 1)        # cycles/px, rows (vertical)
    fx = torch.fft.rfftfreq(n, d=1.0).view(1, -1)       # cycles/px, cols (horizontal)
    rf = torch.sqrt(fx ** 2 + fy ** 2).cpu()            # (256, 129) radial freq grid

    psd_sum = None
    for c in range(2):
        x = disp_px[:, c].cpu()
        spec = torch.fft.rfft2(x, norm='ortho')
        power = (spec.abs() ** 2).mean(dim=0)  # average over images
        if psd_sum is None:
            psd_sum = power.clone()
        else:
            psd_sum = psd_sum + power
    psd = psd_sum / 2.0  # average over channels

    f_c = np.sqrt(np.log(2.0)) / (2 * np.pi * band.SIGMA_LP)
    below = psd[rf <= f_c].sum().item()
    above = psd[rf > f_c].sum().item()
    total = below + above
    frac_below = below / total
    frac_above = above / total

    order = torch.argsort(rf.flatten())
    r_flat = rf.flatten()[order].numpy().tolist()
    psd_flat = psd.flatten()[order].numpy().tolist()

    spectrum_profile = {
        'cycles_per_pixel': r_flat,
        'power': psd_flat,
    }

    result = {
        '_label': 'DESCRIPTIVE PART F MEASUREMENT (no privacy number; VALIDATION only)',
        'generator': 'networks/corrected_baseline/generator_lowest_total_loss_corrected.pth',
        'transform_mode': 'corrected',
        'mu': 0.01,
        'n_images': len(per_image),
        'displacement_stats_px': stats,
        'cutoff_cycles_per_px': f_c,
        'cutoff_cycles_per_image_256': f_c * 256,
        'cutoff_definition': '-3 dB (amplitude half-power) of PART A Gaussian low-pass (sigma_lp=8 px)',
        'energy_fraction_below_cutoff': frac_below,
        'energy_fraction_above_cutoff': frac_above,
        'energy_units': 'normalized DFT power, averaged over images and 2 flow channels',
        'spectrum_profile': spectrum_profile,
        'per_image': per_image,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print('mean_abs_px', stats['mean_abs_px'])
    print('max_abs_px', stats['max_abs_px'])
    print('rms_px', stats['rms_px'])
    print('f_c (cycles/px) = %.5f (= %.2f cycles/image @256)' % (f_c, f_c * 256))
    print('energy below cutoff = %.4f   above = %.4f' % (frac_below, frac_above))
    print('wrote', OUT)


if __name__ == '__main__':
    main()