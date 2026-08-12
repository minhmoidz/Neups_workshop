"""STEP 4A band transforms — deterministic, complementary LOW-PASS / HIGH-PASS.

Design decision (PART A):
    H1 claims residual identity after the corrected baseline is carried by
    coarse/low-frequency thoracic structure, while the ~1 px smooth deformation
    mainly affects finer-scale content.

    Single justified cutoff: the generator smooths its flow field with a Gaussian
    of sigma=2 (kernel 9) at 256x256. A spatial Gaussian low-pass with
    SIGMA_LP = 8 px (4x the flow-smoothing sigma, i.e. ~5.1 cycles/image at 256)
    robustly separates the coarse anatomical band (lung fields, heart silhouette,
    ~30-80 px structures) from the fine-scale band that the ~1 px deformation
    perturbs. No cutoff sweep is performed.

    LOW-PASS  = depthwise Gaussian blur (reflect padding, exact).
    HIGH-PASS = x - LOW_PASS(x)  (exact complement: LP + HP == x bit-for-bit).

    Dynamic range:
        LP(x) in [0,1]  (convex combination of [0,1] inputs).
        HP(x) signed residual, in approximately [-1, 1] (zero-mean for flat regions).
    No clipping or renormalization is applied so the bands remain faithful and
    complementary; downstream networks see the genuine band content.
"""

import torch
import torch.nn.functional as F

SIGMA_LP = 8.0
SIGMA_FLOW = 2.0
KERNEL_SIZE = int(2 * __import__('math').ceil(3 * SIGMA_LP) + 1)  # 49


def gaussian_kernel_2d(sigma, channels=1, kernel_size=None):
    """Return a (channels, 1, K, K) depthwise Gaussian kernel normalized to sum 1."""
    if kernel_size is None:
        kernel_size = KERNEL_SIZE
    k = (kernel_size - 1) / 2.0
    y = torch.arange(-k, k + 1, dtype=torch.float32)
    g1 = torch.exp(-(y ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    g2 = torch.outer(g1, g1)
    g2 = g2 / g2.sum()
    return g2.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def low_pass(x, sigma=SIGMA_LP, kernel_size=None):
    """Apply exact Gaussian low-pass to (B, C, H, W) input using reflect padding.

    :param x: torch.Tensor (B, C, H, W) floats, e.g. deformed image in [0,1].
    :param sigma: float, Gaussian sigma in pixels.
    :param kernel_size: int, kernel extent (default derived from sigma).
    :return: torch.Tensor same shape, LOW-PASS band.
    """
    if kernel_size is None:
        kernel_size = KERNEL_SIZE
    pad = kernel_size // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    kernel = gaussian_kernel_2d(sigma, channels=x.shape[1], kernel_size=kernel_size).to(x.device)
    return F.conv2d(xp, kernel, groups=x.shape[1])


def high_pass(x, sigma=SIGMA_LP, kernel_size=None):
    """Exact complementary HIGH-PASS band: x - LOW_PASS(x)."""
    return x - low_pass(x, sigma=sigma, kernel_size=kernel_size)
