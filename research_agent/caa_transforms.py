"""STEP 4B CAA mechanism diagnostic — canonical deterministic transforms.

Only TWO diagnostic transforms are authorized (plus the identity/reference arm):

H3 border/FOV (research_agent BORDER transform):
    Replace the OUTERMOST BORDER_WIDTH-pixel image border with a constant value
    equal to that image's MEDIAN intensity. Frozen at BORDER_WIDTH = 4 px, the
    boundary region implicated by the historical legacy-operator border defect.
    Deterministic. No crop, no resize-after-mask, no central masking, no tuning.

H2 acquisition/intensity (research_agent INTENSITY transform):
    Robust per-image affine intensity normalization: for every image independently
        p1  = 1st intensity percentile
        p99 = 99th intensity percentile
        X_norm = clip((X - p1) / max(p99 - p1, eps), 0, 1)     eps = 1e-6
    then mapped back into the exact numeric input range the consuming pipeline
    expects (out_lo, out_hi). Removes image-specific global offset/gain/dynamic
    range. Deterministic: preserves geometry + pixel ordering, no CLAHE / histogram
    equalization / noise / blur / geometry / border-combination / percentile sweep.

Both are PURE diagnostic compositions applied AFTER the frozen corrected-baseline
deform ("the mechanism under test" stage), on TRAIN + VALIDATION only.
"""

import torch

BORDER_WIDTH = 4
H2_EPS = 1e-6


def border_normalize(x, border_width=BORDER_WIDTH):
    """Replace the outermost border_width-pixel frame with the per-image median.

    :param x: torch.Tensor (B, C, H, W) float, deformed image intensities.
    :param border_width: int, frozen at 4.
    :return: torch.Tensor same shape; border replaced by per-image median intensity.
    """
    if border_width != BORDER_WIDTH:
        raise ValueError('border width is frozen at %d for the H3 diagnostic' % BORDER_WIDTH)
    out = x.clone()
    b = border_width
    flat = x.flatten(2)                       # (B, C, H*W)
    med = flat.median(dim=2, keepdim=True).values  # (B, C, 1)
    # horizontal (top/bottom) border -> rows [0:b) and [-b:]
    out[:, :, :b, :] = med.unsqueeze(-1)
    out[:, :, -b:, :] = med.unsqueeze(-1)
    # vertical (left/right) border -> cols [0:b) and [-b:]
    out[:, :, :, :b] = med.unsqueeze(-1)
    out[:, :, :, -b:] = med.unsqueeze(-1)
    return out


def intensity_normalize(x, out_lo=0.0, out_hi=1.0, eps=H2_EPS):
    """Robust per-image p1/p99 affine intensity normalization.

    For each image independently maps p1..p99 onto (out_lo, out_hi) after clipping
    into [0,1]. out_lo/out_hi reproduce the numeric range the consuming pipeline
    expects (e.g. (0,1) for attacker/classifier, (-1,1) for the segmenter).

    :param x: torch.Tensor (B, C, H, W) float, deformed image intensities.
    :param out_lo: float, lower bound of the target numeric range.
    :param out_hi: float, upper bound of the target numeric range.
    :param eps: float, frozen at 1e-6 denominator guard.
    :return: torch.Tensor same shape, pixel-ordering preserved.
    """
    flat = x.flatten(2)                       # (B, C, H*W)
    p1 = torch.quantile(flat, 0.01, dim=2, keepdim=True).unsqueeze(-2)   # (B, C, 1, 1)
    p99 = torch.quantile(flat, 0.99, dim=2, keepdim=True).unsqueeze(-2)  # (B, C, 1, 1)
    denom = (p99 - p1).clamp_min(eps)
    xn = ((x - p1) / denom).clamp(0.0, 1.0)
    return out_lo + xn * (out_hi - out_lo)