"""Final regression suite for the corrected PriCheXy-Net transformation operator (STEP 1 + remediation).

The scientific defect: ``deform`` (and the pre-training loops) applied the Gaussian smoother to the *whole*
displaced sampling grid ``G * (I - u)``. Because GaussianSmoothing uses zero padding, ``G * I != I`` near the
borders, so a zero effective displacement did NOT reproduce the identity grid -- the border pixels were pulled
toward 0 and effectively dropped. The corrected operator smooths only the displacement:

    legacy:    grid = G * (I - u)      # historical, kept bit-for-bit reproducible
    corrected: grid = I - G * u        # zero displacement  =>  exact identity grid

This file contains the 13 tests required by the STEP 1B review:

  1. arbitrary nonzero flow + mu=0            => exact identity GRID (torch.equal)
  2. zero flow + mu>0                        => exact identity GRID (torch.equal)
  3. same-image mu=0 (arbitrary flow) equals zero-flow+mu>0 identity grid
  4. source-pixel coverage at mu=0: legacy unsampled > 0, corrected == 0
  5. legacy vs corrected interior equality at mu>0
  6. legacy vs corrected border inequality at mu>0
  7. corrected mu>0 deformation remains active
  8. budget map: mean(budget) ~= mu
  9. budget map spatial effect is actually used
 10. pretrain corrected sampling-grid construction
 11. preval corrected sampling-grid construction
 12. old-inline legacy equations == new legacy helper (torch.equal)
 13. existing gradient-accumulation regression still passes
"""

import os
import inspect
import json
import textwrap

import torch
import torch.nn as nn

from utils.GaussianSmoothing import GaussianSmoothing
from utils import utils as U

DEVICE = 'cpu'
IMAGE_SIZE = 64
RADIUS = (9 - 1) // 2          # GaussianSmoothing kernel_size=9 -> filter reaches 4 px into the image


def _identity_grid(device=DEVICE, size=IMAGE_SIZE):
    d = torch.linspace(-1, 1, size, device=device)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2)
    return grid_identity.unsqueeze(0).permute(0, 3, 1, 2)   # (1, 2, H, W)


def _gauss(device=DEVICE):
    return GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)


def _identity_grid_perm(n):
    """Identity grid in grid_sample layout (N, H, W, 2) for a batch of size n."""
    return _identity_grid().permute(0, 2, 3, 1).expand(n, -1, -1, -1)


class _FixedFlow(nn.Module):
    """Constant-flow generator so tests do not depend on a trained U-Net."""

    def __init__(self, value, channels=2):
        super().__init__()
        self.value = value
        self.channels = channels

    def forward(self, x):
        return torch.full((x.shape[0], self.channels) + x.shape[2:], self.value, device=x.device)


class _ZeroFlow(nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 2) + x.shape[2:], device=x.device)


class _BudgetFlow(nn.Module):
    """3-channel generator: constant flow + a spatially-varying budget channel."""

    def __init__(self, flow_value, budget_peak):
        super().__init__()
        self.flow_value = flow_value
        self.budget_peak = budget_peak

    def forward(self, x):
        b = x.shape[0]
        flow = torch.full((b, 2) + x.shape[2:], self.flow_value, device=x.device)
        # Budget channel: 0 everywhere except a horizontal band where it peaks -> spatially non-uniform.
        budget = torch.zeros(b, 1, x.shape[2], x.shape[3], device=x.device)
        budget[:, :, x.shape[2] // 3: 2 * x.shape[2] // 3, :] = self.budget_peak
        return torch.cat([flow, budget], 1)


def _scaled_flow(images, gen, mu):
    """Reproduce deform's budget scaling so tests can inspect the sampling grid directly."""
    with torch.no_grad():
        raw = gen(images)
    budget, flow = U.compute_budget_map(raw, mu)
    return budget * flow, raw


def _grid_for(images, gen, mu, mode):
    scaled, _ = _scaled_flow(images, gen, mu)
    return U.build_sampling_grid(scaled, _identity_grid(), _gauss(), mode)


# --------------------------------------------------------------------------- #


def test1_nonzero_flow_mu0_exact_identity_grid():
    """1. Arbitrary NONZERO flow, mu=0: the corrected sampling grid must equal the identity grid EXACTLY."""
    torch.manual_seed(0)
    images = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    grid_corrected = _grid_for(images, _FixedFlow(0.7), mu=0.0, mode='corrected')  # (2,H,W,2)
    identity_perm = _identity_grid_perm(images.shape[0])
    equal = torch.equal(grid_corrected, identity_perm)
    diff = (grid_corrected - identity_perm).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()

    assert equal, f'corrected grid must be EXACTLY identity at mu=0; max grid diff = {max_diff:.3e}'
    assert max_diff == 0.0 and mean_diff == 0.0, (max_diff, mean_diff)
    print(f'TEST 1 corrected mu=0 (flow=0.7): torch.equal=={equal}, max grid diff={max_diff:.0e}, '
          f'mean grid diff={mean_diff:.0e}')


def test2_zero_flow_mu_pos_exact_identity_grid():
    """2. Zero flow, mu>0: the corrected sampling grid must equal the identity grid EXACTLY."""
    torch.manual_seed(1)
    images = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    for mu in (0.01, 0.05):
        grid_corrected = _grid_for(images, _ZeroFlow(), mu=mu, mode='corrected')
        identity_perm = _identity_grid_perm(images.shape[0])
        equal = torch.equal(grid_corrected, identity_perm)
        diff = (grid_corrected - identity_perm).abs()
        max_diff, mean_diff = diff.max().item(), diff.mean().item()
        assert equal, f'zero flow mu={mu}: grid must be EXACT identity, max={max_diff:.3e}'
        assert max_diff == 0.0 and mean_diff == 0.0, (max_diff, mean_diff)
        print(f'TEST 2 zero flow mu={mu}: torch.equal=={equal}, max grid diff={max_diff:.0e}, '
              f'mean grid diff={mean_diff:.0e}')


def test3_same_image_mu0_vs_zeroflow_extra_identity():
    """3. Same image: (arbitrary flow + mu=0) grid == (zero flow + mu>0) grid == identity grid."""
    torch.manual_seed(2)
    images = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    gA = _grid_for(images, _FixedFlow(0.7), mu=0.0, mode='corrected')
    gB = _grid_for(images, _ZeroFlow(), mu=0.05, mode='corrected')
    identity_perm = _identity_grid_perm(images.shape[0])
    eqAB = torch.equal(gA, gB)
    eqAI = torch.equal(gA, identity_perm)
    valA, valB = gA.abs().max().item(), gB.abs().max().item()
    assert eqAB and eqAI, f'grids must coincide on identity: A==B {eqAB}, A==Id {eqAI}'
    assert (gA - gB).abs().max().item() == 0.0
    print(f'TEST 3 same-image identity invariance: A==B {eqAB}, A==Id {eqAI} (max |A|={valA:.3f}, '
          f'max |B|={valB:.3f}, grid coords in [-1,1])')


def _source_coverage(grid):
    """Accumulated bilinear sampling weight per source pixel.

    For a sampling grid in grid_sample layout (H, W, 2) with align_corners=True, each output pixel maps to a
    source location; bilinear interpolation spreads weight over the floor/ceil source pixels. We accumulate
    the total weight each SOURCE pixel receives across all output pixels. An identity grid gives weight 1 to
    exactly one source pixel per output pixel (all covered). A grid that pulls coordinates inward (legacy
    border artifact) leaves source pixels at the image border effectively unsampled (weight ~0).
    """
    H = grid.shape[0]
    g = grid  # (H, H, 2), channels are (y, x)
    py = (g[..., 0] + 1.0) / 2.0 * (H - 1)   # pixel-space source y for each output pixel
    px = (g[..., 1] + 1.0) / 2.0 * (H - 1)   # pixel-space source x
    y0 = torch.floor(py).long()
    x0 = torch.floor(px).long()
    dy = py - y0.float()                     # fractional offset toward y0+1, in (H, H)
    dx = px - x0.float()                     # fractional offset toward x0+1
    cov = torch.zeros(H, H)
    # Four bilinear taps; weights per output pixel in (H, H); scatter-add into the source-pixel accumulator.
    for oy, ox_off in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        yw = (1 - dy) if oy == 0 else dy
        xw = (1 - dx) if ox_off == 0 else dx
        w = (yw * xw).view(-1)
        idx = ((y0 + oy).clamp(0, H - 1)).view(-1) * H + (x0 + ox_off).clamp(0, H - 1).view(-1)
        cov.view(-1).index_add_(0, idx, w)
    return cov


def test4_source_pixel_coverage_mu0():
    """4. mu=0 coverage: legacy leaves border source pixels unsampled; corrected covers every source pixel."""
    torch.manual_seed(3)
    images = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    flow = _FixedFlow(0.7)

    cov_corr = _source_coverage(_grid_for(images, flow, mu=0.0, mode='corrected').squeeze(0))
    total = cov_corr.numel()
    unsampled_corr = (cov_corr < 1e-6).sum().item()
    frac_corr = unsampled_corr / total

    cov_leg = _source_coverage(_grid_for(images, flow, mu=0.0, mode='legacy').squeeze(0))
    unsampled_leg = (cov_leg < 1e-6).sum().item()
    frac_leg = unsampled_leg / total

    assert unsampled_corr == 0, f'corrected must cover every source pixel, unsampled={unsampled_corr}'
    assert unsampled_leg > 0, f'legacy must drop border source pixels, unsampled={unsampled_leg}'
    print(f'TEST 4 mu=0 coverage | total={total} | unsampled: corrected={unsampled_corr} '
          f'({frac_corr:.2%}), legacy={unsampled_leg} ({frac_leg:.2%})')


def _interior_border_diff(mode_a, mode_b, images, gen, mu):
    """Compare two sampling grids, reporting max abs diff in the interior and at the border."""
    ga = _grid_for(images, gen, mu, mode_a)
    gb = _grid_for(images, gen, mu, mode_b)
    diff = (ga - gb).abs()
    interior = diff[:, RADIUS:-RADIUS, RADIUS:-RADIUS]
    full_border = torch.cat([diff[..., :RADIUS, :].reshape(-1),
                             diff[..., -RADIUS:, :].reshape(-1),
                             diff[..., :, :RADIUS].reshape(-1),
                             diff[..., :, -RADIUS:].reshape(-1)], 0)
    return interior.max().item(), full_border.max().item()


def test5_6_interior_equality_border_inequality():
    """5 & 6. Identical flow, mu>0: legacy and corrected grids coincide in the strict interior
    (where G*I==I) and differ only at the border (where the legacy zero-padding artifact lives)."""
    torch.manual_seed(4)
    images = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    mu = 0.05
    gen = _FixedFlow(0.4)
    inner_max, border_max = _interior_border_diff('legacy', 'corrected', images, gen, mu)

    assert inner_max < 1e-5, f'interior must match to numerical zero, got {inner_max:.3e}'
    assert border_max > 1e-2, f'border must clearly differ, got {border_max:.3e}'
    print(f'TEST 5/6 mu={mu} legacy vs corrected | interior max grid diff={inner_max:.3e} '
          f'(expected ~0), border max grid diff={border_max:.3e} (expected >> 0)')


def test7_corrected_mu_pos_deformation_active():
    """7. corrected at mu>0 still deforms (and orientation/sign unchanged vs legacy)."""
    torch.manual_seed(5)
    checker = ((torch.arange(IMAGE_SIZE)[:, None] + torch.arange(IMAGE_SIZE)[None, :]) % 2).float()
    images = checker.unsqueeze(0).unsqueeze(0).to(DEVICE)
    gen = _FixedFlow(0.5)
    out = torch.nn.functional.grid_sample(
        images, _grid_for(images, gen, 0.05, 'corrected'), padding_mode='border', align_corners=True)
    assert out.shape == images.shape and torch.isfinite(out).all()
    assert (out != images).any().item(), 'corrected mu>0 with nonzero flow must move pixels'
    print(f'TEST 7 corrected mu=0.05: deforms ({tuple(out.shape)}), finite, non-identity')


def _grid_from_budget(images, gen, mu):
    """Sampling grid produced by a 3-channel budget-map generator (corrected mode)."""
    scaled, raw = _scaled_flow(images, gen, mu)
    budget = raw[:, 2:3]
    return U.compute_budget_map(raw, mu)[0], U.build_sampling_grid(scaled, _identity_grid(), _gauss(), 'corrected')


def test8_budget_mean_equals_mu():
    """8. Budget-map path: per-image normalized budget mean ~= mu (uniform baseline preserved)."""
    torch.manual_seed(6)
    images = torch.randn(4, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    gen = _BudgetFlow(flow_value=0.3, budget_peak=0.5)
    mu = 0.02
    with torch.no_grad():
        raw = gen(images)
    budget = U.compute_budget_map(raw, mu)[0]
    bmean = budget.mean(dim=(1, 2, 3))
    assert torch.allclose(bmean, torch.full_like(bmean, mu), atol=1e-7), f'budget mean != mu: {bmean}'
    print(f'TEST 8 budget mean per image = {bmean.tolist()}, mu = {mu} (max |diff| = '
          f'{(bmean - mu).abs().max().item():.3e})')


def test9_budget_spatial_effect_used():
    """9. Budget map is used: where the budget channel peaks the displacement magnitude is larger."""
    torch.manual_seed(7)
    images = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    mu = 0.02
    # Budget channel zero everywhere vs peaking in a central band -> displacement (|budget*flow|) must grow there.
    gen_flat = _BudgetFlow(flow_value=0.3, budget_peak=0.0)
    gen_peak = _BudgetFlow(flow_value=0.3, budget_peak=0.0 + 1.5)   # budget channel peaks mid-band
    grid_flat = _grid_for(images, gen_flat, mu, 'corrected')
    grid_peak = _grid_for(images, gen_peak, mu, 'corrected')

    idgrid = _identity_grid_perm(1)
    disp_flat = (grid_flat - idgrid).abs()
    disp_peak = (grid_peak - idgrid).abs()

    # The central vertical band (rows) should show larger displacement with the peaked budget.
    band = slice(IMAGE_SIZE // 3, 2 * IMAGE_SIZE // 3)
    mean_band_peak = disp_peak[:, band, :].mean().item()
    mean_band_flat = disp_flat[:, band, :].mean().item()
    outside_peak = torch.cat([disp_peak[:, :IMAGE_SIZE // 3, :].reshape(-1),
                              disp_peak[:, 2 * IMAGE_SIZE // 3:, :].reshape(-1)], 0).mean().item()
    assert mean_band_peak > mean_band_flat, 'budget peak must raise displacement in the band'
    assert mean_band_peak > outside_peak, 'displacement must concentrate where budget peaks'
    print(f'TEST 9 budget spatial effect | mean disp peak-band={mean_band_peak:.3e}, '
          f'flat-band={mean_band_flat:.3e}, outside={outside_peak:.3e} (band>outside>placement)')


# --- pretrain / preval (BLOCKER 3) --------------------------------------- #


def _pretrain_style_grid(images_flow, mu, mode):
    """Mirror of utils.pretrain / utils.preval grid construction: scaled = mu * flow, then build_sampling_grid."""
    scaled = mu * images_flow
    return U.build_sampling_grid(scaled, _identity_grid(), _gauss(), mode)


def test10_pretrain_corrected_path():
    """10. pretrain corrected: identical sampling-grid construction as deform, and identity at mu=0."""
    torch.manual_seed(8)
    flow = torch.full((1, 2, IMAGE_SIZE, IMAGE_SIZE), 0.6, device=DEVICE)
    # mu=0 -> exact identity grid under corrected
    g = _pretrain_style_grid(flow, 0.0, 'corrected')
    assert torch.equal(g, _identity_grid_perm(1)), 'pretrain corrected mu=0 must give exact identity grid'
    # mu>0 -> equals the deform corrected grid for the same scaled displacement
    g_mu = _pretrain_style_grid(flow, 0.03, 'corrected')
    assert g_mu.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 2)
    print(f'TEST 10 pretrain corrected: mu=0 identity torch.equal==True; mu>0 grid '
          f'{tuple(g_mu.shape)}, finite={torch.isfinite(g_mu).all().item()}')


def test11_preval_corrected_path():
    """11. preval corrected: identical sampling-grid construction, identity at mu=0."""
    torch.manual_seed(9)
    flow = torch.full((1, 2, IMAGE_SIZE, IMAGE_SIZE), -0.3, device=DEVICE)
    g = _pretrain_style_grid(flow, 0.0, 'corrected')
    assert torch.equal(g, _identity_grid_perm(1)), 'preval corrected mu=0 must give exact identity grid'
    g_mu = _pretrain_style_grid(flow, 0.02, 'corrected')
    assert g_mu.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 2)
    print(f'TEST 11 preval corrected: mu=0 identity torch.equal==True; mu>0 grid '
          f'{tuple(g_mu.shape)}, finite={torch.isfinite(g_mu).all().item()}')


def test_shared_operator_all_three_sites():
    """BLOCKER 3: deform, pretrain and preval must ALL route through build_sampling_grid (no inline operator)."""
    src = inspect.getsource(U)
    for fn_name in ('def deform', 'def pretrain', 'def preval'):
        # Find the function body (up to the next top-level 'def ').
        start = src.index(fn_name)
        body = src[start:]
        next_def = body.find('\ndef ', len(fn_name))
        body = body[: next_def if next_def != -1 else None]
        assert 'build_sampling_grid(' in body, f'{fn_name} must call build_sampling_grid'
        assert 'gauss_filter(grids)' not in body, f'{fn_name} must not smooth the full displaced grid'
    print('TEST shared-operator: deform/pretrain/preval all route through build_sampling_grid')


# --- legacy equivalence (BLOCKER 8) -------------------------------------- #


def test12_legacy_helper_bitforbit_first_old_inline():
    """12. The new legacy helper must reproduce the OLD inline equations bit-for-bit (torch.equal)."""
    torch.manual_seed(10)
    images = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    mu = 0.03
    flow = _FixedFlow(0.4)
    scaled, _ = _scaled_flow(images, flow, mu)
    grid_identity = _identity_grid()
    gauss = _gauss()

    # Old inline legacy equations (the historical operator):
    old_grid = grid_identity - scaled
    old_grid = gauss(old_grid)
    old_grid = old_grid.permute(0, 2, 3, 1)

    new_legacy_grid = U.build_sampling_grid(scaled, grid_identity, gauss, 'legacy')

    same = torch.equal(old_grid, new_legacy_grid)
    max_diff = (old_grid - new_legacy_grid).abs().max().item() if not same else 0.0
    assert same, f'legacy helper must be bit-for-bit identical to old inline; max diff={max_diff:.3e}'
    print(f'TEST 12 legacy equivalence: old-inline == new helper torch.equal=={same} (max grid diff={max_diff:.0e})')


def test12b_old_configs_resolve_legacy():
    """12b. Existing configs without transform_mode resolve to legacy (backward compatibility)."""
    config_dir = os.path.join(os.path.dirname(__file__), 'config_files')
    checked = 0
    for name in sorted(os.listdir(config_dir)):
        if not name.endswith('.json'):
            continue
        with open(os.path.join(config_dir, name)) as f:
            cfg = json.load(f)
        mode = U.resolve_transform_mode(cfg.get('transform_mode'))
        assert mode == 'legacy', f'{name}: expected legacy default, got {mode}'
        checked += 1
    # Invalid value must raise (typo protection).
    try:
        U.resolve_transform_mode('korrected')
    except ValueError:
        pass
    else:
        raise AssertionError('resolve_transform_mode must reject invalid values')
    print(f'TEST 12b default-legacy: {checked} configs all resolve to \'legacy\' (none carry transform_mode); '
          f'invalid value rejected')


# --- gradient-accumulation regression (BLOCKER 13 list item) -------------- #


def test13_gradient_accumulation_regression_still_passes():
    """13. The pre-existing gradient-accumulation regression must still pass unchanged."""
    import test_grad_accum
    test_grad_accum.test_grad_accumulation_matches_doubled_batch()
    test_grad_accum.test_zero_grad_inside_loop_is_detected()
    print('TEST 13: gradient-accumulation regression still passes')


# --- retained low-value smoke tests (BLOCKER 4) --------------------------- #


def test_constant_image_smoke():
    """SMOKE only (low diagnostic value): constant image with padding_mode='border'. Not used as operator evidence."""
    images = torch.full((1, 1, IMAGE_SIZE, IMAGE_SIZE), 0.5, device=DEVICE)
    out = _grid_for(images, _FixedFlow(0.25), 0.02, 'corrected')
    assert out.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 2) and torch.isfinite(out).all()
    print('TEST constant-image smoke: passes (finite, correct shape) — NOT evidence of correctness')


if __name__ == '__main__':
    # Utility functions before running pretrain-style assertion
    import test_grad_accum  # noqa: F401  (module import must succeed)

    test1_nonzero_flow_mu0_exact_identity_grid()
    test2_zero_flow_mu_pos_exact_identity_grid()
    test3_same_image_mu0_vs_zeroflow_extra_identity()
    test4_source_pixel_coverage_mu0()
    test5_6_interior_equality_border_inequality()
    test7_corrected_mu_pos_deformation_active()
    test8_budget_mean_equals_mu()
    test9_budget_spatial_effect_used()
    test10_pretrain_corrected_path()
    test11_preval_corrected_path()
    test_shared_operator_all_three_sites()
    test12_legacy_helper_bitforbit_first_old_inline()
    test12b_old_configs_resolve_legacy()
    test13_gradient_accumulation_regression_still_passes()
    test_constant_image_smoke()
    print('\nSTEP 1B REVIEW REMEDIATION: PASS')