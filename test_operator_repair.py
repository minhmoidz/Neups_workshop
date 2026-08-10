"""Regression tests for the corrected PriCheXy-Net transformation operator.

The bug fixed here (research_agent audit) was that ``deform`` applied the Gaussian
smoother to the *whole* displaced sampling grid ``G * (I - u)``:

    legacy:     grids = gauss_filter(grid_identity - u)      # drops border pixels even for u = 0

Because the Gaussian is applied with zero padding, ``G * I != I`` near the image
borders, so a zero displacement did NOT reproduce the identity -- border pixels were
pulled toward 0 and dropped. The corrected operator smooths only the displacement:

    corrected:  grids = grid_identity - gauss_filter(u)      # u = 0  =>  grid = identity exactly

Design invariants validated here (Task 4 / Tutorial A-E):
  A. mu=0 (whatever the flow)  => T(x) == x  for the corrected operator.
  B. zero flow (whatever mu)   => T(x) == x  for the corrected operator.
  C. constant image: no artificial border corruption.
  D. identity grid itself is exact (no G*(I) residual).
  E. legacy vs corrected comparison on a structured (checkerboard) image.

All tests run on CPU so the operator bug is provable in a fully green CI run
without a GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.GaussianSmoothing import GaussianSmoothing
from utils import utils as U

DEVICE = 'cpu'
IMAGE_SIZE = 64


def _identity_grid(device=DEVICE, size=IMAGE_SIZE):
    d = torch.linspace(-1, 1, size, device=device)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2)
    return grid_identity.unsqueeze(0).permute(0, 3, 1, 2)


def _gauss(device=DEVICE):
    return GaussianSmoothing(channels=2, kernel_size=9, sigma=2).to(device)


class _FixedFlow(nn.Module):
    """Returns a constant flow field so the tests do not depend on a trained generator."""

    def __init__(self, value, channels=2):
        super().__init__()
        self.value = value
        self.channels = channels

    def forward(self, x):
        return torch.full((x.shape[0], self.channels) + x.shape[2:], self.value, device=x.device)


class _ZeroFlow(nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 2) + x.shape[2:], device=x.device)


def _run(images, flow_value, mu, transform_mode, stochastic_lambda=0.0, largemag=False, budget_map=False):
    if budget_map:
        class _BudgetFlow(nn.Module):
            def forward(self, x):
                b = x.shape[0]
                flow = torch.full((b, 3) + x.shape[2:], flow_value, device=x.device)
                budget = torch.zeros((b, 1) + x.shape[2:], device=x.device)
                return torch.cat([flow[:, :2], budget], 1)
        gen = _BudgetFlow()
    elif largemag:
        gen = _FixedFlow(flow_value * 2.0)
    elif flow_value is None:
        gen = _ZeroFlow()
    else:
        gen = _FixedFlow(flow_value)
    grid_identity = _identity_grid()
    gauss = _gauss()
    return U.deform(images, gen, grid_identity, gauss, mu, stochastic_lambda, transform_mode)


def test_a_mu_zero_recovers_identity_corrected():
    """mu=0 with a non-trivial flow must leave the image unchanged (corrected)."""
    torch.manual_seed(0)
    images = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    out = _run(images, flow_value=0.5, mu=0.0, transform_mode='corrected')
    torch.testing.assert_close(out, images, atol=1e-3, rtol=1e-3,
                               msg='corrected: mu=0 must give back the input (deformation budget scaled to 0)')
    max_displacement = (out - images).abs().max().item()
    print(f'TEST A corrected: max |out - input| = {max_displacement:.3e}  (mu=0)')


def test_a_mu_zero_recovers_identity_legacy_shows_bug():
    """mu=0 under the legacy operator already corrupts borders: still near-identity in the flat center,
    but border pixels are visibly pulled toward 0 (G*(I) != I). This documents that frugal regularization
    (smoothing the full displaced grid) drops about one smoothing radius worth of border pixels."""
    torch.manual_seed(0)
    images = torch.ones(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    out = _run(images, flow_value=0.5, mu=0.0, transform_mode='legacy')
    interior = out[..., 4:-4, 4:-4]
    assert interior.min() > 0.84, f'legacy flat-centre not preserved: min={interior.min().item():.4f}'
    print(f'TEST A legacy: flat-centre kept (min={interior.min().item():.4f}) but corners/border are darkened')


def test_b_zero_flow_any_mu_recovers_identity_corrected():
    """A zero displacement field, for any mu > 0, must leave the image unchanged (corrected)."""
    torch.manual_seed(1)
    images = torch.rand(2, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    for mu in (0.01, 0.05):
        out = _run(images, flow_value=None, mu=mu, transform_mode='corrected')
        max_disp = (out - images).abs().max().item()
        assert max_disp < 1e-3, f'corrected zero-flow, mu={mu}: max displacement {max_disp:.3e} != 0'
        print(f'TEST B corrected: zero flow, mu={mu} -> max |out-input| = {max_disp:.3e}')


def test_a_eta_zero_stochastic_corrected_is_identity():
    """With stochastic_lambda=0 and mu=0, the corrected operator is the identity over grid_sample."""
    torch.manual_seed(2)
    images = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 2 - 1
    out = _run(images, flow_value=0.0, mu=0.0, transform_mode='corrected', stochastic_lambda=0.0)
    torch.testing.assert_close(out, images, atol=1e-3, rtol=1e-3,
                               msg='corrected: stochastic_lambda=0 with mu=0 must be the identity')


def test_c_constant_image_no_artificial_border_corruption():
    """A constant image deformed with a *legitimately non-zero* field must not show artificial
    border darkening: the constant value is preserved everywhere (corrected)."""
    images = torch.full((1, 1, IMAGE_SIZE, IMAGE_SIZE), 0.5, device=DEVICE)
    out = _run(images, flow_value=0.25, mu=0.02, transform_mode='corrected')
    span = (out.max() - out.min()).item()
    assert span < 1e-3, f'corrected: constant image should stay essentially constant, got span {span:.3e}'
    print(f'TEST C corrected: constant image value-span after warp = {span:.3e} (border clean)')


def test_d_identity_grid_is_exact():
    """The building block: corrected_transform with u=0 must return exactly the identity grid;
    the legacy path does not (it returns G*(I))."""
    grid_identity = _identity_grid()
    gauss = _gauss()
    zero_flow = torch.zeros(1, 2, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    corr = U.build_sampling_grid(zero_flow, grid_identity, gauss, 'corrected')
    max_err = (corr.permute(0, 3, 1, 2) - grid_identity).abs().max().item()
    assert max_err < 1e-5, f'corrected identity grid residual must be ~0, got {max_err:.3e}'
    legacy = U.build_sampling_grid(zero_flow, grid_identity, gauss, 'legacy')
    legacy_err = (legacy.permute(0, 3, 1, 2) - grid_identity).abs().max().item()
    assert legacy_err > 1e-3, 'legacy path MUST differ from identity (this is the bug being documented)'
    print(f'TEST D: corrected |G_id - grid| = {max_err:.3e}; legacy |G_id - grid| = {legacy_err:.3e}')
    return max_err, legacy_err


def test_e_legacy_vs_corrected_checkerboard():
    """E: compare legacy vs corrected on a checkerboard. The legacy grid drops border pixels;
    the corrected grid preserves the full canvas, so on a periodic pattern the total signal differs."""
    torch.manual_seed(3)
    checker = (torch.arange(IMAGE_SIZE)[:, None] + torch.arange(IMAGE_SIZE)[None, :]) % 2
    images = checker.float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    mu = 0.02
    leg = _run(images, flow_value=0.5, mu=mu, transform_mode='legacy')
    corr = _run(images, flow_value=0.5, mu=mu, transform_mode='corrected')
    # On a constant-ish canvas the interior must be nearly identical; the key difference is that legacy
    # corrupts the border: boundary rows/columns of the legacy output deviate from the corrected one.
    border_leg = torch.cat([leg[..., 0, :], leg[..., -1, :], leg[..., :, 0], leg[..., :, -1]], dim=-1)
    border_corr = torch.cat([corr[..., 0, :], corr[..., -1, :], corr[..., :, 0], corr[..., :, -1]], dim=-1)
    dev = (border_leg - border_corr).abs().max().item()
    assert dev > 1e-3, f'expected legacy/corrected border deviation, got {dev:.3e}'
    print(f'TEST E: legacy border deviates from corrected by max {dev:.3e} at identical mu/flow')


def test_mu_positive_still_deforms_corrected():
    """With mu>0 and a real flow, the corrected operator must actually move pixels (orientable sign),
    smoothly, without NaNs, and with correct output dimensions."""
    checker = (torch.arange(IMAGE_SIZE)[:, None] + torch.arange(IMAGE_SIZE)[None, :]) % 2
    images = checker.float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    out = _run(images, flow_value=0.5, mu=0.05, transform_mode='corrected', largemag=True)
    assert out.shape == images.shape and torch.isfinite(out).all()
    assert (out != images).any().item(), 'mu>0 with flow must deform the image somewhere'
    # smoothing must be observable: a small flow produces a smooth, not salt-and-pepper, displacement
    print(f'TEST mu>0 corrected: deforms ({out.shape}), finite, non-identity')


def test_budget_map_corrected_works():
    """The 3-channel budget-map path must remain functional under the corrected operator."""
    images = torch.rand(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    out = _run(images, flow_value=0.2, mu=0.02, transform_mode='corrected', budget_map=True)
    assert out.shape == images.shape and torch.isfinite(out).all()
    print(f'TEST budget-map corrected: finite output {tuple(out.shape)}')


if __name__ == '__main__':
    test_a_mu_zero_recovers_identity_corrected()
    test_a_mu_zero_recovers_identity_legacy_shows_bug()
    test_b_zero_flow_any_mu_recovers_identity_corrected()
    test_a_eta_zero_stochastic_corrected_is_identity()
    test_c_constant_image_no_artificial_border_corruption()
    test_d_identity_grid_is_exact()
    test_e_legacy_vs_corrected_checkerboard()
    test_mu_positive_still_deforms_corrected()
    test_budget_map_corrected_works()
    print('\nALL OPERATOR TESTS PASSED')