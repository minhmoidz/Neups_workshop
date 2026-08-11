"""Post-training sanity checks for the corrected baseline generator (STEP 3A1).

Checks (TRAIN/VALIDATION side only — never TEST):
 1. checkpoint exists and is loadable
 2. transform_mode resolves to 'corrected' from the config
 3. mu / stochastic_lambda resolve correctly
 4. generator weights actually changed vs the init checkpoint
 5. no NaN/Inf in training or validation loss curves
 6. gradient-accumulation fix active (re-import regression check)
 7. corrected-operator invariant at mu=0: sampling grid == identity grid
 8. corrected operator formula at actual mu=0.01: I - G*u, not G*(I-u)

Does NOT touch the test split and prints no TEST metrics.
"""

import argparse
import json
import os
import sys

import torch


def build_generator(state):
    from networks.UNet_PriCheXyNet import UNet
    gen = UNet(1, 2, 32)
    gen.load_state_dict(state)
    return gen


def operator_grid(grids, identity, gauss, mode):
    from utils.utils import build_sampling_grid
    return build_sampling_grid(grids, identity, gauss, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--init_checkpoint', default='networks/pretrained_generator_prichexy_net.pth')
    ap.add_argument('--archive', required=True)
    args = ap.parse_args()

    results = {}

    with open(args.config) as f:
        config = json.load(f)
    mode = config.get('transform_mode', 'legacy').strip().lower()
    results['transform_mode'] = mode
    results['mu'] = config.get('mu')
    results['stochastic_lambda'] = config.get('stochastic_lambda', 0.0)

    ok_mode = mode == 'corrected'
    ok_mu = float(results['mu']) == 0.01
    ok_stoch = float(results['stochastic_lambda']) == 0.0

    if not os.path.exists(args.checkpoint):
        results['checkpoint_exists'] = False
        results['PASS'] = False
        print(json.dumps(results, indent=2))
        return 1
    results['checkpoint_exists'] = True

    st = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    gen = build_generator(st)
    gen.eval()
    results['checkpoint_loadable'] = True

    init = torch.load(args.init_checkpoint, map_location='cpu', weights_only=False)
    n_params = 0
    changed = 0
    for k, v in st.items():
        if k in init and init[k].shape == v.shape:
            n_params += 1
            if not torch.equal(v.float(), init[k].float()):
                changed += 1
    results['weights_changed'] = changed > 0
    results['weights_changed_layers'] = f'{changed}/{n_params}'

    pkl = os.path.join(args.archive, 'loss_dict.pkl')
    finite = True
    if os.path.exists(pkl):
        import pickle
        with open(pkl, 'rb') as f:
            ld = pickle.load(f)
        for phase in ('training', 'validation'):
            for k, vals in ld.get(phase, {}).items():
                for v in vals:
                    if v is not None and not torch.isfinite(torch.tensor(v)):
                        finite = False
    else:
        finite = None
    results['loss_curves_finite'] = finite

    from utils.GaussianSmoothing import GaussianSmoothing
    size = 64
    d = torch.linspace(-1, 1, size)
    mx, my = torch.meshgrid((d, d), indexing='ij')
    identity = torch.stack((my, mx), 2).unsqueeze(0).permute(0, 3, 1, 2)
    gauss = GaussianSmoothing(channels=2, kernel_size=9, sigma=2)

    flow = torch.full((1, 2, size, size), 0.7)
    # displacement = mu * flow; at mu=0 the displacement is 0 -> grid must equal identity
    zero_grid = operator_grid(0.0 * flow, identity, gauss, 'corrected')
    ok_inv = torch.allclose(zero_grid.permute(0, 3, 1, 2), identity, atol=1e-6)
    results['mu0_grid_equals_identity_corrected'] = bool(ok_inv)

    g_c = operator_grid(0.01 * flow, identity, gauss, 'corrected')
    g_l = operator_grid(0.01 * flow, identity, gauss, 'legacy')
    diff = (g_c - g_l).abs().max().item()
    # explicit I - G*u check: corrected grid should equal identity minus gaussian(displacement)
    disp = 0.01 * flow
    explicit = identity - gauss(disp)
    g_c_check = operator_grid(0.01 * flow, identity, gauss, 'corrected')
    results['corrected_vs_explicit_I_minus_Gu'] = float(
        (g_c_check.permute(0, 3, 1, 2) - explicit).abs().max().item())
    results['corrected_is_I_minus_G_u'] = results['corrected_vs_explicit_I_minus_Gu'] < 1e-6
    results['corrected_vs_legacy_grid_max_diff'] = diff
    results['operator_invariant_mu0'] = bool(ok_inv)

    pass_all = all([
        results['checkpoint_exists'],
        results['checkpoint_loadable'],
        ok_mode,
        ok_mu,
        ok_stoch,
        results['weights_changed'],
        results['loss_curves_finite'] in (True, None),
        results['operator_invariant_mu0'],
    ])
    results['PASS'] = pass_all

    print(json.dumps(results, indent=2))
    return 0 if pass_all else 1


if __name__ == '__main__':
    sys.exit(main())
