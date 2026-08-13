"""CHECK 8 standalone: real-batch full-graph backward trace (fresh process)."""

import json
import torch
import torch.nn.functional as F

from research_agent.ibr.ibr_model import IBRModel
from research_agent.ibr.s1_loss import compute_s1_loss
from research_agent.ibr.train_s1_stages import SingleImageLabels, _load_frozen
from research_agent.ibr.losses import LAMBDA_REC, LAMBDA_PATH, LAMBDA_ANAT, LAMBDA_ZID, LAMBDA_ADV

OUT = {}

def main():
    device = 'cuda'
    torch.manual_seed(42)
    ds = SingleImageLabels('train', seed=42)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False,
                                         num_workers=4, pin_memory=True)
    x, y_path, names, x_donor, donor_names = next(iter(loader))
    x = x.to(device); x_donor = x_donor.to(device); y_path = y_path.to(device)
    y_pair = torch.zeros(x.shape[0], 1, device=device)

    m8 = IBRModel().to(device)
    f8 = _load_frozen(device)
    egv8 = torch.optim.Adam(list(m8.encoder.parameters()) + list(m8.decoder.parameters())
                            + list(m8.verifier.parameters()), lr=1e-4)
    adv8 = torch.optim.Adam(m8.adv.parameters(), lr=1e-4)

    total, parts = compute_s1_loss(m8, f8, x, x_donor, y_path, y_pair, return_parts=True)
    total.backward()
    print('parts:', {k: '%.6e' % v for k, v in parts.items()})
    groups = {'encoder': m8.encoder, 'decoder': m8.decoder, 'verifier': m8.verifier, 'adv': m8.adv}
    norms = {}
    for name, mod in groups.items():
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        norms[name] = torch.cat([g.flatten() for g in gs]).norm().item() if gs else 0.0
        print('grad norm %-8s = %.6e' % (name, norms[name]))
    frozen_grads = sum(1 for p in list(f8.classifier.parameters()) + list(f8.segmenter.parameters())
                       if p.grad is not None)
    print('frozen model params with grad (must be 0):', frozen_grads)
    # verifier head delta after one step
    vstate_before = {k: v.clone() for k, v in m8.verifier.state_dict().items()}
    egv8.step()
    vstate_after = {k: v.clone() for k, v in m8.verifier.state_dict().items()}
    vdelta = max((vstate_after[k] - vstate_before[k]).abs().max().item() for k in vstate_before)
    print('verifier head max delta after step: %.6f' % vdelta)
    adv_before = {k: v.clone() for k, v in m8.adv.state_dict().items()}
    adv8.step()
    adv_after = {k: v.clone() for k, v in m8.adv.state_dict().items()}
    adelta = max((adv_after[k] - adv_before[k]).abs().max().item() for k in adv_before)
    print('adv head max delta after step: %.6f' % adelta)

    OUT = {'parts': parts, 'grad_norms': norms, 'frozen_grad_count': frozen_grads,
           'verifier_delta': vdelta, 'adv_delta': adelta,
           'y_pair_mean': y_pair.mean().item()}
    with open('/tmp/opencode/check8.json', 'w') as f:
        json.dump(OUT, f, indent=2)
    print('CHECK8 SAVED')


if __name__ == '__main__':
    main()