"""POST-FIX MICRO-VALIDATION (STEP 6C-D). No epoch-scale training."""

import json
import torch
import torch.nn.functional as F

from research_agent.ibr.ibr_model import IBRModel
from research_agent.ibr.s1_loss import compute_s1_loss
from research_agent.ibr.losses import LAMBDA_ZID, LAMBDA_ADV, GRL_LAMBDA
from research_agent.ibr.grl import GradientReversalLayer
from research_agent.ibr.train_s1_stages import SingleImageLabels, _load_frozen

OUT = {}

def main():
    device = 'cuda'
    torch.manual_seed(42)
    ds = SingleImageLabels('train', seed=42)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False,
                                         num_workers=4, pin_memory=True)
    x, y_path, names, x_donor, donor_names, x_partner, y_pair = next(iter(loader))
    print('TRAIN batch y_pair: pos=%d neg=%d' % ((y_pair==1).sum().item(), (y_pair==0).sum().item()))

    m = IBRModel().to(device)
    f = _load_frozen(device)
    egv = torch.optim.Adam(list(m.encoder.parameters()) + list(m.decoder.parameters())
                           + list(m.verifier.parameters()), lr=1e-4)
    adv_opt = torch.optim.Adam(m.adv.parameters(), lr=1e-4)

    x = x.to(device); x_donor = x_donor.to(device); x_partner = x_partner.to(device)
    y_path = y_path.to(device); y_pair = y_pair.to(device)

    total, parts = compute_s1_loss(m, f, x, x_donor, y_path, x_partner, y_pair, return_parts=True)
    print('POST-FIX real batch parts:')
    for k, v in parts.items():
        print('  %-8s = %.8e  finite=%s  >0=%s' % (k, v, bool(torch.isfinite(torch.tensor(v))), v > 0))

    total.backward()
    groups = {'encoder': m.encoder, 'decoder': m.decoder, 'verifier': m.verifier, 'adv': m.adv}
    norms = {}
    for name, mod in groups.items():
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        norms[name] = torch.cat([g.flatten() for g in gs]).norm().item() if gs else 0.0
        print('grad norm %-8s = %.6e' % (name, norms[name]))
    frozen_grads = sum(1 for p in list(f.classifier.parameters()) + list(f.segmenter.parameters())
                       if p.grad is not None)
    print('frozen grads (must be 0):', frozen_grads)

    # verifier/adversary params change after step
    vb = {k: v.clone() for k, v in m.verifier.state_dict().items()}
    ab = {k: v.clone() for k, v in m.adv.state_dict().items()}
    egv.step(); adv_opt.step()
    va = {k: v.clone() for k, v in m.verifier.state_dict().items()}
    aa = {k: v.clone() for k, v in m.adv.state_dict().items()}
    vd = max((va[k]-vb[k]).abs().max().item() for k in vb)
    ad = max((aa[k]-ab[k]).abs().max().item() for k in ab)
    print('verifier delta after step: %.6f  adv delta: %.6f' % (vd, ad))

    OUT['real_batch'] = {'parts': parts, 'grad_norms': norms, 'frozen_grads': frozen_grads,
                         'verifier_delta': vd, 'adv_delta': ad, 'raw_zid': parts['L_zid'],
                         'raw_adv': parts['L_adv'], 'lambda_zid': LAMBDA_ZID, 'lambda_adv': LAMBDA_ADV,
                         'y_pair_pos': (y_pair == 1).sum().item(), 'y_pair_neg': (y_pair == 0).sum().item()}

    # ---- GRL on real data ----
    encg = IBRModel().to(device).encoder
    advg = IBRModel().to(device).adv
    grl = GradientReversalLayer(lambd=GRL_LAMBDA)
    sel = next(iter(loader))
    ia = sel[0][:8].to(device); ib = sel[5][:8].to(device)
    yl = sel[6][:8].to(device)
    encg.zero_grad(); advg.zero_grad()
    za, ma, _ = encg(ia); zb, mb, _ = encg(ib)
    lossA = F.binary_cross_entropy_with_logits(advg(ma, mb), yl)
    lossA.backward()
    gA = torch.cat([p.grad.flatten() for p in encg.parameters() if p.grad is not None])
    encg.zero_grad(); advg.zero_grad()
    za2, ma2, _ = encg(ia); zb2, mb2, _ = encg(ib)
    lossB = F.binary_cross_entropy_with_logits(advg(grl(ma2), grl(mb2)), yl)
    lossB.backward()
    gB = torch.cat([p.grad.flatten() for p in encg.parameters() if p.grad is not None])
    cos = F.cosine_similarity(gA.unsqueeze(0), gB.unsqueeze(0)).item()
    print('GRL: ||g_normal||=%.4e ||g_grl||=%.4e cos=%.6f' % (gA.norm().item(), gB.norm().item(), cos))
    OUT['grl'] = {'g_normal': gA.norm().item(), 'g_grl': gB.norm().item(), 'cos': cos}

    # ---- tiny-batch overfit of verifier with balanced pairs ----
    from research_agent.ibr.train_s1_stages import load_x_image, IMAGE_PATH
    import numpy as np
    pairs = np.loadtxt('image_pairs/image_pairs_validation_2000.txt', dtype=str)
    sel = [r for r in pairs if r[2] == '1.0'][:8] + [r for r in pairs if r[2] == '0.0'][:8]
    ima = torch.stack([load_x_image(IMAGE_PATH + r[0]) for r in sel]).to(device)
    imb = torch.stack([load_x_image(IMAGE_PATH + r[1]) for r in sel]).to(device)
    yl2 = torch.tensor([1.0]*8 + [0.0]*8, device=device).unsqueeze(1)

    # with a FRESH encoder whose z_id is then frozen, overfit the verifier head
    enc_over = IBRModel().to(device).encoder
    ver = IBRModel().to(device).verifier
    optv = torch.optim.Adam(ver.parameters(), lr=1e-3)
    with torch.no_grad():
        za2 = enc_over(ima)[0]; zb2 = enc_over(imb)[0]
    tr = []
    for s in range(300):
        optv.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ver(za2, zb2), yl2)
        loss.backward()
        optv.step()
        tr.append(loss.item())
    with torch.no_grad():
        acc = (((ver(za2, zb2) > 0).float() == yl2).float()).mean().item()
    print('verifier tiny-batch overfit: loss %.4f -> %.4f  acc %.4f' % (tr[0], tr[-1], acc))
    OUT['verifier_overfit'] = {'loss_first': tr[0], 'loss_last': tr[-1], 'acc': acc}

    # ---- tiny-batch overfit of adversary (no GRL, detached z_med) ----
    adv_o = IBRModel().to(device).adv
    opta = torch.optim.Adam(adv_o.parameters(), lr=1e-3)
    enc_o2 = IBRModel().to(device).encoder
    with torch.no_grad():
        ma2 = enc_o2(ima)[1].detach(); mb2 = enc_o2(imb)[1].detach()
    tra = []
    for s in range(300):
        opta.zero_grad()
        loss = F.binary_cross_entropy_with_logits(adv_o(ma2, mb2), yl2)
        loss.backward()
        opta.step()
        tra.append(loss.item())
    with torch.no_grad():
        acca = (((adv_o(ma2, mb2) > 0).float() == yl2).float()).mean().item()
    print('adversary tiny-batch overfit: loss %.4f -> %.4f  acc %.4f' % (tra[0], tra[-1], acca))
    OUT['adv_overfit'] = {'loss_first': tra[0], 'loss_last': tra[-1], 'acc': acca}

    with open('/tmp/opencode/postfix_validation.json', 'w') as f:
        json.dump(OUT, f, indent=2)
    print('POST-FIX VALIDATION SAVED')


if __name__ == '__main__':
    main()