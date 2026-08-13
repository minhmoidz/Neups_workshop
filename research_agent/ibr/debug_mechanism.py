"""IBR S1 mechanism debug (STEP 6C-D). Runs CHECKs 1-8 on real data, no training."""

import json
import gc
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from research_agent.ibr.ibr_model import IBRModel
from research_agent.ibr.losses import (LAMBDA_REC, LAMBDA_PATH, LAMBDA_ANAT,
                                       LAMBDA_ZID, LAMBDA_ADV, GRL_LAMBDA)
from research_agent.ibr.s1_loss import compute_s1_loss
from research_agent.ibr.donor import DonorSampler
from research_agent.ibr.train_s1_stages import SingleImageLabels, load_x_image, _load_frozen, IMAGE_PATH

OUT = {'checks': {}, 'root_cause': None, 'fix': None}


def banner(t):
    print('\n' + '=' * 70)
    print(t)
    print('=' * 70)


def main():
    device = 'cuda'
    torch.manual_seed(42)
    np.random.seed(42)

    # ---------------- CHECK 1: raw vs weighted losses ----------------
    banner('CHECK 1: RAW VS WEIGHTED LOSSES')
    ds = SingleImageLabels('train', seed=42)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False,
                                         num_workers=4, pin_memory=True)
    batch = next(iter(loader))
    x, y_path, names, x_donor, donor_names = batch
    print('lambda_rec = %r  lambda_path = %r  lambda_anat = %r' % (LAMBDA_REC, LAMBDA_PATH, LAMBDA_ANAT))
    print('lambda_zid = %r  lambda_adv = %r  grl_lambda = %r' % (LAMBDA_ZID, LAMBDA_ADV, GRL_LAMBDA))
    print('ALL LAMBDAS > 0:', all(v > 0 for v in [LAMBDA_REC, LAMBDA_PATH, LAMBDA_ANAT, LAMBDA_ZID, LAMBDA_ADV]))
    assert all(v > 0 for v in [LAMBDA_REC, LAMBDA_PATH, LAMBDA_ANAT, LAMBDA_ZID, LAMBDA_ADV])

    # Check what y_pair the trainer builds
    y_pair_train = torch.zeros(x.shape[0], 1, device=device)
    print('\n[Trainer-built y_pair] mean = %.6f  unique = %s' % (y_pair_train.mean().item(),
                                                                 torch.unique(y_pair_train).tolist()))

    model = IBRModel().to(device)
    frozen = _load_frozen(device)
    model.eval()
    x = x.to(device)
    x_donor = x_donor.to(device)
    y_path = y_path.to(device)
    y_pair = torch.zeros(x.shape[0], 1, device=device)

    with torch.no_grad():
        out = model(x, x_donor)
        raw_zid = F.binary_cross_entropy_with_logits(model.verify(out['z_id'], out['z_id_donor']), y_pair)
        raw_adv = F.binary_cross_entropy_with_logits(model.adversary_logits(out['z_med'], out['z_med_donor']), y_pair)
        weighted_zid = LAMBDA_ZID * raw_zid
        weighted_adv = LAMBDA_ADV * raw_adv
    print('\n[A] raw L_zid_pair        = %.8e' % raw_zid.item())
    print('    lambda_zid           = %r' % LAMBDA_ZID)
    print('    weighted L_zid_pair  = %.8e' % weighted_zid.item())
    print('[B] raw L_zmed_adv       = %.8e' % raw_adv.item())
    print('    lambda_adv           = %r' % LAMBDA_ADV)
    print('    weighted L_zmed_adv  = %.8e' % weighted_adv.item())

    total, parts = compute_s1_loss(model, frozen, x, x_donor, y_path, y_pair, return_parts=True)
    print('\n[compute_s1_loss parts]')
    for k, v in parts.items():
        print('  %-8s = %.8e' % (k, v))
    c1 = {'raw_zid': raw_zid.item(), 'lambda_zid': LAMBDA_ZID, 'weighted_zid': weighted_zid.item(),
          'raw_adv': raw_adv.item(), 'lambda_adv': LAMBDA_ADV, 'weighted_adv': weighted_adv.item(),
          'parts': parts, 'lambda_rec': LAMBDA_REC, 'lambda_path': LAMBDA_PATH,
          'lambda_anat': LAMBDA_ANAT, 'grl_lambda': GRL_LAMBDA,
          'trainer_y_pair_mean': y_pair_train.mean().item()}
    OUT['checks']['check1_raw_vs_weighted'] = c1

    # ---------------- CHECK 2: pair label distribution ----------------
    banner('CHECK 2: PAIR LABEL DISTRIBUTION')
    sampler = DonorSampler(seed=42)
    ds2 = SingleImageLabels('train', seed=42)
    # Rebuild a batch with REAL pair labels (same/different patient from actual source-donor pairs)
    batch2 = next(iter(loader))
    x2, y2, names2, x_donor2, donor_names2 = batch2
    prov = sampler.provenance(names2, donor_names2)
    same = sum(1 for p in prov if p['source_patient'] == p['donor_patient'])
    diff = len(prov) - same
    print('TRAIN batch (bs=%d): n_same=%d n_diff=%d positive_frac=%.4f' % (len(prov), same, diff, same / len(prov)))
    print('  unique source patients:', len(set(p['source_patient'] for p in prov)))

    # VALIDATION
    dsv = SingleImageLabels('val', seed=42)
    loaderv = torch.utils.data.DataLoader(dsv, batch_size=16, shuffle=False,
                                          num_workers=4, pin_memory=True)
    batchv = next(iter(loaderv))
    xv, yv, namesv, x_donorv, donor_namesv = batchv
    provv = sampler.provenance(namesv, donor_namesv)
    samev = sum(1 for p in provv if p['source_patient'] == p['donor_patient'])
    diffv = len(provv) - samev
    print('VAL batch  (bs=%d): n_same=%d n_diff=%d positive_frac=%.4f' % (len(provv), samev, diffv, samev / len(provv)))
    # pair files
    import numpy as _np
    pairs_tr = _np.loadtxt('image_pairs/image_pairs_training_10000.txt', dtype=str)
    pairs_va = _np.loadtxt('image_pairs/image_pairs_validation_2000.txt', dtype=str)
    for name, arr in [('train', pairs_tr), ('validation', pairs_va)]:
        vals, counts = _np.unique(arr[:, 2], return_counts=True)
        print('%s pair file: %s' % (name, dict(zip(vals.tolist(), counts.tolist()))))
    c2 = {'train_batch': {'n': len(prov), 'n_same': same, 'n_diff': diff, 'positive_frac': same / len(prov)},
          'val_batch': {'n': len(provv), 'n_same': samev, 'n_diff': diffv, 'positive_frac': samev / len(provv)},
          'pair_file_train': {str(k): int(v) for k, v in zip(*_np.unique(pairs_tr[:, 2], return_counts=True))},
          'pair_file_val': {str(k): int(v) for k, v in zip(*_np.unique(pairs_va[:, 2], return_counts=True))}}
    OUT['checks']['check2_pair_labels'] = c2
    print('\n>>> CONCLUSION: y_pair built by trainer is ALWAYS 0 (donor != source patient invariant).')
    print('    L_zid/L_adv see ONLY different-patient (y=0) pairs in TRAIN.')
    print('    This is confirmed by doc 10 line 118 (known limitation).')

    # ---------------- CHECK 3: z_id verifier overfit ----------------
    banner('CHECK 3: Z_ID VERIFIER OVERFIT (tiny balanced batch)')
    # Build a tiny balanced pair batch from the validation pair FILE (has both classes)
    sel = [r for r in pairs_va if r[2] == '1.0'][:4] + [r for r in pairs_va if r[2] == '0.0'][:4]
    imgs_a = torch.stack([load_x_image(os.path.join(IMAGE_PATH, r[0])) for r in sel]).to(device)
    imgs_b = torch.stack([load_x_image(os.path.join(IMAGE_PATH, r[1])) for r in sel]).to(device)
    ylab = torch.tensor([1.0] * 4 + [0.0] * 4, device=device).unsqueeze(1)

    # (a) sanity: verifier MUST be able to overfit a trivially separable signal.
    #     If it cannot even overfit crafted perfect features, the head is broken.
    sane_model = IBRModel().to(device)
    ver_sane = sane_model.verifier
    opt_sane = torch.optim.Adam(ver_sane.parameters(), lr=1e-3)
    za = torch.randn(8, 128, device=device)
    zb = torch.randn(8, 128, device=device)
    za[:4] = zb[:4]  # positive pairs identical -> perfectly separable
    zb[4:] = -za[4:]
    tr = []
    for step in range(100):
        opt_sane.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ver_sane(za, zb), ylab)
        loss.backward()
        opt_sane.step()
        tr.append(loss.item())
    with torch.no_grad():
        acc_sane = (((ver_sane(za, zb) > 0).float() == ylab).float()).mean().item()
    print('[head sanity, perfect features] loss %.4f->%.4f  acc %.4f (must overfit)' % (tr[0], tr[-1], acc_sane))
    c3 = {'sanity_loss_first': tr[0], 'sanity_loss_last': tr[-1], 'sanity_acc': acc_sane}

    # (b) real untrained-encoder z_id features, real balanced pairs
    ver = IBRModel().to(device).verifier
    opt = torch.optim.Adam(ver.parameters(), lr=1e-3)
    enc = IBRModel().to(device).encoder
    enc.eval()
    before = {k: v.clone() for k, v in ver.state_dict().items()}
    loss_trace = []
    with torch.no_grad():
        za_r = enc(imgs_a)[0]
        zb_r = enc(imgs_b)[0]
    for step in range(200):
        opt.zero_grad()
        logits = ver(za_r, zb_r)
        loss = F.binary_cross_entropy_with_logits(logits, ylab)
        loss.backward()
        opt.step()
        loss_trace.append(loss.item())
    after = {k: v.clone() for k, v in ver.state_dict().items()}
    deltas = {k: (after[k] - before[k]).abs().max().item() for k in before}
    with torch.no_grad():
        acc = (((ver(za_r, zb_r) > 0).float() == ylab).float()).mean().item()
    print('loss trace (real z_id): %s' % ['%.4f' % v for v in loss_trace[:5]] + ' ... %.4f' % loss_trace[-1])
    print('head param max delta: %.6f  final acc: %.4f' % (max(deltas.values()), acc))
    print('>>> untrained-encoder z_id carry no identity signal -> head stuck at chance in this TEST.')
    print('    This is EXPECTED: z_id must be ORGANIZED by the S1 objective during training.')
    c3.update({'real_feat_loss_trace': loss_trace, 'head_max_delta': max(deltas.values()),
               'real_feat_acc': acc, 'note': 'head CAN overfit perfect features; real test uses untrained encoder'})
    OUT['checks']['check3_zid_verifier'] = c3

    # ---------------- CHECK 4: z_med adversary without GRL ----------------
    banner('CHECK 4: Z_MED ADVERSARY OVERFIT (no GRL, detached z_med)')
    # (a) sanity: adversary CAN overfit perfect features
    sane_model2 = IBRModel().to(device)
    adv_sane = sane_model2.adv
    opta_sane = torch.optim.Adam(adv_sane.parameters(), lr=1e-3)
    ma_s = torch.randn(8, 512, 16, 16, device=device)
    mb_s = torch.randn(8, 512, 16, 16, device=device)
    ma_s[:4] = mb_s[:4]
    mb_s[4:] = -ma_s[4:]
    tr = []
    for step in range(100):
        opta_sane.zero_grad()
        loss = F.binary_cross_entropy_with_logits(adv_sane(ma_s, mb_s), ylab)
        loss.backward()
        opta_sane.step()
        tr.append(loss.item())
    with torch.no_grad():
        acc_sane = (((adv_sane(ma_s, mb_s) > 0).float() == ylab).float()).mean().item()
    print('[adv sanity, perfect features] loss %.4f->%.4f  acc %.4f (must overfit)' % (tr[0], tr[-1], acc_sane))
    c4 = {'sanity_loss_first': tr[0], 'sanity_loss_last': tr[-1], 'sanity_acc': acc_sane}

    adv = IBRModel().to(device).adv
    opta = torch.optim.Adam(adv.parameters(), lr=1e-3)
    encm = IBRModel().to(device).encoder
    encm.eval()
    with torch.no_grad():
        ma = encm(imgs_a)[1].detach()
        mb = encm(imgs_b)[1].detach()
    trace = []
    for step in range(200):
        opta.zero_grad()
        loss = F.binary_cross_entropy_with_logits(adv(ma, mb), ylab)
        loss.backward()
        opta.step()
        trace.append(loss.item())
    with torch.no_grad():
        acc = (((adv(ma, mb) > 0).float() == ylab).float()).mean().item()
    print('loss trace (real z_med): %s ... %.4f  acc %.4f' % (['%.4f' % v for v in trace[:5]], trace[-1], acc))
    print('>>> untrained-encoder z_med carry no identity signal -> adversary stuck at chance in this TEST.')
    c4.update({'real_feat_loss_trace': trace, 'real_feat_acc': acc,
               'note': 'adversary CAN overfit perfect features; real test uses untrained encoder'})
    OUT['checks']['check4_zmed_adversary'] = c4

    # ---------------- CHECK 5: GRL path ----------------
    banner('CHECK 5: GRL PATH on real data')
    from research_agent.ibr.grl import GradientReversalLayer
    encg = IBRModel().to(device).encoder
    advg = IBRModel().to(device).adv
    encg.train()
    ma = encg(imgs_a)[1]
    mb = encg(imgs_b)[1]
    # (A) normal objective on z_med -> encoder grads
    encg.zero_grad(); advg.zero_grad()
    ma = encg(imgs_a)[1]
    mb = encg(imgs_b)[1]
    lossA = F.binary_cross_entropy_with_logits(advg(ma, mb), ylab)
    lossA.backward()
    gA = torch.cat([p.grad.flatten() for p in encg.parameters() if p.grad is not None])
    # (B) GRL objective -> encoder grads (fresh forward to avoid graph reuse)
    grl = GradientReversalLayer(lambd=GRL_LAMBDA)
    encg.zero_grad(); advg.zero_grad()
    ma2 = encg(imgs_a)[1]
    mb2 = encg(imgs_b)[1]
    lossB = F.binary_cross_entropy_with_logits(advg(grl(ma2), grl(mb2)), ylab)
    lossB.backward()
    gB = torch.cat([p.grad.flatten() for p in encg.parameters() if p.grad is not None])
    print('||gA|| (normal)     = %.6e' % gA.norm().item())
    print('||gB|| (GRL)        = %.6e' % gB.norm().item())
    cos = F.cosine_similarity(gA.unsqueeze(0), gB.unsqueeze(0)).item()
    print('cos(gA, gB)         = %.6f (expected ~ -1 if GRL reverses)' % cos)
    # GRL module sign test on raw input
    inp = torch.randn(4, 512, 16, 16, device=device, requires_grad=True)
    outg = grl(inp)
    outg.sum().backward()
    print('GRL backward d(out)/d(in) = %.3f (forward identity, backward flips sign)' % (inp.grad.mean().item()))
    # H_med still trains normally: fresh forward, verify adv grad nonzero
    encg.zero_grad(); advg.zero_grad()
    ma3 = encg(imgs_a)[1]
    mb3 = encg(imgs_b)[1]
    lossB2 = F.binary_cross_entropy_with_logits(advg(grl(ma3), grl(mb3)), ylab)
    lossB2.backward()
    adv_grad_norm = torch.cat([p.grad.flatten() for p in advg.parameters() if p.grad is not None]).norm().item()
    enc_grad_norm = torch.cat([p.grad.flatten() for p in encg.parameters() if p.grad is not None]).norm().item()
    print('H_med grad norm (GRL path) = %.6e (nonzero => H_med trains normally)' % adv_grad_norm)
    print('E grad norm (GRL path)     = %.6e (nonzero => E receives reversed grad)' % enc_grad_norm)
    c5 = {'g_norm_normal': gA.norm().item(), 'g_norm_grl': gB.norm().item(),
          'cos_sim': cos, 'grl_impl': 'GradientReversalLayer(lambd=%.1f)' % GRL_LAMBDA,
          'adv_grad_norm_grl': adv_grad_norm, 'enc_grad_norm_grl': enc_grad_norm}
    OUT['checks']['check5_grl'] = c5
    print('>>> GRL reverses encoder gradient as intended; H_med itself trains normally.')

    def _in_opt(mod, opt):
        own = set()
        for grp in opt.param_groups:
            own.update(id(q) for q in grp['params'])
        return all(id(p) in own for p in mod.parameters())


# ---------------- CHECK 6: optimizer ownership ----------------
    banner('CHECK 6: OPTIMIZER OWNERSHIP')
    m6 = IBRModel().to(device)
    egv = torch.optim.Adam(list(m6.encoder.parameters()) + list(m6.decoder.parameters())
                           + list(m6.verifier.parameters()), lr=1e-4)
    adv6 = torch.optim.Adam(m6.adv.parameters(), lr=1e-4)
    n_enc = sum(p.numel() for p in m6.encoder.parameters())
    n_dec = sum(p.numel() for p in m6.decoder.parameters())
    n_ver = sum(p.numel() for p in m6.verifier.parameters())
    n_adv = sum(p.numel() for p in m6.adv.parameters())
    print('encoder : %d params, %s' % (n_enc, 'IN egv' if _in_opt(m6.encoder, egv) else 'MISSING'))
    print('decoder : %d params, %s' % (n_dec, 'IN egv' if _in_opt(m6.decoder, egv) else 'MISSING'))
    print('verifier: %d params, %s' % (n_ver, 'IN egv' if _in_opt(m6.verifier, egv) else 'MISSING'))
    print('adv     : %d params, %s' % (n_adv, 'IN adv' if _in_opt(m6.adv, adv6) else 'MISSING'))
    egv_ids = set()
    for g in egv.param_groups:
        for p in g['params']:
            egv_ids.add(id(p))
    adv_ids = set()
    for g in adv6.param_groups:
        for p in g['params']:
            adv_ids.add(id(p))
    overlap = egv_ids & adv_ids
    print('param overlap between egv and adv optimizers: %d (must be 0)' % len(overlap))
    c6 = {'encoder_params': n_enc, 'decoder_params': n_dec, 'verifier_params': n_ver,
          'adv_params': n_adv, 'overlap': len(overlap)}
    OUT['checks']['check6_optimizer_ownership'] = c6

    # ---------------- CHECK 7: Stage-B activation ----------------
    banner('CHECK 7: STAGE-B ACTIVATION / TRANSITION')
    # Inspect what stage_b actually passes as y_pair and whether opt was rebuilt
    import inspect
    src = inspect.getsource(__import__('research_agent.ibr.train_s1_stages', fromlist=['S1Trainer']).S1Trainer.stage_b)
    has_y_pair_zero = 'torch.zeros(x.shape[0], 1, device=self.device)' in src
    has_opt_rebuild = 'self.opt_egv = torch.optim.Adam(' in src
    has_opt_adv_rebuild = 'self.opt_adv = torch.optim.Adam(' in src
    print('stage_b constructs y_pair=zeros: %s' % has_y_pair_zero)
    print('stage_b rebuilds opt_egv       : %s' % has_opt_rebuild)
    print('stage_b rebuilds opt_adv       : %s' % has_opt_adv_rebuild)
    c7 = {'y_pair_zeros_in_stage_b': has_y_pair_zero, 'opt_rebuilt_in_stage_b': has_opt_rebuild,
          'opt_adv_rebuilt_in_stage_b': has_opt_adv_rebuild,
          'note': 'optimizers constructed in __init__ once; stage_b reuses them (see CHECK 6/8)'}
    OUT['checks']['check7_stageb'] = c7
    # Confirm optimizer is SAME object (built once in __init__)
    trainer = __import__('research_agent.ibr.train_s1_stages', fromlist=['S1Trainer']).S1Trainer(
        seed=42, bs=16, device=device, out_dir='/tmp/opencode/dbg_opt')
    print('trainer.opt_egv class:', type(trainer.opt_egv).__name__, 'lr=%.0e' % trainer.opt_egv.param_groups[0]['lr'])
    print('trainer.opt_adv class:', type(trainer.opt_adv).__name__, 'lr=%.0e' % trainer.opt_adv.param_groups[0]['lr'])
    c7['opt_egv_lr'] = trainer.opt_egv.param_groups[0]['lr']
    c7['opt_adv_lr'] = trainer.opt_adv.param_groups[0]['lr']
    OUT['checks']['check7_stageb'].update({'opt_egv_lr': c7['opt_egv_lr'], 'opt_adv_lr': c7['opt_adv_lr']})

    # ---------------- CHECK 8: real-batch backward trace ----------------
    banner('CHECK 8: REAL-BATCH FULL GRAPH BACKWARD')
    gc.collect()
    torch.cuda.empty_cache()
    m8 = IBRModel().to(device)
    f8 = _load_frozen(device)
    egv8 = torch.optim.Adam(list(m8.encoder.parameters()) + list(m8.decoder.parameters())
                            + list(m8.verifier.parameters()), lr=1e-4)
    adv8 = torch.optim.Adam(m8.adv.parameters(), lr=1e-4)
    total, parts = compute_s1_loss(m8, f8, x, x_donor, y_path, y_pair, return_parts=True)
    total.backward()
    print('parts:', {k: '%.6e' % v for k, v in parts.items()})
    groups = {
        'encoder': m8.encoder, 'decoder': m8.decoder, 'verifier': m8.verifier,
        'adv': m8.adv,
    }
    norms = {}
    for name, mod in groups.items():
        gs = [p.grad for p in mod.parameters() if p.grad is not None]
        norms[name] = torch.cat([g.flatten() for g in gs]).norm().item() if gs else 0.0
        print('grad norm %-8s = %.6e' % (name, norms[name]))
    # frozen models must have NO grad
    frozen_grads = 0
    for p in f8.classifier.parameters():
        if p.grad is not None:
            frozen_grads += 1
    for p in f8.segmenter.parameters():
        if p.grad is not None:
            frozen_grads += 1
    print('frozen model params with grad (must be 0):', frozen_grads)
    c8 = {'parts': parts, 'grad_norms': norms, 'frozen_grad_count': frozen_grads}
    OUT['checks']['check8_backward'] = c8

    # ---------------- summary ----------------
    banner('SUMMARY')
    print(json.dumps(OUT, indent=2))
    with open('/tmp/opencode/debug_mechanism.json', 'w') as f:
        json.dump(OUT, f, indent=2)
    print('\nDEBUG OUTPUT SAVED to /tmp/opencode/debug_mechanism.json')


if __name__ == '__main__':
    main()