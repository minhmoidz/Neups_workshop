"""Pre-flight validation for run_corrected_objective.py.

Run this BEFORE committing GPU-hours. Part 1 is pure numerics (seconds, CPU) and
checks the objective itself. Part 2 constructs the real runner and pushes a
handful of real batches through the real training step, which is what catches
API drift, dtype problems and fail-closed checks that fire spuriously.

    .venv/bin/python reproduction/method_dev/test_corrected_objective.py
    .venv/bin/python reproduction/method_dev/test_corrected_objective.py --live 3
"""
import argparse
import os
import sys

import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_corrected_objective import (  # noqa: E402
    CorrectedObjectiveRunner, SURROGATE_TAU, METHOD_OUT_ROOT, TRAIN_PAIR_FILE,
)

PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name,
                           ('  -- ' + detail) if detail else ''))


def part1_objective():
    print('\nPART 1 -- the objective (Addendum B contract)')
    S = CorrectedObjectiveRunner.auc_surrogate
    g = torch.Generator().manual_seed(0)
    y = torch.cat([torch.ones(400), torch.zeros(400)])
    base = torch.cat([torch.randn(400, generator=g) + 1.5,
                      torch.randn(400, generator=g) - 1.5])

    # (1) scale invariance -- closes the shrink loophole of Addendum B §1.1
    vals = [float(S(base * sc, y)) for sc in (5.0, 1.0, 0.1, 0.03)]
    check('scale-invariant across 170x logit rescaling', max(vals) - min(vals) < 1e-3,
          'spread %.2e' % (max(vals) - min(vals)))

    # (2) accuracy -- tracks true AUC, so targeting 0.5 targets chance
    worst = 0.0
    for sep in (2.0, 1.0, 0.5, 0.2):
        gg = torch.Generator().manual_seed(7)
        z = torch.cat([torch.randn(400, generator=gg) + sep,
                       torch.randn(400, generator=gg) - sep])
        worst = max(worst, abs(float(S(z, y)) - roc_auc_score(y.numpy(), z.numpy())))
    check('tracks true AUC within 0.01', worst < 0.01, 'max deviation %.4f' % worst)

    # (3) optimizing it actually drives TRUE AUC to chance, without overshoot
    z = base.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        ((S(z, y) - 0.5) ** 2).backward()
        opt.step()
    final_auc = roc_auc_score(y.numpy(), z.detach().numpy())
    check('optimization lands at chance AUC (0.45-0.55)', 0.45 <= final_auc <= 0.55,
          'true AUC %.4f' % final_auc)

    # (4) degenerate-batch guard (prereg §2)
    one_class = torch.ones(8)
    check('returns None when a class is absent',
          S(torch.randn(8), one_class) is None and S(torch.randn(8), 1 - one_class) is None)

    # (5) differentiable
    z = base.clone().requires_grad_(True)
    ((S(z, y) - 0.5) ** 2).backward()
    check('produces finite non-zero gradient',
          z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0)

    # (6) target is a stationary point -- it must not push past chance
    gg = torch.Generator().manual_seed(3)
    z = torch.randn(800, generator=gg).requires_grad_(True)   # already ~chance
    ((S(z, y) - 0.5) ** 2).backward()
    check('gradient ~0 when already at chance', float(z.grad.abs().max()) < 1e-3,
          'max|grad| %.2e' % float(z.grad.abs().max()))

    print('  (tau = %.2f)' % SURROGATE_TAU)


def part2_live(n_batches):
    print('\nPART 2 -- live runner on %d real batches' % n_batches)
    if not torch.cuda.is_available():
        print('  SKIPPED: no CUDA')
        return
    device = torch.device('cuda')
    out_dir = os.path.join(METHOD_OUT_ROOT, 'corrected_objective_SMOKE', 'B_dev', 'seed_42')

    try:
        r = CorrectedObjectiveRunner(42, out_dir, device)
        check('runner constructs (all SHA + frozen-invariant assertions pass)', True)
    except Exception as e:
        check('runner constructs', False, repr(e))
        return

    # Truncate the loaders in place; the real train_epoch/validate_epoch code
    # then runs unmodified over a few real batches.
    train_batches = []
    for i, b in enumerate(r.training_loader):
        train_batches.append(b)
        if i + 1 >= n_batches:
            break

    # The VAL pair file is sorted by class (1000 positives then 1000 negatives)
    # and the VAL loader is sequential, so the FIRST k batches are all
    # positives. Take from both ends, otherwise the smoke test exercises a
    # single-class fold that cannot produce an AUC at all.
    all_val = list(r.validation_loader)
    half = max(1, n_batches // 2)
    val_batches = all_val[:half] + all_val[-(n_batches - half):]

    r.training_loader = train_batches
    r.validation_loader = val_batches

    try:
        tm = r.train_epoch(0)
        check('train_epoch runs (incl. grad/param finiteness checks)', True,
              's=%.4f priv=%.5f ac=%.4f' % (tm['train_s_surrogate'],
                                            tm['train_priv_new'], tm['train_ac_bce']))
    except Exception as e:
        check('train_epoch runs', False, repr(e))
        return

    try:
        vm = r.validate_epoch(0)
        check('validate_epoch runs', True,
              's_pooled=%.4f coadapted_auc=%.4f' % (vm['val_s_pooled'],
                                                    vm['val_coadapted_true_auc_DIAGNOSTIC']))
    except Exception as e:
        check('validate_epoch runs', False, repr(e))
        return

    check('all reported metrics finite',
          all(torch.isfinite(torch.tensor(float(v))).item()
              for v in list(tm.values()) + list(vm.values())))
    check('degenerate-batch rate within prereg limit',
          r.degenerate_batches / max(r.total_train_batches, 1) <= 0.05,
          '%d/%d' % (r.degenerate_batches, r.total_train_batches))

    # prereg §4.6 -- the verifier must be persisted alongside the generator
    torch.save(r.generator.state_dict(), r.best_checkpoint_path)
    torch.save(r.verification_loss.verification_model.state_dict(), r.best_verifier_path)
    check('generator + co-adapted verifier both saved',
          os.path.exists(r.best_checkpoint_path) and os.path.exists(r.best_verifier_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', type=int, default=0,
                    help='number of real batches for the live smoke (0 = skip)')
    args = ap.parse_args()

    part1_objective()
    if args.live > 0:
        part2_live(args.live)

    print('\n%d passed, %d failed' % (len(PASS), len(FAIL)))
    if FAIL:
        print('FAILED: %s' % ', '.join(FAIL))
        raise SystemExit(1)
    print('Pre-flight OK.')


if __name__ == '__main__':
    main()
