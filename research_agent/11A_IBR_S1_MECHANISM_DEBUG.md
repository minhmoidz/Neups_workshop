# 11A IBR S1 Mechanism Debug

**Step:** 6C-D (emergency mechanism debug)
**Date:** 2026-08-13
**Status:** FIXED — READY TO RESTART

## 1. Interrupted-run provenance

- Run: seed 42 one-seed S1 (Stage A = 20 epochs, Stage B = 30 epochs), commit `19def44bd3a5a6e190c19ccefb56ee28bd06b6c6`.
- Stage A: completed 20/20 epochs (checkpoints `s1_a_epoch01..20.pth` present).
- Stage B: reached epoch 5/30 (checkpoints `s1_b_epoch01..05.pth` present, 32,434,153 B each).
- Stop: graceful `pkill -TERM`; no corruption. Logs preserved to `/tmp/opencode/stopped_run_preserve/`.
- Logs show `train_L_zid` and `train_L_adv` collapsed to 0.0000 from epoch 2 onward and val `L_zid`/`L_adv` = 0.0 for every Stage-B epoch; `stageB_zid_verifier_auc` ~0.42-0.52 (chance) and `stageB_zmed_adversary_auc` ~0.49-0.53 (chance).

## 2. Raw vs weighted losses (CHECK 1)

- All lambdas = 1.0 (`LAMBDA_REC=LAMBDA_PATH=LAMBDA_ANAT=LAMBDA_ZID=LAMBDA_ADV=1.0`, `GRL_LAMBDA=1.0`), frozen by STEP 6A lock. Not the cause.
- Raw `L_zid = 0.686`, raw `L_adv = 0.689` at init on the real pipeline — non-zero. So zeroing is not a lambda/logging bug.
- Trainer-built `y_pair` was all zeros (`y_pair_mean = 0.0` in CHECK 8).

## 3. Pair-label audit (CHECK 2)

- TRAIN batch: 0 same-patient / 16 different-patient pairs; VAL batch: 0/16 → `positive_frac = 0.0`.
- The donor protocol guarantees donor patient != source patient, so the (source, donor) pair used for `L_zid`/`L_adv` is **always** a different-patient pair (y = 0).
- BCE on all-negative labels minimizes trivially → `L_zid`, `L_adv` → 0 and heads collapse to chance. This is the root cause.
- Pre-existing balanced pair files are fold-correct and contain both classes (5000/5000 train, 1000/1000 val, 0 bad labels, train/val patient-disjoint) but were not used in Stage B.
- Data statistics: 42.2% (train) / 43.5% (val) of patients have >=2 images; median patient has 1 image; same-patient partners therefore require a per-image partner mechanism, not within-batch pairing.

## 4. z_id verifier audit (CHECK 3)

- Verifier head can overfit perfect features: train acc 1.0000.
- Verifier head can overfit real untrained z_id: acc 0.875.
- Head is functional; failure is the data, not the head.

## 5. z_med adversary audit (CHECK 4)

- Adversary head can overfit perfect features: acc 1.0000.
- Adversary head can overfit real z_med: acc 0.875.
- Head is functional; failure is the data, not the head.

## 6. GRL audit (CHECK 5)

- `cos(g_normal, g_grl) = -1.0000` on real data → GRL reverses encoder gradient as designed.
- `H_med` gradient nonzero; encoder receives reversed gradient. Mechanism intact.

## 7. Optimizer ownership (CHECK 6)

- No parameter overlap: encoder 4,779,296 / decoder 3,114,881 / verifier 33,025 / adversary 164,481 params.
- Optimizers built once in `__init__` and reused in Stage B (not rebuilt → not a stale-optimizer bug).
- CHECK 8: all grad norms nonzero (encoder 6.80, decoder 5.50, verifier 0.90, adv 1.01), frozen models 0 grads, heads update after step.

## 8. Root cause

Stage B hardcoded `y_pair = torch.zeros(x.shape[0], 1, device=self.device)`. Because donor != source by protocol, the identity pair always had label y=0, so both BCE terms saw only negatives and collapsed. Line 118 of `10_IBR_S1_IMPLEMENTATION.md` documented this as a "known limitation"; in practice it silently destroyed the mechanism.

## 9. Exact minimal fix

- `s1_loss.py` `compute_s1_loss` now takes `x_pair, y_pair` and computes `L_zid`/`L_adv` from the (source, partner) pair instead of the (source, donor) pair. Donor is still used for the anonymized branch (`x_anon = decode(z_id, z_med_donor, skips)`).
- `train_s1_stages.py` `SingleImageLabels.__getitem__` returns a 7-tuple `(x, y, name, x_donor, donor, x_partner, y_pair)`; `_partner_for(name)` picks a same-patient image with p=0.5 when the patient has >=2 images in the split (y=1), else the different-patient donor (y=0). Deterministic (sha256 key + RNG seed).
- Stage B loop no longer builds zeros; it passes the real `x_partner`/`y_pair`. `_validation_diagnostics` updated to stack `v[5]`/`v[6]` and pass them to `compute_s1_loss`.
- Legacy scripts `train_s1.py` and `dry_run_s1.py` updated to the new signature (synthetic batch now mixes both classes).
- NOT changed: lambdas, architecture, z_id dim, donor mechanism, batch size, lr, no V2.

## 10. Post-fix micro-validation

- Real train batch: `y_pair` positive=6, negative=10 (both classes present).
- Raw losses on real batch: L_zid = 6.971e-01, L_adv = 6.875e-01 (finite, > 0). All 5 terms finite, total 2.288.
- Gradients flow to encoder (6.78), decoder (5.50), verifier (0.265), adv (0.264); frozen grads = 0; verifier/adv params move after step.
- GRL on real data: `cos = -1.000000` (sign flip works).
- Verifier tiny balanced-batch overfit: loss 0.700 -> 0.005, acc 1.0000.
- Adversary tiny balanced-batch overfit (no GRL): loss 0.693 -> 0.000, acc 1.0000.
- TEST untouched (no TEST code path executed).

## 11. Restart recommendation

- Restart cleanly: Stage A = 20 epochs (resume or fresh; `s1_a_epoch20.pth` usable), then Stage B = 30 epochs, seed 42, bs 16, all lambdas 1.0, GRL lambda 1.0, z_id_dim 128, Adam lr 1e-4.
- Launch with `setsid` + `PYTHONPATH=.` (nohup alone is killed with the shell process group).
- Success criteria for Stage B: `train_L_zid`/`train_L_adv` remain > 0 and verifier/adversary AUCs move away from 0.5 (verifier AUC high for same-patient z_id; adversary AUC stays near 0.5 due to GRL suppression).
- **Not auto-launched.** Awaiting authorization.

---

IBR S1 MECHANISM DEBUG: FIXED — READY TO RESTART