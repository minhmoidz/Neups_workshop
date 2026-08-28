# PHASE B PREREGISTRATION — SEPARABILITY-TARGETING PRIVACY OBJECTIVE (LOCKED BEFORE EXECUTION)

**Date/time locked:** 2026-08-28.
**Supersedes:** the candidate track of
`V2_CONFIRMATION_PREREGISTERED_HYPOTHESES_2026-08-26.md`, abandoned before
execution per `V2_CONFIRMATION_ADDENDUM_A_2026-08-28.md` §0.
**This file must not be edited after unblinding; corrections go into a dated
addendum.**

## LOCK CONDITIONS (verified 2026-08-28, before writing this file)

- **No candidate artifact of any kind exists.** A repo-wide search for
  `*corrected_objective*`, `*phase_b*`, `*auc_surrogate*` returns nothing. The
  runner named in §4 has not been written. **Zero candidate outcome values have
  been observed, and none can have been.**
- The V2 nested pools of that prereg's §2 were never built; that protocol never
  ran.
- Everything the author already knows is disclosed in §9. Nothing in §§1–8 was
  chosen after seeing a candidate number, because no candidate number exists.

---

## 0. HYPOTHESIS

**H-B.** PriCheXy-Net's privacy objective is invariant to the ROC AUC used to
report privacy. Replacing it with an objective that targets *separability*
will reduce adaptive-attacker Re-ID AUC relative to the certified B_dev
generator, at non-inferior classification utility.

This is a **mechanism experiment on a single generator**, not a powered
population claim. Generator-seed variance is not estimated here (§10).

## 1. THE DIAGNOSIS THIS RESTS ON (already sealed, not re-litigated here)

`reproduction/method_dev/privacy_objective_diagnosis/privacy_objective_diagnosis.json`
(schema `PRIVACY_OBJECTIVE_DIAGNOSIS_V1`, commit `09046f7`), reproducible via
`reproduction/method_dev/diagnose_privacy_objective.py`:

- **D1.** The objective `L_priv = -log(1 - sigmoid(z)) = softplus(z)` is exactly
  AUC-invariant: a uniform logit shift drives it 1.0583 -> 0.000000 with ROC AUC
  constant to 12 decimals. Source files SHA-256'd; `utils/VerificationLoss.py`
  is byte-identical to upstream `29245d1`.
- **D4.** Across four real trained checkpoints, logged `ver_loss` varies 9x
  (0.034 -> 0.309) while true re-identification AUC stays pinned in
  0.892–0.915.
- **D5.** The co-adapted training verifier reaches AUC 0.9147 where fresh
  adaptive attackers reach 0.7258 (same generator bytes `2f285743`):
  `weak_critic_hypothesis_supported: false`.
- **D3.** D_BDEV deforms as hard as U_PUBLISHED (mean|flow| 0.8977 vs 0.8752)
  yet leaks identity — the generator was never asked to remove identity.
- **D2.** Train/val patient overlap 0; VAL pool exactly 1000/1000.

**D5 is why this intervention has a favourable prior:** the co-adapted critic is
the *strongest* re-identifier measured, so an objective that genuinely defeats
it is defeating a harder adversary than the one used for scoring.

## 2. THE INTERVENTION (exactly one variable changes)

For a training batch with verifier logits `z_i` and true identity labels
`y_i in {0,1}`, let `P = {i : y_i = 1}`, `N = {j : y_j = 0}`.

**Differentiable AUC surrogate**

```
s = (1 / (|P| * |N|)) * sum_{i in P} sum_{j in N} sigmoid(z_i - z_j)
```

**Privacy loss (replaces softplus)**

```
L_priv_new = (s - 0.5)^2
```

**Generator objective** (weights unchanged from the certified config):

```
L_gen = ac_loss_weight * L_AC_BCE + ver_loss_weight * L_priv_new
      = 1.0 * L_AC_BCE + 1.0 * (s - 0.5)^2
```

**Rationale, locked now.** `s` estimates `P(z_pos > z_neg)`, which *is* ROC AUC.
Targeting `0.5` targets chance-level re-identification. The target is a
stationary point of the squared penalty, so the objective does not push past
chance into anti-correlation — an AUC of 0.0 is as re-identifying as 1.0 and is
explicitly not sought.

**Degenerate-batch guard.** If `|P| = 0` or `|N| = 0`, set `L_priv_new := 0` for
that batch and increment a counter reported in the manifest (see §8 validity).

**The verifier critic update is UNCHANGED** — still `BCEWithLogitsLoss` against
true identity labels, same position and order in the step. The adversary stays
honest; only what the generator asks of it changes.

## 3. WHAT STAYS FROZEN (single-variable discipline)

Identical to `config_dev_restored_baseline.json` (SHA `14d3943f…`), the frozen
B_dev config: `mu = 0.01`, `image_size = 256`, `batch_size = 16`,
`accumulation_steps = 1`, Adam(g) lr 1e-4, Adam(ver) lr 1e-4,
SGD(ac) lr 1e-4 momentum 0.9 wd 1e-4, `max_epochs = 250`,
`ac_loss_weight = 1.0`, `ver_loss_weight = 1.0`, `feature_loss_weight = 0.0`,
seed 42, init generator `10122689…`, classifier and verifier init as frozen,
upstream pair files (`3c535eed…` train / `9e33a081…` val).

**`batch_size` stays 16, deliberately.** Upstream's published anonymization
config uses 64; that discrepancy is a separate, known question and is
**explicitly out of scope here**, because the comparison anchor is D_BDEV,
which is a batch-16 artifact. Changing two variables would make the result
uninterpretable. A batch-size ablation requires its own preregistration.

**Loss-weight tuning is NOT part of this experiment.** `ver_loss_weight = 1.0`
is inherited, not chosen. Any future weight search requires a new
preregistration and multiplicity control.

## 4. RUNNER AND INTEGRITY GATES

New script `reproduction/method_dev/run_corrected_objective.py`, modelled on
`run_hardened_verifier.py` (itself a line-for-line copy of the certified
`M2AnonymizerRunner` minus the S1-specific scientific locks). It **must**
restore the guards `STRATEGY_REVIEW_2026-08-21.md` §3.5 found dropped:

1. Frozen-config SHA assertion before use; initial-generator, classifier and
   verifier SHA assertions.
2. Per-epoch `order_sha256` logged **and compared against an expected schedule**
   (not merely logged).
3. NaN/Inf fail-closed checks on every load-bearing loss, and on gradients and
   post-step parameters.
4. `torch.backends.cudnn.deterministic = True` and `benchmark = False`
   (the V2 path set only the latter).
5. Outputs under `reproduction/method_dev/`, labelled `method_uncertified: true`,
   never under `research_runs/M2_S1/`.
6. **Persist the co-adapted verifier** at both the selected epoch and epoch 250
   (`ver_model_trained_lowest_total_loss.pth`, `ver_model_trained_latest.pth`).
   The certified B_dev run did not, which is why diagnosis D4/D5 could only be
   measured on V2 arms. This closes that gap on a certified-recipe arm.
7. Log `s` (train and val) per epoch, plus the degenerate-batch counter.

**TEST firewall CLOSED.** No TEST loader is constructed at any point.

## 5. CHECKPOINT SELECTION (and its declared limitation)

Selection rule, structurally identical to the certified
`lowest_validation_total_loss_method_neutral`, with each method using its own
privacy term (the same convention C4 used):

```
val_selection_total = val_ac_bce + val_L_priv_new
```

argmin over epochs 0–249; tie-break **earliest epoch**. This is the PRIMARY
checkpoint.

**Declared limitation, not hidden:** `val_L_priv_new` and D_BDEV's
`val_privacy_term` are on different scales, so selection is *structurally*
parallel but not *numerically* method-independent —
`STRATEGY_REVIEW_2026-08-21.md` §3.6 is right that this is imperfect. Building a
truly method-independent external selector requires an inner attacker on
TRAIN-derived pools disjoint from the evaluation pairs, which does not exist
yet. Recorded as a limitation of this experiment.

**Secondary, declared now:** the epoch-250 checkpoint is also evaluated, and
reported alongside. **The primary classification in §7 uses the selected
checkpoint only.** Reporting the better of the two is forbidden.

## 6. EVALUATION HARNESS, ANCHOR, SEEDS

- **Harness:** the governed P0 bridge, unchanged
  (`review/p0-runner-attacker-loop-20260823`): Siamese ResNet-50, Adam lr 1e-4,
  batch 32, max 100 epochs, patience 5; train/selection geometry anon/anon;
  scoring geometry anon(x1)/real(x2) on the 2000 VAL pairs; raw ROC AUC with
  orientation fixed a priori. Post-hoc flipping and "effective AUC" remain
  forbidden.
- **Anchor:** D_BDEV (`18381d92…`) as measured by the same harness,
  `runs_screen/D_BDEV/` — **paired by attacker seed**, not compared to its mean.
- **Seeds: exactly 8, master seeds 42–49**, for which D_BDEV values already
  exist. No seed may be added, replaced or rerun after unblinding.
- **Control is NOT re-run.** Verified: the five code defects of Addendum A §6 do
  not touch `research_agent/m2_dev/anonymizer_runner.py` (defects 1–2 are
  confined to `UNetAtt`; 4 needs `accumulation_steps > 1`, which the certified
  config does not use). D_BDEV therefore remains valid as executed, saving
  ~32 GPU-h.

**Power, as a planning statement.** D_BDEV attacker-seed SD is 0.031; Stage-A
measured arm-to-arm seed correlation rho ~ 0.17, so paired-difference SD is
planned at ~0.043, giving SE ~ 0.015 at n=8 and a smallest reliably-detectable
Δ of about −0.029 — deliberately matched to the frozen δ = 0.03. If the true
effect is the ~−0.10 the diagnosis suggests, n=8 is ample. If the true effect
is marginal, this design will **not** confirm it, and that is accepted now
rather than fixed later by adding seeds.

## 7. PRE-REGISTERED DECISION RULES (mutually exclusive)

Let Δ = mean over seeds of `AUC(candidate, seed) − AUC(D_BDEV, seed)`, paired,
n = 8. CI = one-sided 95% upper bound on the paired differences (bootstrap over
seeds, 10000 resamples, **fixed seed 12345**). δ = 0.03, frozen since the P0
review §5.4.

| Rule | Classification |
|---|---|
| Δ ≤ −0.03 **AND** CI_upper < 0 **AND** utility gate PASS | **H-B-SUPERIOR** — the corrected objective reduces adaptive re-identification under this harness |
| \|Δ\| < 0.03 (or CI spans it) **AND** utility gate PASS | **H-B-EQUIVALENT** |
| Δ > +0.03 **OR** utility gate FAIL | **H-B-NOT-SUPPORTED** |
| Futility stop triggered (§8) | **H-B-MECHANISM-FAILED** |

`H-B-MECHANISM-FAILED` is deliberately distinct: it means the objective failed
to move *its own target* `s`, which is a different and separately informative
outcome from moving `s` without moving the real AUC.

## 8. UTILITY GATE AND VALIDITY RULES

**Utility gate (PRIMARY, declared before any measurement):** macro-AUC over the
14 pathologies, governed evaluator, fold=val, must satisfy

```
macro_AUC(candidate) >= 0.7840 - 0.03 = 0.7540
```

where 0.7840 is D_BDEV's certified classification VAL macro-AUC.

*Justification, recorded before seeing results:* a non-inferiority margin is
used rather than the zero-degradation gate of the V2 prereg because this
intervention **changes the objective itself**, and a privacy method is expected
to trade some utility; a zero-tolerance gate would make any privacy result
unreportable regardless of magnitude. The margin equals the project's already
frozen δ = 0.03 — no new threshold is invented. The exact macro-AUC is reported
either way, together with whether it *also* meets zero-degradation (≥ 0.7840),
so a reader can apply the stricter gate themselves.

**Validity rules (any failure invalidates the run; no substitution, no retune):**

1. Any NaN/Inf in a load-bearing loss, gradient or post-step parameter -> abort.
2. Degenerate batches (`|P| = 0` or `|N| = 0`) exceeding **5%** of training
   batches -> run invalid.
3. Any `order_sha256` mismatch against the expected schedule -> run invalid.
4. Any SHA assertion failure on config/checkpoints -> abort.

**Futility stop (declared to protect GPU budget, not to permit tuning):** if at
epoch 50 the running minimum of `|s_val − 0.5|` has not improved by at least
0.05 relative to its epoch-0 value, stop and classify **H-B-MECHANISM-FAILED**.
A futility stop **does not** authorize changing the objective, its weight, or
any hyperparameter and re-running; that would require a new preregistration.

## 9. FULL DISCLOSURE OF PRIOR KNOWLEDGE

The author of this file has already seen, and lists here so the design cannot
be claimed to have been fitted to them: every number in Addendum A §5 (V2_UINIT
0.7258, V2_UINIT_VER 0.6969, U_PUBLISHED 0.6985/0.7139, D_BDEV 0.8244, I_M2
0.8366) and the entire diagnosis JSON of §1, including D_BDEV's per-seed P0
values for seeds 42–67.

**Consequence, stated plainly:** the anchor side of the §7 comparison is already
known to the author at every seed. The candidate side is not, and cannot be —
it does not exist. This design is therefore anti-anchored where it matters (the
unknown arm), and the known-anchor asymmetry is disclosed rather than papered
over. No untouched attacker-seed pool remains for D_BDEV under the P0 harness;
the P0.3 §3 "never inspected" 47–67 partition has since been unblinded, and
seeds 42–49 are chosen for pairing convenience with that fact acknowledged.

## 10. LIMITATIONS ACKNOWLEDGED NOW

- **One generator seed (42).** Attacker seeds are nested replicates, not method
  replicates. Any claim is scoped to "this generator state". Generator-seed
  variance remains future work (~32 GPU-h/seed).
- **One attacker family** (Siamese ResNet-50). A second, training-free family
  (SSIM / pixel-kNN retrieval) is planned but is **not** part of this
  preregistration and cannot be used to rescue a failed classification.
- Selection is structurally but not numerically method-independent (§5).
- Patient-clustered bootstrap CIs remain UNVALIDATED (P0 review §9); seed-level
  intervals only.
- Segmentation utility: BLOCKED, excluded from all claims.
- Shared-VAL harness: selection and scoring draw on the same 2000 pairs
  (`attacker_loop.py` uses `self.val_loader` for both). Known and disclosed; it
  applies identically to both arms, so it largely cancels in the paired Δ.

## 11. ANTI-ANCHORING COMMITMENTS

No additional seeds. No objective variants, weight changes or schedule changes
after seeing any result. No threshold edits. No post-hoc substitution of the
epoch-250 checkpoint for the selected one. Classification made **exactly once**
from the sealed summary JSON. This file immutable after unblinding; corrections
by dated addendum only.

## 12. COST AND CONSEQUENCE MAP

Planning estimate: generator ~14 GPU-h (250 epochs at the observed batch-16
pace); 8 adaptive attackers ~6.4 GPU-h; utility evaluation ~0.5 GPU-h.
**Total ~21 GPU-h.**

| Classification | Immediate action |
|---|---|
| **H-B-SUPERIOR** | Freeze candidate + manifests. Manuscript = diagnosis (§1) + corrected objective + this confirmation. Then, and only then, spend budget on generator-seed replication and the second attacker family. |
| **H-B-EQUIVALENT** | Manuscript = diagnosis only; candidate reported as an honest null. The diagnosis stands on its own and does not depend on this outcome. |
| **H-B-NOT-SUPPORTED** | Same as EQUIVALENT, plus a loss-curve/`s`-trajectory forensics appendix: the objective moved its own target but not the real AUC, which would itself be a substantive finding about surrogate-vs-metric gaps. |
| **H-B-MECHANISM-FAILED** | Same as EQUIVALENT, plus report why `s` did not move (weight scale, gradient magnitude). Explicitly **not** a licence to retune within this preregistration. |

**The manuscript does not depend on this experiment succeeding.** The §1
diagnosis is already complete, sealed and reproducible at zero GPU cost. Phase B
is upside.

*End of preregistration.*
