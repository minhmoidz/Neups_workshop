# PHASE B2 PREREGISTRATION — SEPARABILITY-TARGETING OBJECTIVE, UPSTREAM RECIPE (LOCKED BEFORE EXECUTION)

**Date/time locked:** 2026-08-29.
**Successor to:** Phase B, classified `H-B-MECHANISM-FAILED` and closed
(`PHASE_B_ADDENDUM_C_2026-08-28.md`). That preregistration forbids retuning and
re-running within itself; this is the separate document it requires.
**This file must not be edited after unblinding; corrections go into a dated addendum.**

## LOCK CONDITIONS (verified before writing)

- No Phase B2 artifact of any kind exists. The runner does not yet accept the
  configuration below. **Zero candidate outcome values have been observed.**
- Everything the author already knows is disclosed in §10. In particular the
  anchor's per-seed values are known; the candidate's cannot be.

---

## 0. HYPOTHESIS

**H-B2.** PriCheXy-Net's privacy objective is invariant to the ROC AUC it
reports, and consequently spends its deformation budget without regard to
identity. Replacing it with an objective that targets *separability*, while
holding the published training recipe otherwise fixed, will reduce
adaptive-attacker Re-ID AUC below the released checkpoint's, at non-inferior
classification utility.

## 1. WHAT PHASE B ESTABLISHED, AND WHAT KILLED IT

Phase B used `L = (s − 0.5)²` with `s` a temperature-0.1 standardized pairwise
sigmoid surrogate. It produced no movement in six epochs. Cause, measured:
gradient saturation. At the near-perfect-separation state the generator is
initialized into, `‖dL/dz‖ = 1.76e-06` — five orders of magnitude below its
value at moderate separation.

Measured comparison of candidate replacements at that same initial regime
(batch 16, AUC ≈ 0.995), together with where each drives TRUE AUC when
optimizing free logits from a start of 1.0000:

| objective | ‖dL/dz‖ at init | vs original `softplus(z)` | TRUE AUC reached |
|---|---:|---:|---:|
| original `softplus(z)` | 3.55e-02 | 1.0x | — |
| **`gap²` (standardized)** | **3.38e-01** | **9.5x** | **0.5400** |
| surrogate `tau=1.0` | 1.29e-02 | 0.4x | 0.5459 |
| `gap²` + surrogate `tau=0.3` | 3.44e-01 | 9.7x | 0.5429 |

The `tau`-annealing repair sketched in Addendum C §5 is **rejected on this
evidence**: at `tau=1.0` the surrogate is *weaker than the objective it
replaces*, and would very likely have futility-stopped exactly as Phase B did.
Recording this because it was the plan of record until measured.

## 2. THE INTERVENTION

For a batch with verifier logits `z` and identity labels `y`, `P = {i: y_i=1}`,
`N = {j: y_j=0}`:

```
zhat = (z - mean(z)) / (std(z) + 1e-6)
L_priv = ( mean_{i in P}(zhat_i) - mean_{j in N}(zhat_j) )^2
```

Generator objective: `L_gen = ac_loss_weight * L_AC_BCE + ver_loss_weight * L_priv`.

**Why this form, fixed now:**
- **Standardized**, so it is invariant to any affine rescaling of `z` — the
  generator cannot satisfy it by shrinking the logit scale, which is the
  calibration-not-separability failure the whole project indicts.
- **Non-saturating**, which is the specific defect that killed Phase B.
- **Zero hyperparameters.** No temperature, no mixing weight. `gap² + surrogate`
  measured 0.2x better on gradient and is not taken: two extra hyperparameters
  for no measured benefit is exactly the tuning freedom this project forbids.
- Its fixed point (equal class means) is a necessary condition for AUC = 0.5,
  and empirically it reaches 0.5400 from 1.0000 — the theoretical concern that
  equal means can coexist with separability does not materialize.

**Degenerate-batch guard:** if `|P| = 0` or `|N| = 0`, `L_priv := 0` for that
batch and a counter is incremented. **Validation `L_priv` is computed from
POOLED VAL logits, never per-batch** — the VAL pair file is sorted by class and
the VAL loader is sequential, so exactly 1 of 125 batch-16 windows contains both
classes (Phase B Addendum B §3b).

**The verifier critic update is UNCHANGED**: `BCEWithLogitsLoss` against true
identity labels, same position and order in the step.

## 3. RECIPE — AND THE DELIBERATE RETURN TO UPSTREAM

A fidelity audit of this project's training path against upstream commit
`29245d1` found the anonymization operator, ACLoss semantics, update order,
`zero_grad` placement, critic mode cycling, both preprocessing paths, all three
optimizers, the pair files and the checkpoint-selection rule to be faithful.
**One substantive deviation exists: `batch_size` 64 (paper) versus 16 (this
project's frozen config).**

Phase B2 therefore trains at **effective batch 64**, matching the paper, so that
a win cannot be attributed to a recipe deviation.

| parameter | value | status |
|---|---|---|
| `batch_size` (micro) | 16 | frozen config |
| **`accumulation_steps`** | **4** | **declared override** (frozen config says 1) |
| effective batch | **64** | matches the paper |
| **`ver_loss_weight`** | **1.0 (arm A) / 3.0 (arm B)** | **declared override** (frozen says 1.0) |
| `mu` | 0.01 | frozen, unchanged |
| `ac_loss_weight` | 1.0 | frozen |
| `learning_rate` | 1e-4 | frozen |
| `image_size` | 256 | frozen |
| initial generator | **`4d82dcdd…` (U_PUBLISHED)** | **declared override** (frozen config inits from `10122689…`) |

Exactly three frozen values are overridden, each named above with its reason.
Nothing else is touched, and no further override may be added after execution.

**Why initialize from U_PUBLISHED.** It makes the anchor and the initialization
the same object, so the primary comparison directly answers *"does continued
training under the corrected objective improve the released model?"* with no
initialization confound. Partial evidence that continued training does **not**
automatically improve it already exists: the same initialization trained under
the ORIGINAL objective moved to 0.7258, i.e. **+0.0906 worse** (V2_UINIT, n=3).

**Epoch budget: 50.** Justified by observed selection epochs — B_dev 13, C4 8,
hardened-verifier 15 — and by the selection statistic rising monotonically
after ~epoch 20 in every run inspected. **If the argmin lands at epoch ≥ 45 the
run is flagged `SELECTION_POSSIBLY_TRUNCATED` and reported as such**; the
classification still stands but the caveat is mandatory.

## 4. INTEGRITY GATES

All seven of Phase B §4 are retained and were confirmed working there (6/6 order
hashes verified, 0 degenerate batches, all fail-closed checks live):
SHA assertions; per-epoch order hash **compared** against
`compute_epoch_order_hash`; NaN/Inf checks on losses, gradients and post-step
parameters; `cudnn.deterministic=True`; outputs under `reproduction/method_dev/`
labelled `method_uncertified`; **the co-adapted verifier persisted at the
selected and final epoch**; per-epoch logging of `gap`, pooled `s_val`, the
degenerate-batch counter and the diagnostic co-adapted true AUC.

**Additional gate for the accumulation path.** Phase B's audit found that the
AC critic was left in `train()` across micro-batches when `accumulation_steps>1`
(commit `10d9212`, defect 4). The runner MUST assert, every micro-batch, that
`ac_model.training is False` at the point the generator loss is computed.

**TEST firewall CLOSED** for all of §§5–8. §9 is the single exception and has
its own gate.

## 5. SELECTION BETWEEN ARMS — ON UTILITY, NEVER ON PRIVACY

Two arms are run: **A** (`ver_loss_weight=1.0`) and **B** (`ver_loss_weight=3.0`).
Both are declared now; arm B is **not** a reaction to arm A's result.

> **The primary arm is whichever arm's classification macro-AUC is closest to
> 0.7730 from ABOVE.** If both are below 0.7730, neither is primary and the
> classification is `H-B2-UTILITY-FAIL`.

0.7730 is U_PUBLISHED's macro-AUC under the governed evaluator — the released
model's own operating point. This rule matches the operating point on utility
and then compares privacy, so **no degree of freedom touches the reported
privacy metric**. The non-primary arm is reported in full alongside.

## 6. EVALUATION, ANCHOR AND SEEDS

- **Harness:** the governed P0 bridge, protocol `P0_PROTOCOL_V1_2`, unchanged.
  Raw ROC AUC, orientation fixed a priori; post-hoc flipping and "effective AUC"
  remain forbidden.
- **Anchor:** `U_PUBLISHED` (`4d82dcdd…`), **paired by attacker seed** against
  its existing sealed manifests — never against its mean.
- **Stage 1 (this preregistration): the protocol's `screen` seed list,
  42–46, n=5.** On those seeds the anchor averages **0.6336** (SD 0.0551). The
  subset matters: the anchor ranges 0.6336 (42–46) to 0.7139 (47–67) depending
  on seeds. Pairing removes that as a source of bias in Δ, but it is disclosed
  because it changes the absolute numbers reported.
- **Stage 2 (declared conditional):** if Stage 1 classifies `H-B2-SUPERIOR`,
  the primary arm — and only that arm — is rerun on the protocol's `full` seed
  list, 42–67, n=26. No other arm and no other seed set may be added.

**Power, stated as a planning claim.** Paired-difference SD is planned at
~0.043; at n=5 that is SE ≈ 0.019 and a smallest reliably-detectable effect of
about **−0.041**. Stage 1 is therefore explicitly a **screen**: it can confirm a
large effect and cannot confirm a marginal one. Accepted now rather than
repaired later by adding seeds.

## 7. DECISION RULES (mutually exclusive, applied once)

Δ = mean over seeds of `AUC(candidate, seed) − AUC(U_PUBLISHED, seed)`, paired,
n = 5. CI = one-sided 95% upper bound on the paired differences (bootstrap over
seeds, 10000 resamples, **fixed seed 12345**). δ = 0.03, frozen.

| Rule | Classification |
|---|---|
| Δ ≤ −0.03 AND CI_upper < 0 AND utility gate PASS | **H-B2-SUPERIOR** |
| \|Δ\| < 0.03 (or CI spans it) AND utility gate PASS | **H-B2-EQUIVALENT** |
| Δ > +0.03 OR utility gate FAIL | **H-B2-NOT-SUPPORTED** |
| both arms below 0.7730 utility | **H-B2-UTILITY-FAIL** |
| `gap` fails to move (see below) | **H-B2-MECHANISM-FAILED** |

**Futility stop.** If at epoch 25 the running minimum of the pooled VAL `gap`
has not fallen by at least 25% from its epoch-0 value, stop and classify
`H-B2-MECHANISM-FAILED`. As in Phase B, a futility stop **does not** authorize
retuning inside this preregistration.

## 8. UTILITY GATE

Classification macro-AUC, 14 pathologies, governed evaluator, fold=val:
**candidate ≥ 0.7730** (U_PUBLISHED's own value). This is zero-degradation
against the released model, not a margin — the §5 selection rule already targets
that operating point, so no tolerance is needed or taken.

The measured budget for context, not as a gate: the released model spends
**4.28 points** of utility (81.58 original → 77.30 at μ=0.01). A candidate at
matched utility is spending the same budget the published method already spent.

## 9. TEST CONFIRMATION — ONE SHOT, CONDITIONAL

Permitted **only** if Stage 2 classifies `H-B2-SUPERIOR`, and executed **exactly
once**: the primary arm evaluated under the **upstream protocol** (retrained SNN,
TEST fold, 10 runs), the same procedure that produced this project's
`60.80 ± 4.35` for the released checkpoint — which is the paired control and is
**not** rerun.

**Mandatory dual reporting.** Both numbers are reported together, always:
the adaptive-harness result and the upstream-protocol result. Reporting only the
more favourable protocol is the precise practice this project's own findings
indict, and is forbidden here. Likewise the published `57.7` may be quoted only
alongside this project's `60.80` reproduction of it, never as a direct
comparator for a number produced in this environment.

## 10. FULL DISCLOSURE OF PRIOR KNOWLEDGE

Known to the author before writing: every P0 per-seed value for `U_PUBLISHED`
(42–67) and `D_BDEV`; the adaptive mu-curve (84.00 / 76.62 / 69.85 at
μ = 0.001 / 0.005 / 0.01) and the matching utility curve (81.28 / 80.35 / 77.30,
original 81.58); V2_UINIT 0.7258 and V2_UINIT_VER 0.6969; the entire sealed
diagnosis; that the anchor on seeds 42–46 is 0.6336, i.e. **a harder bar than
its 26-seed mean of 0.6985**; and the gradient table of §1.

The anchor side of the §7 comparison is fully known; the candidate side does not
exist. The design is anti-anchored where it can be, and the asymmetry is
disclosed rather than papered over.

## 11. ANTI-ANCHORING COMMITMENTS

No seeds added. No third arm. No objective, weight, schedule or temperature
change after any result. No threshold edits. No post-hoc substitution of the
final-epoch checkpoint for the selected one. Classification made **exactly once**
from the sealed manifests. Immutable after unblinding; dated addenda only.

## 12. COST AND CONSEQUENCE MAP

Stage 1: 2 generators × 50 epochs ≈ 11 GPU-h; 2 arms × 5 attackers ≈ 8.5 GPU-h;
utility ≈ 0.2 GPU-h. **≈ 20 GPU-h.** Stage 2 adds ≈ 22 GPU-h; §9 adds ≈ 6.

| Classification | Action |
|---|---|
| **H-B2-SUPERIOR** | Stage 2, then §9. Manuscript gains a method section. Recompute the saliency-alignment measurement on the winning checkpoint: the diagnosis predicts the corrected objective reallocates deformation toward identity-bearing pixels (currently deformation is uniform at 1.08x while attacker saliency is 1.89x concentrated, spatial overlap 1.03 ≈ independent). Confirming that prediction turns a number into a mechanism. |
| **H-B2-EQUIVALENT / NOT-SUPPORTED** | Reported as an honest null with the `gap` and saliency trajectories as forensics. |
| **H-B2-MECHANISM-FAILED** | Report why `gap` did not move. Not a licence to retune here. |
| **H-B2-UTILITY-FAIL** | Report the privacy-utility points obtained; no privacy claim. |

**The manuscript does not depend on this experiment.** The sealed diagnosis and
the adaptive mu-curve — the latter obtained with zero generator training — stand
on their own.

*End of preregistration.*
