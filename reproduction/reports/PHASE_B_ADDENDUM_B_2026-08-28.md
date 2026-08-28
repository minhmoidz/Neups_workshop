# ADDENDUM B to PHASE_B_CORRECTED_OBJECTIVE_PREREGISTRATION_2026-08-28

**Date:** 2026-08-28 (same day, hours after the parent lock at commit `96aacbe`).
**Amends:** §2 of `PHASE_B_CORRECTED_OBJECTIVE_PREREGISTRATION_2026-08-28.md`
— the definition of the AUC surrogate. That file is unmodified.
**Type:** pre-execution design correction.

## 0. WHY THIS IS LEGITIMATE, AND WHAT WOULD MAKE IT NOT

**No run exists.** Verified at the time of writing, exactly as at the parent
lock: no candidate artifact of any kind, `run_corrected_objective.py` had not
been executed even once, and `reproduction/method_dev/corrected_objective_*/`
does not exist. **No outcome value has been observed, so none can have
influenced this change.**

The evidence below comes entirely from **synthetic logits**, generated from a
fixed seed inside a self-contained numerical test. No project data, no
checkpoint, no attacker and no AUC of any real generator was consulted.

This addendum would be illegitimate if it were written after seeing a candidate
result and used to explain away a failure. It is written *before* the first
training step precisely so that cannot happen. Had this flaw been found after
the run, the correct action would have been to report the flawed run and
preregister a new experiment — not to amend.

## 1. THE DEFECT IN THE PARENT §2 FORMULA

Parent §2 specifies

```
s = mean_{i in P, j in N} sigmoid(z_i - z_j),    L_priv = (s - 0.5)^2
```

Pre-flight testing found this reproduces, in a different guise, **the very
defect the experiment exists to diagnose**.

### 1.1 It is not scale-invariant — the shrink loophole

True ROC AUC is invariant to any monotone rescaling of the logits. `sigmoid(z_i
- z_j)` is not. Scaling a fixed, genuinely separable logit vector (true AUC
0.9808 at every row):

| logit scale | mean \|z\| | surrogate `s` | true AUC | gap |
|---:|---:|---:|---:|---:|
| 5.00 | 7.954 | 0.9779 | 0.9808 | 0.003 |
| 1.00 | 1.591 | 0.9092 | 0.9808 | 0.072 |
| 0.10 | 0.159 | 0.5752 | 0.9808 | 0.406 |
| 0.03 | 0.048 | 0.5228 | 0.9808 | **0.458** |

At small scale the surrogate reads **0.523 — apparent chance — while every bit
of identity ordering is intact at AUC 0.981**. A generator could therefore
satisfy the objective by producing images on which the verifier is merely
*uncertain in magnitude*, without removing any identity information. That is
the same failure mode as `softplus(z)`: a calibration manipulation that leaves
the ranking untouched.

### 1.2 It is biased, so hitting 0.5 does not mean hitting chance

Even at unit scale the estimator is compressed toward 0.5 (0.9092 vs a true
0.9808). Driving a biased estimate to its target overshoots the real quantity.
Optimizing free logits to minimize `(s - 0.5)^2`:

| variant | final `s` | **true AUC reached** |
|---|---:|---:|
| standardized, tau = 1.0 | 0.5000 | **0.4249** |
| standardized, tau = 0.5 | 0.5000 | 0.4772 |

An AUC of 0.425 re-identifies exactly as well as 0.575 — the parent's own §2
notes that "an AUC of 0.0 is as re-identifying as 1.0 and is explicitly not
sought", yet the formula as written does not enforce it.

## 2. THE AMENDED DEFINITION

```
zhat = (z - mean(z)) / (std(z) + eps),     eps = 1e-6
s    = mean_{i in P, j in N} sigmoid((zhat_i - zhat_j) / tau),   tau = 0.1
L_priv_new = (s - 0.5)^2
```

Two changes, each fixing one defect above:

- **Standardization** over the batch removes the scale degree of freedom
  entirely, so §1.1's loophole is unreachable: `s` is now invariant to any
  affine rescaling of `z`, exactly as true AUC is.
- **Temperature `tau = 0.1`** removes the bias of §1.2. As `tau -> 0` the
  sigmoid approaches the indicator and `s` approaches the exact
  Wilcoxon–Mann–Whitney statistic, which *is* ROC AUC.

Everything else in the parent is unchanged: the guard, the squared penalty, the
target 0.5, the weights, the unchanged verifier critic update, and every frozen
hyperparameter of §3.

## 3. EVIDENCE FOR `tau = 0.1`

**Accuracy** — standardized surrogate vs true AUC across separations:

| tau | sep 2.0 | sep 1.0 | sep 0.5 | sep 0.2 |
|---:|---:|---:|---:|---:|
| 1.00 | 0.8401 | 0.7650 | 0.6633 | 0.5689 |
| 0.50 | 0.9482 | 0.8554 | 0.7185 | 0.5925 |
| 0.20 | 0.9915 | 0.9062 | 0.7511 | 0.6062 |
| **0.10** | **0.9958** | **0.9154** | **0.7576** | **0.6087** |
| 0.05 | 0.9965 | 0.9178 | 0.7594 | 0.6093 |
| **true AUC** | **0.9968** | **0.9184** | **0.7600** | **0.6096** |

At `tau = 0.1` the estimator tracks true AUC to ~0.003 across the whole range.

**No overshoot** — optimizing free logits against each temperature:

| tau | final `s` | true AUC reached |
|---:|---:|---:|
| 1.00 | 0.5000 | 0.4249 |
| 0.50 | 0.5000 | 0.4772 |
| 0.20 | 0.5000 | 0.5045 |
| **0.10** | **0.5000** | **0.4921** |

`tau = 0.1` lands at chance. `tau = 0.05` was measured as marginally more
accurate but sharpens gradients for no decision-relevant gain, so `0.1` is
taken. **`tau` is fixed at 0.1 and is NOT a tunable hyperparameter within this
preregistration.** Any future search over `tau` requires a new preregistration
and multiplicity control.

**Gradient scale** — the amended objective is not weaker than the one it
replaces, so `ver_loss_weight = 1.0` (frozen, inherited) remains viable:

| objective | value at a realistic operating point | ‖dL/dz‖ |
|---|---:|---:|
| old `softplus(z).mean()` | 0.0951 | 0.0336 |
| **new, tau = 0.1** | 0.1648 | **0.0663** |

The new term is ~2x stronger in gradient and comparable in magnitude to
`ac_bce` (~0.147) at equal weights. This was checked to reduce the chance of a
futility stop caused merely by a weak term rather than by a real mechanism
failure — it does **not** change the §8 futility rule, which stands unamended.

## 3b. SECOND PRE-FLIGHT FINDING — the VAL privacy term must be POOLED

Found by the same pre-flight process, before any run, when the live smoke test
returned `s_pooled = nan` and sklearn warned *"Only one class is present in
y_true"*.

**Cause, verified in the data:** `image_pairs/image_pairs_validation_2000.txt`
is **sorted by class** — 1000 positives followed by 1000 negatives — and the
VAL loader is sequential (`shuffle=False`, `build_dev_anonymizer_loaders`).
With batch 16, **exactly 1 of 125 validation windows contains both classes**:

| fold | sequential batch-16 windows with both classes |
|---|---|
| validation (sequential loader) | **1 / 125** |
| training (permuted by `FingerprintedRandomSampler`) | unaffected — the sampler shuffles |

**Consequence had this not been caught:** the surrogate is undefined on a
single-class batch, so a per-batch average of `L_priv_new` over the VAL fold
would have been computed from that one window and been ~0 everywhere else. The
prereg §5 selection statistic `val_ac_bce + val_L_priv_new` would have silently
collapsed to `val_ac_bce` alone — i.e. checkpoint selection would have ignored
privacy entirely, for the whole run, with nothing in the logs looking wrong.
Training is **not** affected: the TRAIN sampler permutes, and the smoke test
confirms real training batches contain both classes.

**Resolution.** `val_L_priv_new` and the `s_val` of the §8 futility rule are
computed from the **pooled** logits of the entire VAL fold:

```
s_val      = surrogate(all VAL logits, all VAL labels)
val_L_priv_new = (s_val - 0.5)^2
```

This is within the text of §5 (`val_selection_total = val_ac_bce +
val_L_priv_new`); only the estimator of `val_L_priv_new` is specified, and
pooling is both the sole computable choice on this fold and the honest
estimator. The per-batch value is retained in the logs as
`val_s_surrogate_perbatch_DIAGNOSTIC` together with
`val_n_batches_with_both_classes`, explicitly non-decisive, so the degeneracy
stays visible in every epoch record rather than being hidden by the fix.

## 4. WHAT THIS ADDENDUM DOES NOT CHANGE

- No decision rule, threshold, classification, seed, anchor or utility gate.
  §§5–12 of the parent stand verbatim; `delta` = 0.03 remains frozen.
- No frozen hyperparameter of §3. `batch_size` is still 16.
- The §8 futility rule, its epoch and its 0.05 threshold are unchanged; the
  quantity it reads (`|s_val - 0.5|`) is now computed with the amended `s`.
- The anti-anchoring commitments of §11 apply to this addendum too: `tau` and
  the standardization are fixed **now**, before execution, and may not be
  revisited on the basis of any result.

## 5. IMPLEMENTATION POINTER

`reproduction/method_dev/run_corrected_objective.py`,
`CorrectedObjectiveRunner.auc_surrogate`, constants `SURROGATE_TAU = 0.1` and
`SURROGATE_EPS = 1e-6`. The run manifest records `surrogate_standardized`,
`surrogate_tau` and a pointer to this file, so any output is bound to the
amended definition rather than the parent's.

*End of Addendum B. Corrections require Addendum C, dated.*
