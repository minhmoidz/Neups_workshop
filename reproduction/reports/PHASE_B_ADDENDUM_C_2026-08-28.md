# ADDENDUM C to PHASE_B_CORRECTED_OBJECTIVE_PREREGISTRATION_2026-08-28

**Date:** 2026-08-28, evening.
**Amends:** the parent preregistration (commit `96aacbe`) and Addendum B
(commit `780e618`). Both files are unmodified.
**Type:** outcome record — Phase B is **CLASSIFIED AND CLOSED**.

## 0. CLASSIFICATION

**`H-B-MECHANISM-FAILED`** (parent §7, row 4).

Per the parent §7 this means the objective failed to move *its own target* `s`,
which is a distinct and separately informative outcome from moving `s` without
moving the real AUC. The distinction was preregistered precisely so this case
could be recorded honestly rather than reported as a generic null.

**The outcome variable was never observed.** No candidate checkpoint was
submitted to the P0 adaptive attacker, no candidate AUC exists, and no
comparison against D_BDEV was computed. This classification rests entirely on
the training diagnostic that parent §8 designates for exactly this purpose.

## 1. WHAT WAS RUN

| | |
|---|---|
| Commit | `780e618` |
| Runner | `reproduction/method_dev/run_corrected_objective.py` |
| Config | frozen `config_dev_restored_baseline.json` (SHA `14d3943f…`), seed 42 |
| Epochs completed | **6** of 250 |
| GPU time consumed | **0.66 h** |
| Order hashes verified against oracle | **6 / 6** |
| Degenerate batches | **0** |
| NaN/Inf | none |

Every integrity gate of parent §4 passed. **The runner is not at fault; the
objective is.**

Trajectory (pooled VAL, the quantity parent §8 reads):

| epoch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `s_val` | 0.9953 | 0.9952 | 0.9953 | 0.9952 | 0.9950 | 0.9954 |
| `val_ac_bce` | 0.1455 | 0.1458 | 0.1455 | 0.1452 | 0.1456 | 0.1455 |

Flat to four decimals. For contrast, the **original** objective on the same
recipe moves its own privacy term 0.415 → 0.233 → 0.197 → … → 0.047 over epochs
0–5 (`hardened_verifier_k3/B_dev/seed_42/epoch_metrics.csv`). The corrected
objective produced no movement at all.

## 2. DIAGNOSED CAUSE — GRADIENT SATURATION

Measured directly, not inferred. Gradient norm of `L_priv = (s − 0.5)^2` with
respect to the verifier logits, as a function of class separation and `tau`:

| separation regime | `tau` | `s` | ‖dL/dz‖ |
|---|---:|---:|---:|
| weak (AUC ≈ 0.6) | 0.1 | 0.7309 | 9.26e-02 |
| moderate (AUC ≈ 0.9) | 0.1 | 0.9584 | 9.18e-02 |
| **Phase B start (AUC ≈ 0.995)** | **0.1** | **1.0000** | **1.76e-06** |
| Phase B start | 0.3 | 0.9968 | 1.70e-03 |
| Phase B start | 1.0 | 0.8645 | 5.03e-03 |

At the regime the run actually begins in, `tau = 0.1` gives a gradient **five
orders of magnitude** smaller than at moderate separation. Standardization fixes
`std(ẑ) = 1`, so with well-separated balanced classes the pairwise differences
are ≈ 2; at `tau = 0.1` the sigmoid argument is ≈ 20 and is fully saturated. The
objective is numerically dead exactly where training starts.

**This is an error in Addendum B, and its origin is specific and worth
recording.** Addendum B §3 validated `tau = 0.1` on two properties — accuracy
against true AUC, and no overshoot under optimization — and both validations
stand. It additionally reported a gradient-magnitude check showing the new
objective was ~2x stronger than `softplus(z)`. That check was performed at a
*moderate* operating point (logits centred near −2.9 with unit spread), **not at
the near-perfect-separation state the generator is initialized into**. The
right check was the right idea evaluated in the wrong regime.

## 3. EARLY STOP — DISCLOSED DEVIATION

Parent §8 sets the futility check at **epoch 50**. The run was stopped at
**epoch 6**.

Declared reasons, recorded so this cannot be mistaken for outcome-driven
stopping:

1. The decision used a **mechanistic measurement** — the gradient norm of §2 —
   and the observed flat `s_val`. It did **not** use the outcome variable, which
   was never computed.
2. Continuing would have consumed a further ~4.8 GPU-h to reconfirm a quantity
   already measured to be 1.76e-06.
3. The futility rule exists to protect GPU budget. Stopping earlier on direct
   evidence of the mechanism serves that purpose more strongly, not less.

The deviation is nonetheless a deviation and is recorded as one. A stricter
reading would have run to epoch 50; nothing about the classification would have
changed.

## 4. WHAT IS NOT AUTHORIZED

Parent §8 states that a futility stop **does not** authorize changing the
objective, its weight, or any hyperparameter and re-running, and that doing so
requires a new preregistration. That clause is honored here.

Specifically:
- `tau` is **not** retuned within this preregistration.
- No Phase B result is recomputed, reinterpreted, or rescued.
- The corrected-objective route is **not** abandoned; it is moved to a
  successor experiment (§5) under its own lock.

## 5. SUCCESSOR (design recorded, not yet preregistered)

The saturation is a property of a *fixed* small `tau`, not of separability
targeting as such. The intended repair is **`tau` annealing**: begin at
`tau ≈ 1.0`, where §2 shows gradient flows at the initial separation, and anneal
toward `tau ≈ 0.1`, where Addendum B §3 shows the estimator is accurate and does
not overshoot. Large `tau` early buys direction; small `tau` late buys accuracy.

A Phase B2 preregistration must, at minimum:
1. Fix the annealing schedule before execution and forbid tuning it afterwards.
2. Verify gradient magnitude **at the actual initial separation regime**, which
   is the check Addendum B performed in the wrong regime.
3. Retain every integrity gate of parent §4, all of which passed here.

## 6. WHAT PHASE B ESTABLISHED

Not nothing. The run demonstrated on real data that:
- the runner and all seven integrity gates work end-to-end;
- `s_val` (pooled) tracks the co-adapted critic's true AUC to ~0.0015
  (0.9953 vs 0.9968 at epoch 0), confirming Addendum B §3b on-device;
- exactly **1 of 125** VAL batches contains both classes, confirming that the
  pooling repair of Addendum B §3b was necessary rather than merely preferable;
- a separability-targeting objective must be designed against gradient
  saturation at the near-perfect-separation initialization — a technical
  finding that belongs in the manuscript regardless of whether Phase B2 succeeds.

**The manuscript does not depend on this experiment.** Parent §12 recorded that
before execution, and it remains true: the sealed diagnosis
(`privacy_objective_diagnosis.json`, commit `09046f7`) stands at zero GPU cost.

*End of Addendum C. Phase B is closed. Corrections require Addendum D, dated.*
