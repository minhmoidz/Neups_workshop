# S2 CONFIRMATORY DESIGN PROPOSAL — Seed-Count Justification (Planning Document, Not Authorized for Execution)

**Date:** 2026-08-18
**Status:** DESIGN ONLY. No experiment, seed count, or execution is authorized by this document. Produced in
response to the independent peer review (2026-08-18) which withdrew the prior ad-hoc "5 anonymizer seeds ×
3 attacker seeds" recommendation for lacking a power/margin justification. This document supplies that
justification framework and a worked table; it does not commit the project to any specific number until the
inputs below are confirmed with real M2 data (currently unavailable — S1 has not finished).

---

## 1. What S2 needs to test

The frozen S1 gate (`research_agent/m2_dev/run_m2_s1.py:989`) is a one-seed point estimate:
`Δpriv = AUC_C4 − AUC_B_dev ≤ +0.03`. A confirmatory S2 needs to answer the same question with an estimated
**paired-difference distribution** across multiple independent seed replicates, and a **one-sided
non-inferiority confidence interval** on that distribution, per the peer review's required final decisions.

**Design target (one-sided non-inferiority test on paired differences):**

```
H0:  E[Δ] ≥ margin        (C4 is inferior by at least the margin — bad)
H1:  E[Δ] < margin        (C4 is non-inferior — the claim we want to support)

Δ_i = AUC_C4,i − AUC_B_dev,i   for each independent paired replicate i
margin = 0.03   (unchanged from the frozen S1 gate)
```

## 2. What we do and do not know about the variance

**Known (from Workstream A, the original PriCheXy-Net reproduction — a *proxy*, not a direct M2 measurement):**
- Re-training the Siamese attacker on a **fixed** generator checkpoint, varying only the attacker's own seed,
  gives Re-ID AUC standard deviation of **0.0435** (n=10, FP32 block) to **0.0507** (n=20, AMP block).
- This is the best available empirical estimate of **attacker-seed variance holding the generator fixed** — it
  is *not* an M2-pipeline measurement (different codebase, different architecture wiring, different loss), and
  it is *not* an estimate of anonymizer-seed (generator retraining) variance at all, because the original
  reproduction always reused one fixed released generator checkpoint across all seeds.

**Unknown — a real, disclosed gap, not glossed over:**
- **Anonymizer-seed variance** (how much `AUC_C4 − AUC_B_dev` would move if the *entire* 250-epoch generator
  training were rerun under a different seed) has never been measured anywhere in this project. Generator
  training involves multi-loss adversarial dynamics (generator vs. verifier critic vs. auxiliary critic) and
  is plausibly **at least as seed-sensitive** as attacker retraining alone, possibly more.
- **Correlation (ρ) between the B_dev and C4 arms' paired difference**, induced by the common-random-numbers
  design already in place (`b_dev_epoch0_order_matches_c4: true` in the execution lock — both arms see
  identical batch order). Positive correlation *reduces* the paired-difference variance below the marginal
  variance; it has never been measured.

Because both of these are unknown, **any seed-count number produced today is a planning estimate under
explicit sensitivity assumptions, not a final number.** The table in §4 is presented as a sensitivity grid
for exactly this reason — pick a design point once real data narrows it, don't pick one number now.

## 3. Formula

Standard one-sided non-inferiority sample size (paired-difference framework), assuming the true difference
under the design scenario is `Δ_true = 0` (arms assumed truly equivalent — the conservative/base case):

```
n = σ_d² · (z_(1-α) + z_(1-β))² / margin²

σ_d  = paired-difference SD = σ_marginal · √(2·(1 − ρ))
α    = 0.05 one-sided  →  z_(1-α) = 1.6449
```

## 4. Sensitivity table (planning grid, not a decision)

| σ_marginal proxy | ρ (arm correlation) | σ_d (paired SD) | n for 80% power | n for 90% power |
|---:|---:|---:|---:|---:|
| 0.0435 (FP32 block) | 0.0 (no correlation — pessimistic) | 0.0615 | 26 | 37 |
| 0.0435 | 0.3 | 0.0515 | 19 | 26 |
| 0.0435 | 0.5 | 0.0435 | 13 | 19 |
| 0.0435 | 0.7 (strong CRN effect — optimistic) | 0.0337 | 8 | 11 |
| 0.0507 (AMP block) | 0.0 | 0.0717 | 36 | 49 |
| 0.0507 | 0.3 | 0.0600 | 25 | 35 |
| 0.0507 | 0.5 | 0.0507 | 18 | 25 |
| 0.0507 | 0.7 | 0.0393 | 11 | 15 |

**Reading:** required paired replicates range from **8 to 49** depending entirely on the unmeasured correlation
assumption and which variance proxy is used — a 6× spread. This range, not a single number, is the honest
current answer to "how many seeds does S2 need." It also assumes `Δ_true = 0`; if C4 is expected to run close
to the 0.03 margin rather than truly equivalent, required n grows sharply (this table does not cover that case
and should not be read as covering it).

## 5. Recommended path to close the gap (design only — not authorized to run now)

A cheap, staged approach to replace assumption with measurement, **once S1 finishes** and the B_dev/C4
generator checkpoints exist:

1. **Stage A — attacker-seed pilot on the existing S1 checkpoints (cheap).** Re-run only the attacker
   training (not the generator) for a handful of additional attacker seeds on the *already-selected* B_dev and
   C4 generator checkpoints from S1. Attacker retraining is far cheaper than generator retraining (observed
   ≈10–20 minutes per run in the reproduction logs, vs. ≈27 hours per 250-epoch generator arm). This directly
   measures attacker-seed variance and the B_dev/C4 correlation **inside the actual M2 pipeline**, replacing
   the external reproduction-based proxy in §2 with a real one.
2. **Stage B — decide, with Stage A's numbers, whether anonymizer-seed variance can be responsibly assumed
   small or must be measured.** Only if Stage A's attacker-seed-only variance is a large fraction of the 0.03
   margin does a small number of full anonymizer-seed reruns (expensive) become necessary before S2 can be
   trusted; if attacker-seed variance alone is already tight relative to the margin, fixing the anonymizer seed
   for S2 (accepting that source of variance as an explicitly disclosed limitation) may be an acceptable,
   cheaper design.
3. **Re-run §4's table with Stage A's measured σ_d and ρ** to get a real, not sensitivity-bounded, required n.

None of Stage A, B, or the table re-run is authorized by this document. This is a plan to execute *after* S1
completes and *with explicit sign-off*, not a task in progress.

## 7. Stage A — concrete pilot plan (now that S1 has finished, 2026-08-20)

S1 completed with verdict **"C4 S1: PROMOTE TO S2"** (Δpriv = −0.0109, PASS ≤+0.03; Δclass = +0.0216,
PASS ≥0.0; single seed=42, VAL fold, `test_touched: false`). The two selected generator checkpoints now
exist and their manifests are real, not hypothetical:

| Arm | Selected generator checkpoint | SHA256 | Best epoch |
|---|---|---|---:|
| B_dev | `research_runs/M2_S1/B_dev/seed_42/generator_best_method_neutral.pth` | `18381d92c6...` | 13 |
| C4 | `research_runs/M2_S1/C4/seed_42/generator_best_method_neutral.pth` | `366a7dd083...` | 8 |

`train_s1_attacker_arm()` (`research_agent/m2_dev/run_m2_s1.py:212-292`) already supports reusing an
**existing** generator checkpoint via its `checkpoint_manifest.json` and training only a fresh attacker —
exactly Stage A's cheap operation. Observed real cost from the S1 run: attacker training with
`patience=5` early-stopped at epoch 12 in **1233–1263 s (≈20–21 min)** per arm — confirms the "cheap
pilot" cost estimate in §5 was not optimistic.

**Concrete blocker found while checking feasibility (not present in §5's abstract description):**
`run_m2_s1.py:250-251` hard-enforces
`if attacker_seed != attacker_cfg.get('attacker_seed', 42): raise ValueError(...)`, and the frozen
`config_files/config_dev_attacker_s1.json` pins `"attacker_seed": 42`. This means **the current
scientific pipeline refuses to run any attacker seed other than 42** — this is a deliberate
scientific-integrity guard, not a bug. A Stage A pilot (attacker_seed ∈ {43, 44, 45, 46, 47}, say) is
therefore **not executable today** without a prior, explicit governance step: minting a new frozen
attacker-config artifact (e.g. `config_dev_attacker_s2_pilot.json`, same fields, a different pinned seed
or a seed-list, with its own SHA256 registered the same way `FROZEN_ATTACKER_CONFIG_SHA` is registered
in `evaluator_common.py`) — the same predeclare-then-hash-gate pattern already used for every other
frozen artifact in this pipeline. That is a protocol/config decision for a human to make, not something
to route around with an ad-hoc code edit.

**Proposed Stage A scope once that governance step is taken (not authorized by this document):**
- 5 additional attacker seeds per arm (e.g. 43–47) reusing the existing B_dev and C4 checkpoints above.
- Estimated cost: 5 × 2 arms × ~21 min ≈ **3.5 GPU-hours total** — negligible next to the ~59 h the S1
  generator training itself took.
- Output: 10 new `(arm, attacker_seed)` privacy AUC points, from which real attacker-seed variance and
  the B_dev/C4 paired correlation ρ can be estimated directly inside the M2 pipeline, replacing the
  external-reproduction proxy in §2.

This section documents feasibility and cost; it does not create the new config artifact, does not run
any attacker training, and does not modify `config_dev_attacker_s1.json` or any frozen SHA registered in
`evaluator_common.py`.

**Resolution actually taken (2026-08-20, explicit user authorization):** rather than the governance route
above, the pilot was run via a standalone script,
`reproduction/s2_pilot/run_stage_a_pilot.py`, living entirely outside `research_agent/m2_dev/`. It never
touches `config_dev_attacker_s1.json` or any frozen SHA; it reads that file read-only (asserting its
SHA256 still equals `FROZEN_ATTACKER_CONFIG_SHA` before use) and overrides only the in-memory
`attacker_seed` field. Its training/validation loop (`PilotAttacker`) is a line-for-line copy of
`DevAttacker` (`research_agent/m2_dev/dev_attacker.py`) minus the `attacker_seed == 42` assertion — same
optimizer, geometry, early-stopping, and NaN/Inf fail-fast checks. The final AUC for every pilot point was
computed by calling the real, unmodified `evaluate_reid_val()` — not a reimplementation. All 10 pilot runs
(seeds 43–47 × {B_dev, C4}) were executed **sequentially** (not in parallel — an initial parallel launch
was caught and reverted before either job trained, since peak per-job VRAM (~8.8–9.7 GB) left no safety
margin on the 16.3 GB card and parallel execution has no precedent in how S1 itself was run). Every output
is labeled `"pilot_uncertified": true` and lives under `reproduction/s2_pilot/results/`, never under
`research_runs/M2_S1/`.

## 8. Stage A pilot results — final (real data, n=26 seeds/arm, 2026-08-20)

Extended from the initial n=6 pilot to the full **n=26 seeds per arm** (seeds 42 [S1, certified] + 43–67
[pilot], same fixed S1-selected generator checkpoints, attacker-only retraining). Full per-seed table
lives in `reproduction/s2_pilot/results/`; summary statistics:

| Quantity | Value |
|---|---:|
| B_dev AUC: mean / SD | 0.8237 / 0.0309 |
| C4 AUC: mean / SD | 0.8326 / 0.0288 |
| Δ (C4 − B_dev): mean / SD | +0.0089 / 0.0385 |
| Pearson ρ(B_dev, C4), matched by attacker seed | 0.171 |
| Seeds where Δ > 0.03 (would FAIL the S1 gate if drawn as "the" seed) | 7 / 26 (27%) |

**Formal non-inferiority test** (H0: mean Δ ≥ 0.03 vs H1: mean Δ < 0.03, one-sided, α=0.05, paired t-test,
df=25): **t = −2.791, p = 0.0050 → REJECT H0.** One-sided 95% upper confidence bound on mean Δ = **0.0218**,
safely below the 0.03 margin. **This is the confirmatory result S2 was designed to produce for the privacy
gate**, conditional on the fixed S1-selected generator checkpoints (attacker-seed variance only —
anonymizer-seed variance, see below, is still not covered).

**Separately: no superiority.** Two-sided test of mean Δ against 0 gives t = 1.185, p = 0.247 — **not
significant**. The 95% CI on mean Δ is **[−0.0066, +0.0245]**, which contains zero. C4 cannot be claimed to
reduce Re-ID AUC relative to B_dev; the point estimate is even slightly in the unfavorable direction
(+0.0089). Non-inferiority and superiority are different claims — S1's single-seed "PROMOTE" verdict was
compatible with the (unproven) hope that C4 *actively helps* privacy; n=26 confirms it does *not hurt*
privacy beyond the margin, but does not support it *helping*.

**Key methodological finding — the common-random-numbers correlation assumption in §4 was optimistic.**
Real ρ ≈ 0.17 (updated from the n=6 estimate of ≈0.02, now more precise at n=26) — much closer to zero than
the 0.3–0.7 rows explored in §4's original sensitivity table. Practical implication: this pipeline's CRN
design (identical batch order at generator training time) does very little to correlate B_dev/C4 outcomes
once attacker-seed noise is introduced at eval time — future confirmatory designs here should not assume
CRN buys much variance reduction.

**What remains open:** this whole analysis fixes the generator at its S1-selected checkpoint for each arm.
**Anonymizer-seed variance (retraining the 250-epoch generator under a different seed) is still completely
unmeasured** — Stage B (§5, item 2) remains open, and is expensive (~27–59 GPU-hours per retrain vs. ~21
minutes per attacker-only pilot run). Whether the non-inferiority conclusion above is robust to a different
generator training run is genuinely unknown.

## 9. What this document does not do

- Does not authorize, schedule, or begin any training, evaluation, or GPU use.
- Does not modify `research_agent/m2_dev/`, any config, lock, checkpoint, or manifest.
- Does not alter the frozen S1 gate (`Δpriv ≤ 0.03`) — that gate remains the governing rule for the S1 decision
  already in flight.
- Does not commit to 80% vs. 90% power, or to any specific σ_d/ρ point in §4 — those choices belong to whoever
  authorizes S2, informed by Stage A measurements, not to this planning document.
