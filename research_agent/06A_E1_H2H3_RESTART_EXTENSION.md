# STEP 4D — E1 H2/H3 Restart Extension

**Final validation package before freezing the negative/mechanistic-paper claim. TRAIN/VALIDATION only. CAA status unchanged (NO-GO).**

## 1. Purpose and scope

STEP 4B measured each of the two CAA mechanism diagnostics (H2 intensity, H3 border/FOV) with a **single** fresh attacker restart (seed 42). That is underpowered for a mechanistic paper claim. E1 extends each diagnostic to **n = 3** using exactly two additional authorized seeds (**43, 44**), nothing else.

Only the **precision** of the mechanism estimate changes. The H2/H3 transforms, the attacker recipe, the reference, and the validation protocol are **unchanged** and reused from STEP 4B. CAA remains NO-GO regardless of outcome. No TEST, no Stage E, no Top-k, no generator training, no method search reopened.

## 2. Preconditions

- `git status`: clean (zero lines) before execution.
- `git branch --show-current`: `main`.
- `git rev-parse HEAD`: `f050c79d94afc0760df91b447465b3f5ac1964cd` (includes STEP 4B ancestry; required commit `f050c79` confirmed).
- Reference recomputed from the **raw** STEP 3D per-seed records (not hardcoded):
  `best_validation_auc` over seeds 0–9 → `n = 10`, `mean = 0.8382`, **sample SD (ddof = 1) = 0.0363**, median 0.8454, min 0.7927, max 0.8776.

## 3. Arm H2 — intensity

Canonical transform **unchanged** (STEP 4B): per-image `p1`/`p99` robust affine normalization, `eps = 1e-6`, applied consistently to attacker TRAIN and VALIDATION. Attacker-A recipe unchanged (SiameseNetwork ResNet-50, Adam 1e-4, batch 16, patience 5, checkpoint = lowest validation loss, max 100 epochs). Fresh runs completed for seeds 43 and 44 (seed 42 reused from STEP 4B).

Per-seed adaptive validation AUC (best over epochs, checkpoint = lowest validation loss):

| seed | best val AUC | best epoch | epochs run | termination | test_touched |
|-----:|-------------:|-----------:|-----------:|-------------|:------------:|
| 42   | 0.80750 | 7  | 13 | early_stopping | false |
| 43   | 0.87659 | 5  | 11 | early_stopping | false |
| 44   | 0.81602 | 9  | 12 | early_stopping | false |

## 4. Arm H3 — border/FOV

Canonical transform **unchanged** (STEP 4B): outermost 4-pixel image border replaced by per-image median intensity (`BORDER_WIDTH = 4`), applied consistently to TRAIN and VALIDATION. Attacker-A recipe unchanged. Fresh runs completed for seeds 43 and 44 (seed 42 reused from STEP 4B).

Per-seed adaptive validation AUC:

| seed | best val AUC | best epoch | epochs run | termination | test_touched |
|-----:|-------------:|-----------:|-----------:|-------------|:------------:|
| 42   | 0.79380 | 8  | 11 | early_stopping | false |
| 43   | 0.83919 | 7  | 13 | early_stopping | false |
| 44   | 0.82036 | 10 | 11 | early_stopping | false |

## 5. Validity

Every completed finite run is VALID regardless of performance; near-chance runs stay included and are never replaced merely for performance. All six runs terminated by early_stopping with `test_touched = false`. **No replacement was needed** — no seed was replaced, and no seed 45+ was invented.

## 6. Summary statistics (sample SD, ddof = 1)

| Arm           | Seeds    |  n | Mean adaptive val AUC | Sample SD | Median | Min | Max |
| ------------- | -------- | -: | --------------------: | --------: | -----: | --: | --: |
| H2 intensity  | 42,43,44 |  3 | **0.8334**             | 0.0377    | 0.8160 | 0.8075 | 0.8766 |
| H3 border/FOV | 42,43,44 |  3 | **0.8178**             | 0.0228    | 0.8204 | 0.7938 | 0.8392 |
| reference     | 0–9      | 10 | 0.8382                 | 0.0363    | 0.8454 | 0.7927 | 0.8776 |

## 7. Between-arm effect (independent-arm CI, no seed pairing)

Reduction = `mean_AUC_reference − mean_AUC_arm`. Uncertainty via Welch t-interval on the difference of two independent group means, plus an independent-arm percentile bootstrap (10 vs 3 resamples, 5,000 draws) as a robustness check. No seed IDs are paired; no significance is claimed from point estimates.

| Arm           | Reduction | Welch 95% CI | Bootstrap 95% CI | Material ≥0.10 supported? |
| ------------- | --------: | ------------ | ----------------- | -------------------------- |
| H2 intensity  | +0.0048   | [−0.0706, 0.0803] | [−0.0379, 0.0426] | **no** (CI upper ≪ 0.10) |
| H3 border/FOV | +0.0204   | [−0.0233, 0.0641] | [−0.0081, 0.0493] | **no** (CI upper ≪ 0.10) |

Statistical details (Welch): H2 SE 0.0246, df 3.2, t(vs 0) 0.20; H3 SE 0.0175, df 5.5, t(vs 0) 1.17. One-sided mass above 0.10: H2 0.986 (i.e. ~1.4% tail) and H3 0.998 — the data **strongly disfavor** a material ≥ 0.10 effect.

## 8. Predeclared scientific interpretation

For paper-mechanism purposes only (CAA method status is unaffected):

- H2 intensity mean reduction **+0.0048** — far below 0.10; uncertainty excludes 0.10 → **no material ≥ 0.10 effect detected**.
- H3 border/FOV mean reduction **+0.0204** — far below 0.10; uncertainty excludes 0.10 → **no material ≥ 0.10 effect detected**.
- Neither channel shows a MATERIAL adaptive effect. The extended three-seed estimates corroborate STEP 4B (single-seed H2 0.8075 / H3 0.7938), now at higher precision, and support the negative/mechanistic claim that residual (corrected-baseline) identity is not carried predominantly by the acquisition/intensity or border/FOV channel.

The old 0.05 screening threshold is **not** reused as a powered mechanistic falsification threshold; the ≥ 0.10 material-effect rubric is applied throughout.

## 9. Utility

Transforms are unchanged from STEP 4B; STEP 4B already established VALIDATION utility (diagnostics, not TEST): H2 intensity class AUC 0.7947 / Dice 0.9542; H3 border class AUC 0.7928 / Dice 0.9528. Not rerun (no transform change); no TEST utility statistics applied.

## 10. Artifacts and provenance

Per-seed evidence (all `test_touched: false`, `mechanism_diagnostics.json` SHAs):
- `05B_artifacts/adaptive/diag_4b_arm_intensity_seed42/…` (STEP 4B, `27495762…feb3e`)
- `05B_artifacts/adaptive/diag_4b_arm_intensity_seed43/mechanism_diagnostics.json` `0399d528…e41`
- `05B_artifacts/adaptive/diag_4b_arm_intensity_seed44/mechanism_diagnostics.json` `33b2ae85…bdec`
- `05B_artifacts/adaptive/diag_4b_arm_border_seed42/…` (STEP 4B, `9af65421…dc9b9`)
- `05B_artifacts/adaptive/diag_4b_arm_border_seed43/mechanism_diagnostics.json` `5234f8f7…3158`
- `05B_artifacts/adaptive/diag_4b_arm_border_seed44/mechanism_diagnostics.json` `0b0633e2…b307`

Checkpoint paths + SHAs (large .pth, not committed):
- `diag_4b_arm_border_seed43_best_network.pth` `ac2e99b2…0673`
- `diag_4b_arm_border_seed44_best_network.pth` `1fb3d830…149d`
- `diag_4b_arm_intensity_seed43_best_network.pth` `ba487905…301d`
- `diag_4b_arm_intensity_seed44_best_network.pth` `ba97a002…c9a9`

Reference lineage and transforms unchanged from STEP 4B (see `05B_CAA_MECHANISM_DIAGNOSTIC.md`).

## 11. Results table (paper-ready)

| Arm           | Seeds    |  n | Mean adaptive val AUC | Sample SD | Reduction vs ref | 95% CI of reduction | Interpretation |
| ------------- | -------- | -: | --------------------: | --------: | ---------------: | ------------------- | -------------- |
| H2 intensity  | 42,43,44 |  3 | 0.8334                | 0.0377    | +0.0048          | [−0.071, 0.080]      | no material ≥0.10 effect |
| H3 border/FOV | 42,43,44 |  3 | 0.8178                | 0.0228    | +0.0204          | [−0.023, 0.064]      | no material ≥0.10 effect |

95% CIs are Welch independent-arm intervals for the mean difference (reference n=10 vs arm n=3).

---

## Verdict

**STEP 4D E1 COMPLETE**

Three-seed adaptive validation estimates: H2 intensity mean 0.8334 (reduction +0.0048, Welch 95% CI [−0.071, +0.080]); H3 border mean 0.8178 (reduction +0.0204, CI [−0.023, +0.064]). Both uncertainty intervals exclude / strongly disfavor a material ≥ 0.10 effect. CAA method status unchanged (NO-GO); no TEST touched; no seed invented beyond authorized set {42, 43, 44}.