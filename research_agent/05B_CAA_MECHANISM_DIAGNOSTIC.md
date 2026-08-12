# STEP 4B — CAA Mechanism Diagnostic

**Channel-Ablative Anonymization (CAA) hypothesis test — TRAIN/VALIDATION only, corrected baseline frozen.**

## 1. Objective

STEP 4A falsified the coarse-band hypothesis (H1; STIS NO-GO). This step activates the backup method family **CAA** and tests its two remaining mechanistic hypotheses using **diagnostic transformations plus adaptive TRAIN/VALIDATION attackers only**:

> **H2** — acquisition / intensity nuisance carries substantial identity.
> **H3** — border / field-of-view (FOV) information carries substantial identity.

CAA is **NOT authorized for implementation** in this step. No generator training, no TEST access, no new anonymization method — only one deterministic transform per hypothesis and one fresh adaptive attacker each (seed 42).

## 2. Protocol and absolute rules

- **Data lock**: TRAIN + VALIDATION only. Official Re-ID TEST pairs, utility TEST evaluation, `test_metrics.json`, Stage E, and Top-k are never touched. Every result is marked `DEVELOPMENT / MECHANISM DIAGNOSTIC — NOT A PAPER PRIVACY ESTIMATE`; `test_touched: false` is recorded in every run.
- **Reference arm (reused, not retrained)**: corrected baseline operating point — `transform_mode='corrected'`, `mu=0.01`, `stochastic_lambda=0.0`. Adaptive reference = existing compatible 03D/04A V LIDATION records: mean best val AUC **0.8382 ± 0.0344** (10 seeds). Frozen triage reference = seed-4 checkpoint evaluated in the same run (0.81729, n=2,000).
- **Attacker protocol**: unchanged A (SiameseNetwork ResNet-50, 128-d head; Adam lr=1e-4, batch 16, early stopping patience 5, checkpoint = lowest validation loss, max 100 epochs); ROC-AUC over the 2,000 VALIDATION pairs.
- **Decision rule (predeclared, per hypothesis)**: reduction vs compatible reference —
  - `reduction ≥ 0.10` → **SUPPORTED**
  - `0.05 ≤ reduction < 0.10` → **AMBIGUOUS**
  - `reduction < 0.05` → **NOT SUPPORTED**
  - thresholds never moved after results.
- **CAA go/no-go (PART 5)**: SUPPORTED requires **at least one** H2/H3 adaptive reduction ≥ 0.10 **AND** class AUC ≥ 0.765 and Dice ≥ 0.930 for that diagnostic. FALSIFIED if **both** reductions < 0.05.

## 3. PART 1 — H3 border/FOV diagnostic

**Canonical transform (frozen):** replace the **outermost 4-pixel image border** with that image's **median intensity**. `BORDER_WIDTH = 4` at 256×256 (the boundary region implicated by the historical legacy-operator border defect). No crop, no resize-after-mask, no central masking, no width tuning. Deterministic; verified interior rows/cols [4:252] unchanged and every border pixel equal to the per-image median.

### H3-A — frozen-attacker triage (VALIDATION, seed-4 checkpoint)

| condition | validation AUC |
|-----------|----------------|
| reference          | 0.81729 |
| border-normalized  | 0.81072 |
| **Δ**              | **−0.0066** |

FROZEN-ATTACKER DISTRIBUTION-SHIFT TRIAGE — diagnostic only, **not** used to decide H3.

### H3-B — one fresh adaptive attacker (seed 42)

`best_validation_auc = 0.79380` (epoch 8/11, early stopping; `test_touched: false`).

**Adaptive reduction vs reference (0.8382) = +0.0444 → H3 NOT SUPPORTED.**

## 4. PART 2 — H2 acquisition/intensity diagnostic

**Canonical transform (frozen):** robust per-image affine intensity normalization — `p1`, `p99` percentiles; `X_norm = clip((X − p1)/max(p99 − p1, 1e-6), 0, 1)`, mapped back into the pipeline's expected numeric range. No CLAHE, no histogram equalization, no noise/blur, no geometry change, no percentile sweep. Deterministic; verified pixel-ordering preserving, outlier-robust, range-preserving.

### H2-A — frozen-attacker triage (VALIDATION, seed-4 checkpoint)

| condition | validation AUC |
|-----------|----------------|
| reference          | 0.81729 |
| intensity-normalized | 0.81699 |
| **Δ**              | **−0.0003** |

FROZEN-ATTACKER DISTRIBUTION-SHIFT TRIAGE — diagnostic only, **not** used to decide H2.

### H2-B — one fresh adaptive attacker (seed 42)

`best_validation_auc = 0.80750` (epoch 7/13, early stopping; `test_touched: false`).

**Adaptive reduction vs reference (0.8382) = +0.0307 → H2 NOT SUPPORTED.**

## 5. PART 3 — VALIDATION utility sanity (frozen models)

Development diagnostics (no TEST non-inferiority statistics). Frozen screening gates: class AUC < 0.765 or Dice < 0.930 → gross collapse.

| condition | classification mean AUC-14 | segmentation mean Dice | gross collapse |
|-----------|---------------------------|------------------------|----------------|
| reference | 0.7938                     | 0.9550                 | no             |
| border    | 0.7928                     | 0.9528                 | no             |
| intensity | 0.7947                     | 0.9542                 | no             |

All conditions remain above the screening gates — neither diagnostic collapses utility.

## 6. PART 4 — Interpretation

| Hypothesis | adaptive val AUC | reduction vs ref | utility intact | verdict |
|------------|-----------------:|-----------------:|:--------------:|---------|
| H3 border/FOV | 0.79380 | +0.0444 | yes | **NOT SUPPORTED** |
| H2 intensity  | 0.80750 | +0.0307 | yes | **NOT SUPPORTED** |

Neither border/FOV nor acquisition/intensity dominates residual identity under the canonical diagnostics: an adapted attacker re-learns identity from border-normalized and intensity-normalized corrected-baseline images at essentially the reference level. Together with STEP 4A (LP ≈ HP), this pattern is consistent with **redundant / distributed identity across channels**:

**CASE D — both adaptive reductions < 0.05 → CAA FALSIFIED; H4 (distributed/redundant identity) gains support.**

Per protocol, individual weak results are NOT combined to rescue CAA.

## 7. PART 5 — CAA go/no-go

- Best adaptive reduction = **0.0444** (H3) — below the 0.10 support requirement (H2: 0.0307).
- **Both mechanisms < 0.05** → CAA is **FALSIFIED** under the predeclared rubric (no vanishing utility confound: class/Dice well above gates for both).
- No implementation. Method formulation remains un-authorized.

## 8. PART 6 — Boundaries

Even if a mechanism had shown support, **no CAA implementation** would occur here; only the mechanism would be identified for later formulation. STOP is respected: no CAA code, no generator retraining, no TEST touch, no new method family.

## 9. Artifacts and provenance

Scripts (hashed):
- `research_agent/caa_transforms.py` `8c56b8c2…0711b` (border + intensity canonical transforms)
- `research_agent/diag_4b_frozen_triage.py` `e64f9a2e…9782` (H3-A/H2-A)
- `research_agent/diag_4b_adaptive.py` `0f863edf…1452` (H3-B/H2-B)
- `research_agent/diag_4b_utility_sanity.py` `a1c95596…abe6b` (PART 3)

Evidence (hashed):
- `05B_artifacts/frozen_triage_border.json` `aee9d797…331b8`
- `05B_artifacts/frozen_triage_intensity.json` `ced34b0f…3480`
- `05B_artifacts/adaptive/diag_4b_arm_border_seed42/mechanism_diagnostics.json` `9af65421…dc9b9`
- `05B_artifacts/adaptive/diag_4b_arm_intensity_seed42/mechanism_diagnostics.json` `27495762…feb3e`
- `05B_artifacts/utility_sanity.json` `45bc10a9…5696`

Reference lineage: 03D/04A per-seed best val AUC mean 0.8382 ± 0.0344 (seeds 0–9); frozen generator SHA-256 `8a489eec…84c2`; pair files `image_pairs_training_10000.txt` / `image_pairs_validation_2000.txt`; attacker seed 4 (triage) / 42 (adaptive).

Note: two pre-existing diagnostic `.pth` checkpoints (~95 MB each) are left on disk (`05B_artifacts/adaptive/diag_4b_arm_*/…_best_network.pth`); paths + SHAs recorded, not committed.
- `05B_artifacts/adaptive/diag_4b_arm_border_seed42/diag_4b_arm_border_seed42_best_network.pth` `c06a6e90…3273`
- `05B_artifacts/adaptive/diag_4b_arm_intensity_seed42/diag_4b_arm_intensity_seed42_best_network.pth` `05cdf93a…1dbc`

## 10. Summary table

| Arm           | Frozen attacker AUC | Adaptive val AUC | Δ adaptive AUC | Class AUC | Dice | Verdict |
| ------------- | ------------------: | ---------------: | -------------: | --------: | ---: | ------- |
| reference     |             0.81729 |       0.8382*    |              — |   0.7938 | 0.9550 | —       |
| H2 intensity  |             0.81699 |       0.8075     |           +0.031 |   0.7947 | 0.9542 | NOT SUPPORTED |
| H3 border/FOV |             0.81072 |       0.7938     |           +0.044 |   0.7928 | 0.9528 | NOT SUPPORTED |

*reference asterisked: reused multi-seed 03D/04A VALIDATION record (mean of 10 seeds), not an adaptive retrain.

---

### Verdict

**STEP 4B CAA DIAGNOSTIC: FALSIFIED**

Adaptive reductions vs reference: H3 border +0.0444 (< 0.05), H2 intensity +0.0307 (< 0.05), with utility intact → **CASE D**: residual identity is distributed / redundant across channels rather than localized in a border/FOV or intensity channel.