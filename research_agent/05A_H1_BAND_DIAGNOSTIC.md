# STEP 4A — Residual Identity Band Diagnostic (H1)

**Scale-Targeted Identity Suppression hypothesis test — TRAIN/VALIDATION only, corrected baseline frozen.**

## 1. Objective

Test hypothesis **H1** on the frozen corrected baseline:

> H1: after the corrected baseline, the residual identity re-identifiable from the anonymized training set is carried predominantly by coarse / low-frequency thoracic structure, while the ~1 px smooth deformation mainly affected finer-scale content. If true, a scale-targeted identity-suppression stage (STIS) that perturbs / destroys the coarse band should reduce residual Re-ID more than a fine-band perturbation of equal strength.

H1 is answered by a **diagnostic only** (no new anonymization operator is implemented, no paper privacy number is produced, and the official TEST split is never touched). Nor is the corrected generator, the attacker protocol, or the utility success criteria modified.

## 2. Design and absolute rules

- **Data protection**: TRAIN + VALIDATION splits only (`image_pairs_training_10000.txt`, `image_pairs_validation_2000.txt`, `chexnet/nih_labels.csv fold in {train,val}`). The official TEST split (`image_pairs_testing_5000.txt`, `fold == test`) is never read. `test_touched` is recorded `false` in every run.
- **Frozen baseline**: corrected generator `networks/corrected_baseline/generator_lowest_total_loss_corrected.pth`, `utils.deform`, `transform_mode='corrected'`, `mu=0.01`, `stochastic_lambda=0.0`, all bit-for-bit unchanged.
- **Frozen utility models**: DenseNet-121 classifier (`networks/pretrained_classifier.pth`, `mean_auc_14`) and UNetSeg (`archive/train_seg_unet/best.pth`, Dice / IoU / HD95), identical to STEP 3C.
- **Attacker protocol**: `adaptive_reid` SiameseNetwork (ResNet-50, 128-d head + `fc_end` head), Adam lr=1e-4, batch 16, early stopping patience 5, checkpoint = lowest validation loss, max 100 epochs. Evaluation metric = ROC-AUC over the 2,000 VALIDATION verification pairs (same as the confirmatory protocol).
- **Band transform**: single justified cutoff, no sweep (see PART A).
- **Decision rule (predeclared, not moved after results)**:
  - `AUC_lowpass − AUC_highpass ≥ 0.10` → **SUPPORTED**
  - `0.05 ≤ Δ < 0.10` → **AMBIGUOUS**
  - `Δ < 0.05` → **NOT SUPPORTED (falsified)**
  - Also require `AUC_lowpass` to remain clearly above chance (non-vacuous).

## 3. PART A — Band transforms (`research_agent/band.py`)

- **LOW-PASS** = exact Gaussian blur with `SIGMA_LP = 8.0` px (kernel 49, reflect padding).
- **HIGH-PASS** = exact complement `HP(x) = x − LP(x)`; verified `max|x − (LP+HP)| = 0.0` (bit-for-bit).
- **Justification**: the generator smooths its flow with `GaussianSmoothing(channels=2, kernel_size=9, sigma=2)`; `SIGMA_LP = 4 × σ_flow = 8 px` (~5.1 cycles/image at 256) robustly separates coarse thoracic anatomy (lung fields, heart silhouette ≈ 30–80 px structures) from the fine-scale band the ~1 px flow perturbs. No cutoff sweep.
- Reproducibility: `band.py` hash `fcba3510…5623`; deterministic, no border artifacts (flat-image HP ≈ 9e-7).

## 4. PART B — S0a: frozen-attacker triage (`diag_4a_frozen_triage.py`)

Frozen seed-4 attacker checkpoint (`archive/retrain_snn_seed4/…_best_network.pth`) evaluated on VALIDATION pairs under the three input conditions.

| band          | validation AUC |
|---------------|----------------|
| original      | 0.81692        |
| low_pass      | 0.70982        |
| high_pass     | 0.63589        |

`n = 2,000` pairs for each. **This is a FROZEN-ATTACKER DISTRIBUTION-SHIFT DIAGNOSTIC only** — it measures how the *fixed* seed-4 weights respond to a shift in input statistics, not what an attacker can recover after adaptation. It is **never** used to conclude H1. Sanity: original 0.81692 vs recorded seed-4 best val AUC 0.8249 (compatible). Artifact: `05A_artifacts/frozen_attacker_triage.json`.

## 5. PART C — S0b: adaptive band diagnostic (`diag_4a_adaptive_bands.py`)

Fresh attackers (seed 42, **one restart per band**, identical recipe and pair files as the confirmatory protocol, deformation applied BEFORE the band transform) trained on VALIDATION for each band.

| arm  | best val AUC | best AUC epoch | epochs run | termination |
|------|--------------|----------------|------------|-------------|
| LP   | **0.8705**   | 20             | 26         | early_stopping |
| HP   | **0.8659**   | 6              | 12         | early_stopping |

- `test_touched: false` in both per-arm `band_diagnostics.json`.
- Reference arm (unfiltered corrected baseline) **reused** from the existing compatible 03D per-seed records (10 seeds, same protocol/hash lineage): mean best val AUC **0.8382 ± 0.0344** (median 0.8454, min 0.7927, max 0.8776). No retraining for convenience.
- Both adaptive band attackers recover ≈ the same identity, at the **upper end of (or slightly above) the unfiltered reference** — i.e. neither band removal materially suppresses what an adapted attacker can learn.

## 6. PART D — Decision-rule application

- `AUC_lowpass = 0.870531`, `AUC_highpass = 0.865919`
- `Δ = AUC_lowpass − AUC_highpass = 0.0046`
- `0.0046 < 0.05` → **NOT SUPPORTED (H1 falsified)** by the predeclared rule.
- Non-vacuous check passed: `AUC_lowpass = 0.87` is far above chance (validated against near-chance baseline ≈ 0.55).

An adapted attacker recovers residual identity essentially equally from the coarse (LP) and fine (HP) bands alone. The claim that the residual is carried *predominantly* by coarse/low-frequency thoracic information is **not supported** under the rubric fixed before the experiment.

## 7. PART E — Utility by band (`diag_4a_utility_by_band.py`, VALIDATION, frozen models)

For the three conditions of the anonymized VALIDATION images, using the exact frozen STEP 3C utility models:

| band      | classification mean AUC-14 | segmentation Dice / IoU / HD95 px |
|-----------|---------------------------|-----------------------------------|
| original  | **0.7938**                | **0.9550 / 0.9172 / 1.307**       |
| low_pass  | 0.6210                    | 0.8640 / 0.7680 / 9.530           |
| high_pass | 0.6756                    | 0.9111 / 0.8428 / 5.023           |

`n = 10,816` val cases per band. Neither band alone preserves full frozen utility; low-passing removes the bright structural cues the raw-trained classifier/segmenter relied on, which is expected for a *frozen* model. This is a landscape measurement and does **not** enter the H1 rule.

## 8. PART F — Flow spectrum of the actual corrected baseline (`diag_4a_flow_spectrum.py`)

Actual applied displacement `disp_px = gaussian_σ2( μ·flow ) × 127.5` on 128 VALIDATION images:

- `mean |u| = 1.049 px` (std 0.0145), `max |u| = 1.80 px`, RMS = 1.125 px → confirms the "~1 px smooth deformation" premise empirically.
- Cutoff `f_c = sqrt(ln2)/(2π·σ_LP) = 0.01656 cyc/px` (= 4.24 cyc/image @256), the -3 dB point of the PART A Gaussian low-pass.
- **Displacement energy fraction below cutoff: 0.108; above cutoff: 0.892** → the flow energy is concentrated in the *fine* band.

Interpretation: the applied warp sits mostly in the fine-scale band; yet an adapted attacker still clears 0.87 from either band. Flow energy location therefore does **not** translate into a coarse-dominant residual-identity signature after adaptation.

## 9. PART G — Wording correction and limitations

- Reported quantities are **re-identification AUC** and **differences in AUC** (the "AUC-above-chance gap"). They are **not** and must not be read as "identity signal %", "privacy %", or any leak ratio.
- All band results are **diagnostic only** on TRAIN/VALIDATION; no privacy number and no paper claim derive from them.
- **Limitations**:
  - One adaptive restart per band (seed 42) with early-stopped checkpoints; the 03D reference is 10 seeds, so the band arms sit within, not statistically above, the reference spread.
  - A single Gaussian low-pass cutoff (σ=8 px) was used by design; the conclusion is specific to that partition of scale and could differ for other pass-bands.
  - Frozen utility numbers must not be compared across different fold geometry than STEP 3C (those were TEST; PART E is VALIDATION).

## 10. PART H — STIS go/no-go

- H1 is **NOT SUPPORTED** (falsified under the predeclared adaptive rule).
- Consequence for STIS: a scale-targeted identity-suppression stage whose premise is "residual identity lives predominantly in the coarse band" is **not indicated** by this diagnostic — an adapted attacker recovers ≈0.87 AUC from either band alone, i.e. removing/perturbing the coarse band alone should not be expected to suppress residual identity more than shaping the fine band.
- A go/no-go on STIS shaped as *coarse-only suppression* is therefore **NO-GO based on current evidence**; a future formulation would need a different rationale (e.g. attacking identity across both bands jointly) and a fresh predeclared protocol.

---

### Verdict

**STEP 4A H1 DIAGNOSTIC: FALSIFIED**

`Δ = AUC_lowpass(adaptive 0.8705) − AUC_highpass(adaptive 0.8659) = 0.0046 < 0.05` (predeclared threshold) with `AUC_lowpass` clearly above chance → H1 **NOT SUPPORTED**: residual identity after the corrected baseline is **not** predominantly coarse-carried.