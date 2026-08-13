# 13 — P1 Capacity-Limited Anatomy Condition Test

**Step:** 7B
**Date:** 2026-08-13
**Status:** FAIL

## 1. Context

STEP 7A showed the exact 3-map anatomy condition leaks substantial patient identity:
anatomy-only AUC ≈ 0.833, pathology-only ≈ 0.608, joint ≈ 0.820. Therefore P0 exact-map
resynthesis was NO-GO. This step tests ONE frozen capacity reduction: P1 binary 64×64.

## 2. P1 transform (frozen, no sweep)

For every anatomy probability map (teacher output, 256×256):

1. Hard mask: `M = 1[p >= 0.5]`
2. Partition into non-overlapping 4×4 blocks
3. Block mean: `q = mean(block)`
4. Coarse hard mask: `M64 = 1[q >= 0.5]`

Final condition: shape **3×64×64, binary (uint8)**. No interpolation back to 256×256
for the attacker. No sweep over 128/32/16, thresholds, soft 64, signed-distance, morphology,
or random perturbation.

## 3. Data & provenance

- TRAIN + VALIDATION only; TEST never accessed.
- Source maps: exact STEP 7A cache (`research_agent/ibr_s1_condition_cache/`), teacher SHA-256
  `2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0` unchanged, source-ID
  provenance identical. Teacher NOT rerun.
- P1 cache: `research_agent/ibr_s1_condition_p1_cache/` → `train_p1.npy` (18,089), `val_p1.npy`
  (3,484), each (N, 3, 64, 64) uint8; `p1_meta.json`, `anatomy_retention.json`.
- Pair files: same frozen balanced files as STEP 7A (train 10,000 = 5k/5k; val 2,000 = 1k/1k).

## 4. Attacker

Reused STEP 7A pairwise design. Anatomy encoder = ResNet-18 class adapted for 3×64×64 input.
Pair operation ORDER-INVARIANT `|e1 - e2|` (verified). BCEWithLogitsLoss, Adam lr=1e-4,
bs=64, patience 5, min-validation-BCE checkpoint, max 60 epochs.

- ARM A — P1 coarse anatomy only
- ARM B — pathology only (REUSED exact STEP 7A records; identical inputs/code path)
- ARM C — P1 coarse anatomy + pathology (PRIMARY)

## 5. Seeds

All arms seeds 0, 1, 2. All runs numerically valid (finite weights/losses, updates observed,
checkpoints loadable). No replacement.

## 6. Primary statistic — P1 JOINT (ARM C)

n = 3

- mean best-validation AUC = **0.71152**
- sample SD (ddof=1) = 0.01094
- median = 0.71174
- min = 0.70048
- max = **0.72236**

Per-seed:

| seed | best val loss | best val AUC | epoch | termination |
|------|--------------|--------------|-------|-------------|
| 0 | 0.63032 | 0.70048 | 4 | early_stopping_patience_5 |
| 1 | 0.61988 | 0.71174 | 2 | early_stopping_patience_5 |
| 2 | 0.61432 | 0.72236 | 5 | early_stopping_patience_5 |

## 7. P1 decision gate

- STRONG PASS: mean ≤ 0.55 AND max ≤ 0.60
- PASS: mean ≤ 0.60 AND max ≤ 0.65
- PROMISING BUT INSUFFICIENT: 0.60 < mean ≤ 0.70
- FAIL: mean > 0.70 OR max > 0.75

P1 JOINT mean 0.71152 > 0.70 → **P1 CONDITION CAPACITY: FAIL**

## 8. Anatomy retention diagnostic (descriptive, NOT final metric)

M64 upsampled to 256×256 with nearest-neighbor ONLY, compared against original teacher hard
masks (validation split):

| structure | Dice | IoU | HD95 (px) |
|-----------|------|-----|-----------|
| Left Lung | 0.9725 | 0.9465 | 0.01 |
| Right Lung | 0.9757 | 0.9525 | 0.00 |
| Heart | 0.9755 | 0.9523 | 0.00 |
| MEAN | 0.9746 | 0.9504 | 0.01 |

P1 retains nearly all coarse anatomy (Dice ~0.975), which explains why identity capacity
remains high (~0.71): coarse lung/heart shape still encodes patient identity.

## 9. Descriptive comparison vs STEP 7A

| condition | anatomy AUC | pathology AUC | joint AUC |
|-----------|-------------|---------------|-----------|
| exact (7A) | 0.833 | 0.608 | 0.820 |
| P1 (7B) | 0.721 | 0.608 | 0.712 |

P1 reduces joint AUC 0.820 → 0.712, but the coarse binary 64×64 condition still carries
patient-identifying shape well above the PASS bands.

## 10. Load-bearing tests

All 8 PASS: (1) P1 transform frozen spec; (2) no TEST access; (3) no pixels/identifiers;
(4) pairs balanced; (5) order-invariant; (6) P1 maps match source IDs; (7) teacher SHA
matches 7A; (8) seed reproducible.

## 11. Decision logic outcome

FAIL → P1 is closed. Do NOT train a generator. The next possible primary variant is
**P2 — bounded stochastic anatomy normalization**, but P2 is NOT implemented automatically.
Evidence returned for scientific review.

## 12. Artifacts

- `research_agent/13_P1_CONDITION_CAPACITY.md` (this file)
- `research_agent/13_P1_CONDITION_CAPACITY_summary.json`
- P1 cache + provenance + retention: `research_agent/ibr_s1_condition_p1_cache/`
- Results + checkpoints: `research_agent/ibr_s1_condition_p1_capacity/`
- Code: `research_agent/ibr/build_p1_cache.py`, `research_agent/ibr/anatomy_retention.py`,
  `research_agent/ibr/train_p1_condition_capacity.py`, `research_agent/test_p1_condition_capacity.py`

---

P1 CONDITION CAPACITY: FAIL