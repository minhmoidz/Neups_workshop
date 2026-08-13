# 12 — Condition Identity Capacity

**Step:** 7A — source-condition identity-capacity test (highest-information experiment before resynthesis)
**Date:** 2026-08-13
**Status:** FAIL

## 1. Scientific question

Can a freshly trained patient verifier recover identity using ONLY the proposed source condition?

- Anatomy: Heart / Left Lung / Right Lung soft probability maps (3 × 256 × 256)
- Pathology: 14 NIH pathology labels
- NO image pixels, NO image features, NO source appearance latent.

This measures empirical patient-verification capacity of the condition under this attacker. It does NOT measure mutual information, information-theoretic privacy, or a formal lower bound on final Re-ID AUC.

## 2. Data & split discipline

- TRAIN + VALIDATION only; TEST never accessed (load-bearing TEST 1).
- Frozen pair files (tab-separated, `img1 img2 label`):
  - train: `image_pairs/image_pairs_training_10000.txt` — 10,000 rows, 5,000/5,000 balanced; 18,089 unique images; all train-fold.
  - val: `image_pairs/image_pairs_validation_2000.txt` — 2,000 rows, 1,000/1,000 balanced; 3,484 unique images; all val-fold.
- Pair-file SHA-256:
  - train `3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268`
  - val `9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7`
- Patient IDs derived from the CSV `Patient ID` column only; no filenames-as-features (TEST 3).

## 3. Condition cache provenance

- Teacher: `archive/train_seg_unet/best.pth`, SHA-256 `2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0` (hard-verified, TEST 7).
- `UNetSeg(in=1, out=3, init_features=16)`, 1,942,323 params, epoch 20, mean_dice 0.9548.
- Output: sigmoid soft probability maps, shape (B, 3, 256, 256); channel order `[Left Lung, Right Lung, Heart]`.
- Cache: `research_agent/ibr_s1_condition_cache/` → `train_maps.npy` (18,089, 3, 256, 256, float16), `train_images.json`, `val_maps.npy` (3,484), `val_images.json`, `cache_meta.json`.
- Cache meta records: source image id, fold, teacher SHA, map shape, dtype, build timestamp, commit `448ccb5`.
- No TEST cache. TEST never processed.

## 4. Attacker architecture (predeclared)

- ARM A — anatomy only: ResNet-18 (3-ch maps) → 128-d `e_map`; logit = MLP(|e1 − e2|).
- ARM B — pathology only: small MLP 14→64→64→128 → `e_path`; logit = MLP(|e1 − e2|).
- ARM C — JOINT (PRIMARY): concat(e_map, e_path) → 256-d; logit = MLP(|e1 − e2|).
- All pair ops ORDER-INVARIANT via |e1 − e2| (never ordered concat); verified (TEST 5).
- Objective: BCEWithLogitsLoss; Adam lr=1e-4; bs=64; early stopping patience 5; checkpoint = min validation BCE; max 60 epochs.
- No pixels/features/identifiers enter attacker tensors (TEST 2, TEST 3).

## 5. Seeds

- All arms run seeds 0, 1, 2 (JOINT 0/1/2 mandatory; A/B full set included).
- All 9 runs numerically valid (finite weights/losses, updates observed, checkpoints loadable); no replacement needed.

## 6. Primary statistic — ARM C (JOINT)

n = 3

- mean best-validation AUC = **0.82042**
- sample SD (ddof=1) = 0.01346
- median = 0.81689
- min = 0.80909
- max = **0.83530**

Per-seed:

| seed | best val loss | best val AUC | epoch | termination |
|------|--------------|--------------|-------|-------------|
| 0 | 0.50432 | 0.83530 | 9 | early_stopping_patience_5 |
| 1 | 0.53528 | 0.80909 | 8 | early_stopping_patience_5 |
| 2 | 0.52387 | 0.81689 | 7 | early_stopping_patience_5 |

## 7. Decision gate

- PASS requires: JOINT mean <= 0.60 AND max <= 0.65 AND all 3 runs valid.
- mean 0.82042 > 0.60; max 0.83530 > 0.65.
- → **CONDITION CAPACITY: FAIL**

## 8. Diagnostics (ablation)

- ARM A — anatomy only: mean AUC 0.83256 (SD 0.01647), min 0.82077, max 0.85137.
- ARM B — pathology only: mean AUC 0.60787 (SD 0.00387), min 0.60359, max 0.61113.
- ARM C — joint: mean AUC 0.82042 (SD 0.01346).

Interpretation: identity recoverability comes predominantly from ANATOMY SHAPE (maps alone ≈ joint ≈ 0.83); pathology labels contribute only a small lift over chance (~0.61). Stated as observed AUC values, NOT as percentages of identity information.

## 9. Load-bearing tests

All 8 PASS: (1) TEST inaccessible in dev mode; (2) no pixels in attacker tensors; (3) no filename/patient-id in features; (4) pairs balanced; (5) order-invariant |e1−e2|; (6) maps match source ids; (7) teacher SHA matches; (8) fixed seed reproduces init/order.

## 10. Implications (scientific review, not implementation)

- The exact proposed source condition (3 full-res anatomy maps + 14 labels) is too identity-revealing (JOINT ~0.82 verifier AUC) to justify building the generator unchanged.
- Per protocol: do NOT train the generator, do NOT return to IBR. Next allowed primary-family variant is **P1 — CAPACITY-LIMITED ANATOMY CONDITION** (e.g. spatial downsampling/quantization of anatomy maps preserving structural content). P1 is NOT implemented here; evidence returned for scientific review.

## 11. Artifacts

- `research_agent/12_CONDITION_IDENTITY_CAPACITY.md` (this file)
- `research_agent/12_CONDITION_IDENTITY_CAPACITY_summary.json`
- Cache: `research_agent/ibr_s1_condition_cache/` (maps + provenance)
- Checkpoints + per-run logs: `research_agent/ibr_s1_condition_capacity/`
- Code: `research_agent/ibr/build_condition_cache.py`, `research_agent/ibr/condition_attacker.py`, `research_agent/ibr/train_condition_capacity.py`, `research_agent/test_condition_capacity.py`

---

CONDITION CAPACITY: FAIL