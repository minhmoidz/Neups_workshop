# STEP 6C.1 — S1 Training-Schedule + Checkpoint-Selection Amendment

**Status:** amendment to STEP 6C. Unblocks the already-authorized ONE-SEED SANITY.
**Purpose:** supplies the two missing predeclared items (training schedule, checkpoint-selection rule) after the STEP 6C `CHECKPOINT SELECTION UNSPECIFIED` STOP.
**Commit:** `fbd92ee` (previous). This amendment does not change any other STEP 6C decision.

---

## 1. Training schedule (FIXED, exactly two stages)

### Stage A — Self-reconstruction pretrain

- **Duration:** exactly **20 epochs**.
- **Purpose:** initialize E/G to produce a valid radiograph reconstruction manifold.
- **Graph:** `x_self = G(z_id, z_med)`.
- **Optimize ONLY:** `L_rec(x_self, x)`.
- **NOT used:** donor swap objective, `L_path`, `L_anat`, `L_zid_pair`, `L_zmed_adv`.
- **No early stopping.**
- **Saves:** epoch-by-epoch logs; final Stage-A **epoch-20** checkpoint.
- **Stage-A checkpoint selection:** fixed LAST epoch = 20. No validation-based selection.

### Stage B — Full S1 training

- **Initialize from:** Stage-A epoch-20 checkpoint.
- **Duration:** exactly **30 epochs**.
- **Graph (locked):**
  ```
  (z_id, z_med) = E(x)
  x_self = G(z_id, z_med)
  x_anon = G(z_id_donor, z_med)
  ```
- **Objective (exactly the frozen S1 objective):**
  ```
  L_S1 = lambda_rec*L_rec + lambda_path*L_path + lambda_anat*L_anat
         + lambda_zid*L_zid_pair + lambda_adv*L_zmed_adv
  ```
- **Coefficients / optimizer config:** exactly as frozen in `09_IBR_PREIMPLEMENTATION_LOCK.md` and implemented/tested in STEP 6B (all λ=1.0, GRL λ=1.0; Adam lr=1e-4 for E/G/V; Adam lr=1e-4 for H_med; bs=16).
- **NOT changed:** lambda values, optimizer, LR, batch size, z_id dimension (128), donor protocol, architecture.
- **No early stopping.**

## 2. Checkpoint-selection rule

- **NO validation-based checkpoint selection** for the S1 sanity.
- **Frozen S1 checkpoint:** the **LAST NUMERICALLY VALID COMPLETED CHECKPOINT of Stage B epoch 30** (expected Stage-B epoch = 30).
- That checkpoint is frozen **BEFORE** training the adaptive attacker.
- **Do NOT choose by:** validation total loss, validation classification, validation Dice, z_id verifier AUC, z_med adversary AUC, reconstruction quality, frozen Re-ID proxy, or adaptive attacker result.
- **No retrospective checkpoint change allowed.**

## 3. Numerical failure

If any epoch has NaN/Inf weights, unrecoverable training crash, corrupted checkpoint, or invalid optimizer state → **STOP and classify NUMERICALLY INVALID**. No silent fallback to an earlier "good-looking" epoch. Report the failure before rerunning.

## 4. Monitoring does not select

Every epoch continues to log: validation total loss, all component losses, classification diagnostic, Dice diagnostic, z_id verifier metrics, z_med adversary metrics, donor leakage, reconstruction metrics. These are **diagnostics only**; they do not determine which epoch is frozen.

## 5. Utility collapse during training

Do not modify training because an intermediate epoch looks bad. Complete the fixed schedule unless numerically invalid. The **final epoch-30 checkpoint** is evaluated against the predeclared 1-seed utility gates. No trajectory-dependent manual intervention.

## 6. One-seed sanity budget (unchanged from STEP 6C)

After freezing Stage-B epoch 30, run **exactly ONE fresh canonical adaptive attacker A** (seed 42), TRAIN + VALIDATION only.

- Privacy: AUC ≤ 0.750 → PASS; 0.750–0.800 → WEAK; > 0.800 → FAIL.
- Utility: classification drop ≤ 0.030; Dice drop ≤ 0.025.
- No TEST.

## 7. Strict stop

After the ONE adaptive attacker seed-42 result: STOP. No seed 43/44, no hyperparameter changes, no V2, no TEST.

---

END EXACTLY ONE OF:

- `IBR S1 ONE-SEED SANITY: PASS`
- `IBR S1 ONE-SEED SANITY: WEAK`
- `IBR S1 ONE-SEED SANITY: FAIL`

---

S1 TRAINING SCHEDULE + CHECKPOINT SELECTION: LOCKED