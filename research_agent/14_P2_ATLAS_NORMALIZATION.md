# 14 — P2 Utility-Constrained Atlas Normalization Test

**Step:** 7C  
**Date:** 2026-08-14  
**Status:** FAIL  

---

## 1. Scientific Question & Design Principle

Can we explicitly remove patient-specific anatomical shape residuals while retaining sufficient coarse anatomy for segmentation?

Unlike P1 (which reduced spatial resolution to binary $64\times 64$), **P2 Capacity-Limited Atlas Normalization** shrinks each patient's signed Euclidean distance field $D_{\text{src}}$ toward a **TRAIN-only population atlas** $D_{\text{atlas}}$ under a strict macro-anatomy fidelity constraint:
$$D_\lambda = (1 - \lambda) D_{\text{src}} + \lambda D_{\text{atlas}}, \quad M_\lambda = \mathbf{1}[D_\lambda \ge 0]$$

### Key Design Principle
The condition transform is **strictly deterministic**. No stochastic noise is added during this capacity evaluation, ensuring we measure how much patient-identifying shape information remains in the retained anatomy channel itself.

---

## 2. TRAIN Population Atlas Provenance

- **Source Split:** TRAIN split ONLY ($N = 18,089$ images). VALIDATION images ($N = 3,484$) and TEST images ($N = 0$) did NOT contribute to atlas construction.
- **Transform:** Each $3\times 64\times 64$ binary P1 mask was converted to a signed distance field ($D > 0$ inside, $D < 0$ outside).
- **Atlas File:** `research_agent/ibr_s1_condition_p2_cache/train_atlas.npy` (shape $3\times 64\times 64$, float32).
- **Atlas SHA-256:** `91ed4eefec257c793ff646ec1f7e347ce6c0feaa3ae22ed8c5825cf8c39e0eb9`

---

## 3. $\lambda^*$ Calibration on TRAIN Retention ONLY

$\lambda^*$ was calibrated strictly on TRAIN macro-anatomy retention using high-precision bisection search (independent of Re-ID performance or validation split).

### Predeclared Target Constraints
- TRAIN macro mean Dice $\ge 0.940$
- TRAIN mean Dice for EACH structure $\ge 0.930$

### Selection Outcome
- **Selected $\lambda^*$:** **0.3400** (frozen)
- **TRAIN Retention at $\lambda^*$:**
  - Macro Mean Dice: **0.94000**
  - Left Lung Dice: **0.94247**
  - Right Lung Dice: **0.94466**
  - Heart Dice: **0.93287**

Once calibrated on TRAIN, $\lambda^* = 0.3400$ was **frozen** before transforming validation data or training identity attackers.

---

## 4. Validation Anatomy Retention

P2 validation masks ($M_{\lambda^*}$) upsampled $64 \to 256$ via nearest-neighbor interpolation and compared against original teacher hard masks ($M_{\text{teacher}} = \mathbf{1}[p \ge 0.5]$):

| Structure | Val Dice Mean (SD) | Val IoU Mean (SD) | Val HD95 Mean px (SD) |
| :--- | :---: | :---: | :---: |
| **Left Lung** | 0.9266 (0.0182) | 0.8658 (0.0298) | 2.54 px (1.12) |
| **Right Lung** | 0.9318 (0.0154) | 0.8742 (0.0256) | 2.62 px (1.15) |
| **Heart** | 0.9227 (0.0188) | 0.8598 (0.0305) | 2.78 px (1.21) |
| **MACRO MEAN** | **0.9270** | **0.8666** | **2.65 px** |

### Structural Requirement Evaluation
- Predeclared primary validation retention target: **Validation macro Dice $\ge 0.930$**.
- Observed validation macro Dice: **$0.9270 < 0.930$**.
- **Utility Status:** **FAIL** (Validation retention falls slightly below the $0.930$ threshold).

---

## 5. Attacker Architecture & Arms

Reused the exact STEP 7B pairwise condition-attacker family:
- **Anatomy Branch:** ResNet-18 backbone adapted for $3\times 64\times 64$ binary input $\to 128$-d embedding.
- **Pathology Branch:** MLP $14\to 64\to 64\to 128 \to 128$-d embedding.
- **Pair Operation:** Order-invariant $|e_1 - e_2|$ embedding distance.
- **Objective & Optimizer:** `BCEWithLogitsLoss`, Adam $\text{lr}=10^{-4}$, batch size 64, patience 5, minimum validation BCE checkpointing, max 60 epochs.

### Experimental Arms
- **ARM A:** P2 normalized anatomy only (seeds 0, 1, 2)
- **ARM B:** Pathology labels only (reused exact STEP 7A records; identical code path & inputs)
- **ARM C (PRIMARY):** P2 normalized anatomy + pathology (seeds 0, 1, 2)

---

## 6. Primary Statistic & Results

### P2 Per-Seed Attacker Results

| Arm | Seed | Best Val Loss | Best Val AUC | Best Epoch | Termination Reason | Provenance |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **ARM A** (P2 Anatomy) | 0 | 0.60897 | 0.72770 | 5 | early_stopping_patience_5 | Fresh run |
| **ARM A** (P2 Anatomy) | 1 | 0.61956 | 0.69973 | 2 | early_stopping_patience_5 | Fresh run |
| **ARM A** (P2 Anatomy) | 2 | 0.62300 | 0.69449 | 2 | early_stopping_patience_5 | Fresh run |
| **ARM B** (Pathology) | 0 | 0.67601 | 0.60359 | 16 | early_stopping_patience_5 | Reused 7A |
| **ARM B** (Pathology) | 1 | 0.67473 | 0.60889 | 8 | early_stopping_patience_5 | Reused 7A |
| **ARM B** (Pathology) | 2 | 0.67286 | 0.61113 | 22 | early_stopping_patience_5 | Reused 7A |
| **ARM C** (P2 Joint) | 0 | 0.62707 | 0.69794 | 4 | early_stopping_patience_5 | Fresh run |
| **ARM C** (P2 Joint) | 1 | 0.61913 | 0.69322 | 3 | early_stopping_patience_5 | Fresh run |
| **ARM C** (P2 Joint) | 2 | 0.62394 | 0.70284 | 4 | early_stopping_patience_5 | Fresh run |

### P2 Primary Statistics — ARM C (JOINT)
$n = 3$:
- **Mean Best-Validation AUC:** **0.69800**
- **Sample SD ($ddof=1$):** 0.00482
- **Median:** 0.69794
- **Min:** 0.69322
- **Max:** **0.70284**

### Incremental Statistic
- **Pathology Baseline Mean AUC:** $\text{pathology\_mean} = \mathbf{0.60787}$
- **Anatomy Incremental Leakage:**
  $$\Delta_{\text{anatomy}} = \text{joint\_mean} - \text{pathology\_mean} = 0.69800 - 0.60787 = \mathbf{0.09013}$$

---

## 7. Predeclared P2 Decision Gate Evaluation

### Frozen Predeclared Bands
- **STRONG PASS:** Val macro Dice $\ge 0.930$ AND Joint mean $\le 0.630$ AND Joint max $\le 0.660$ AND $\Delta_{\text{anatomy}} \le 0.020$.
- **PASS:** Val macro Dice $\ge 0.930$ AND Joint mean $\le 0.650$ AND Joint max $\le 0.680$ AND $\Delta_{\text{anatomy}} \le 0.030$.
- **PROMISING BUT INSUFFICIENT:** Val macro Dice $\ge 0.930$ AND Joint mean $\le 0.690$ AND $\Delta_{\text{anatomy}} \le 0.060$.
- **FAIL:** Any of:
  - Val macro Dice $< 0.930$ (observed: $0.9270 < 0.930$) $\to$ **FAIL**
  - Joint mean $> 0.690$ (observed: $0.69800 > 0.690$) $\to$ **FAIL**
  - $\Delta_{\text{anatomy}} > 0.060$ (observed: $0.09013 > 0.060$) $\to$ **FAIL**

All three FAIL criteria are triggered independently.

$$\mathbf{P2\ CONDITION\ CAPACITY:\ FAIL}$$

---

## 8. P2 Pre-Result Decision Amendment Rationale

In STEP 7A, pathology labels alone yielded $\approx 0.608$ AUC. Because pathology labels are medically necessary clinical inputs, the P2 bands explicitly benchmarked whether normalized anatomy contributes substantial additional identity ($\Delta_{\text{anatomy}}$). P2 atlas normalization reduced joint AUC from $0.820$ (STEP 7A exact) and $0.712$ (STEP 7B P1) down to $0.698$, but $\Delta_{\text{anatomy}} = 0.0901$ remains well above the $0.030$ PASS ceiling.

---

## 9. Load-Bearing Verification Tests

All 10 load-bearing verification tests passed cleanly ([test_p2_condition_capacity.py](file:///home/minhtt/Neups_workshop/research_agent/test_p2_condition_capacity.py)):
1. `test1_atlas_train_only`: Atlas built strictly from 18,089 TRAIN images.
2. `test2_val_never_affects_lambda`: Validation split never accessed during $\lambda^*$ calibration.
3. `test3_test_inaccessible`: TEST set strictly inaccessible and unreferenced.
4. `test4_signed_distance_deterministic`: Signed Euclidean distance transform bitwise deterministic.
5. `test5_lambda_reproducible`: Bisection search reproducibly yields frozen $\lambda^* = 0.3400$.
6. `test6_output_mask_binary_64x64`: P2 masks verified binary uint8 $3\times 64\times 64$.
7. `test7_no_pixels_no_identifiers`: Attacker tensors contain zero raw pixels, image features, or patient IDs.
8. `test8_pairs_balanced`: Pair files strictly balanced ($5k/5k$ train, $1k/1k$ val).
9. `test9_order_invariant`: Attacker distance head $|e_1 - e_2|$ order-invariant.
10. `test10_pathology_identical_to_7A`: Pathology labels and teacher SHA match STEP 7A provenance.

---

## 10. Summary Comparison across Primary Condition Family

| Step | Condition Variant | Macro Dice (Val) | Anatomy AUC | Pathology AUC | Joint AUC | $\Delta_{\text{anatomy}}$ | Decision Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **7A** | Exact 3-Map ($256\times 256$) | 1.0000 | 0.833 | 0.608 | 0.820 | 0.212 | **FAIL** |
| **7B** | P1 Binary Coarse ($64\times 64$) | 0.9746 | 0.721 | 0.608 | 0.712 | 0.104 | **FAIL** |
| **7C** | P2 Atlas Normalization ($\lambda^*=0.34$) | 0.9270 | 0.707 | 0.608 | 0.698 | 0.090 | **FAIL** |

---

## 11. Final Decision Logic & Research Route Forward

1. **P2 is CLOSED.**
2. **Do NOT train a generator** for P2 resynthesis.
3. **Do NOT invent P3** or test additional spatial resolution/threshold sweeps.
4. **Primary condition family is CLOSED.**
5. Per protocol, the research route switches to the predeclared **BACKUP**:
   $$\text{\textbf{Multi-Attacker Constrained Latent-Manifold Anonymization}}$$

---

## 12. Artifacts Produced

- [14_P2_ATLAS_NORMALIZATION.md](file:///home/minhtt/Neups_workshop/research_agent/14_P2_ATLAS_NORMALIZATION.md) (this report)
- [14_P2_ATLAS_NORMALIZATION_summary.json](file:///home/minhtt/Neups_workshop/research_agent/14_P2_ATLAS_NORMALIZATION_summary.json)
- `research_agent/ibr_s1_condition_p2_cache/` (TRAIN atlas, P2 masks, retention JSON, metadata)
- `research_agent/ibr_s1_condition_p2_capacity/` (checkpoints and per-seed log JSON)
- Code: `research_agent/ibr/build_p2_cache.py`, `research_agent/ibr/train_p2_condition_capacity.py`, `research_agent/test_p2_condition_capacity.py`

---

P2 CONDITION CAPACITY: FAIL
