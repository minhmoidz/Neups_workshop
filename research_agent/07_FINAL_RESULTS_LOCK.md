# STEP 5 — Final Results Lock

Frozen results and defensible interpretations only. No new hypothesis, no new method, no proposed experiment. Evidence assembly phase. Experiment phase closed at commit `b546155`.

---

## 1. Corrected baseline — official TEST privacy

Freshly retrained adaptive attacker, corrected operator (`I - G*u`), frozen corrected generator, official TEST pair, 10 restarts.

| Statistic | Value |
| --------- | ----- |
| Mean adaptive Re-ID AUC | **0.739194** |
| SD | 0.049847 |
| Median | 0.722183 |
| Max | 0.803662 |
| n | 10 (n_valid=10, n_near_chance=0, n_invalid=0) |

Per-seed TEST AUC: 0.7847, 0.7780, 0.7211, 0.6601, 0.7233, 0.8037, 0.7103, 0.7175, 0.8032, 0.6902.

## 2. Corrected baseline — official TEST utility

| Metric | Value |
| ------ | ----- |
| Classification mean AUC (14 labels) | **0.781227** |
| Segmentation Dice | **0.942717** |
| Segmentation IoU | **0.895454** |
| Segmentation HD95 (px) | **2.006120** |

## 3. Mechanism diagnostics — VALIDATION only (never TEST, never a privacy estimate)

### 3.1 Reference (validation)

Corrected baseline, fresh adaptive attacker, validation per-seed best AUC over 10 seeds: mean **0.8382**, sample SD (ddof=1) **0.0363**, median 0.8454, min 0.7927, max 0.8776.

### 3.2 H1 — spatial band isolation (n=1 per band, seed 42)

| Band | Adaptive val AUC | Classification AUC14 | Dice |
| ---- | ---------------- | -------------------- | ---- |
| Low-pass only | 0.8705 | 0.6210 | 0.8640 |
| High-pass only | 0.8659 | 0.6756 | 0.9111 |
| Reference | 0.8382 | 0.7938 | 0.9550 |

Both bands alone sustain adaptive recovery ≈ reference while diagnostic utility collapses.

### 3.3 H2 — intensity normalization (n=3, seeds 42/43/44)

Mean adaptive val AUC **0.8334** (SD 0.0377); reduction vs reference **0.0048**, 95% CI [−0.071, +0.080] (Welch independent-arm); bootstrap CI [−0.0379, +0.0426]. Utility preserved (classification AUC14 0.7947, Dice 0.9542).

### 3.4 H3 — border/FOV normalization (n=3, seeds 42/43/44)

Mean adaptive val AUC **0.8178** (SD 0.0228); reduction vs reference **0.0204**, 95% CI [−0.023, +0.064] (Welch independent-arm); bootstrap CI [−0.0081, +0.0493]. Utility preserved (classification AUC14 0.7928, Dice 0.9528).

### 3.5 Material-effect assessment

Predeclared material-effect threshold: **≥0.10 AUC reduction**. Both H2 and H3 upper confidence bounds lie below +0.10; the data strongly disfavor a material effect (one-sided mass above 0.10 ≈ 0.014 for H2, 0.002 for H3). **Neither diagnostic supports a MATERIAL ≥0.10 adaptive reduction.** This is not proof of zero effect.

## 4. Flow-spectrum summary (descriptive, VALIDATION)

| Quantity | Value |
| -------- | ----- |
| Cutoff (−3 dB) cycles per px | 0.01656 (4.24 cycles/image at 256) |
| Energy fraction above cutoff | 0.892 |
| Energy fraction below cutoff | 0.108 |
| Mean abs displacement (px) | 1.049 |
| Max abs displacement (px) | 1.803 |

## 5. Operator correction evidence

Corrected operator (`I - G*u`) removes the legacy border defect: at μ=0 the legacy operator drops 736/4096 (17.97%) source pixels vs 0 for corrected; identity-grid invariant 0.652 (legacy) vs 0 (corrected); border max grid diff 6.516e-01 vs ~0; real-batch proxy Re-ID 0.6995 (legacy) vs 0.7208 (corrected) — proxy, non-adaptive, not a privacy estimate.

## 6. Defensible interpretations (locked wording)

1. Identity remains highly recoverable from either coarse or fine spatial band alone.
2. Restricting either band causes substantial diagnostic utility loss while not reducing adaptive Re-ID.
3. Two utility-preserving single-channel normalizations did not produce a detectable material adaptive Re-ID reduction of ≥0.10 AUC.
4. These results are consistent with residual identity being redundantly recoverable and indicate that single-axis suppression is not a promising anonymization strategy under a freshly retrained attacker.

## 7. Non-comparable historical context (context only, no inferential comparison)

- Corrected baseline TEST mean 0.739194.
- Historical legacy-operator baseline ≈ 0.635 — different operator, validity criteria, and restart policy; non-comparable.
- External published figure ≈ 0.577 — different operator and protocol; aspirational target only.
- No inferential comparison among these three figures is permitted.

## 8. Status

- CAA: not implemented; method-level mechanism diagnostics reported; CAA status NO-GO. "CAA is statistically falsified" is a prohibited overclaim.
- STIS: not implemented.
- No TEST arm was accessed by any mechanism diagnostic (`test_touched = false` throughout 4A/4B/4D E1).
- No method/seed/experiment was invented beyond the authorized sets.

---

EXPERIMENTAL PHASE: CLOSED