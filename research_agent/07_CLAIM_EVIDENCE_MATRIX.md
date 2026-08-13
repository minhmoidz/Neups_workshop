# STEP 5 — Claim-to-Evidence Matrix

Every intended paper claim, mapped to the exact supporting experiment, split, statistic, limitation, safe wording, and prohibited overclaim. Frozen at commit `b546155`. Evidence assembly only; no new experiments.

---

## 1. Operator correction

- **CLAIM**: The anonymization operator was corrected (legacy `G*(I-u)` → corrected `I-G*u`), removing a border-sampling defect.
- **SUPPORTING EXPERIMENT**: STEP 1 operator review / repair (`00F_OPERATOR_REPAIR.md`, `00F2_OPERATOR_REVIEW_EVIDENCE.md`) — unit tests TEST 4/5/6/D/E, real-batch proxy Re-ID smoke.
- **SPLIT**: unit test + VALIDATION smoke (proxy, 2000 pairs). Not a privacy TEST.
- **STATISTIC**: legacy drops 736/4096 (17.97%) source pixels at μ=0 vs 0 for corrected; identity-grid invariant 0.652 (legacy) vs 0 (corrected); border max grid diff 6.516e-01 vs ~0; proxy Re-ID 0.6995 (legacy) vs 0.7208 (corrected).
- **LIMITATION**: proxy metric is non-adaptive feature-space measure (D7); not a privacy estimate.
- **SAFE WORDING**: "The legacy operator dropped ~18% of source pixels at the border; the corrected operator samples every source pixel and removes the grid-satellite artifact."
- **PROHIBITED OVERCLAIM**: "Correcting the operator made images more private."

## 2. Adaptive multi-restart leakage

- **CLAIM**: A freshly retrained adaptive attacker recovers substantial identity from corrected-baseline anonymized images (mean Re-ID AUC 0.739).
- **SUPPORTING EXPERIMENT**: STEP 3B/3D corrected baseline confirmatory arm, 10 attacker restarts, official TEST pair.
- **SPLIT**: TEST (official, n=10).
- **STATISTIC**: mean 0.739194, SD 0.049847, median 0.722183, max 0.803662; n_valid=10, n_near_chance=0, n_invalid=0.
- **LIMITATION**: single generator checkpoint; attacker retrained per restart with the same architecture/recipe.
- **SAFE WORDING**: "Under a freshly retrained attacker with the corrected operator, adaptive Re-ID reaches a mean AUC of 0.739."
- **PROHIBITED OVERCLAIM**: any claim that a specific percentage of identity "remains" (e.g. "74% of identity is leaked").

## 3. Large restart variance

- **CLAIM**: Adaptive Re-ID varies substantially across attacker restarts.
- **SUPPORTING EXPERIMENT**: same corrected baseline arm (STEP 3B/3D).
- **SPLIT**: TEST (official, n=10).
- **STATISTIC**: per-seed [0.7847, 0.7780, 0.7211, 0.6601, 0.7233, 0.8037, 0.7103, 0.7175, 0.8032, 0.6902]; SD 0.0498, min 0.6601, max 0.8037 (range ≈ 0.14).
- **LIMITATION**: n=10; large SD means a single-restart estimate is unreliable.
- **SAFE WORDING**: "Re-ID shows high variance across attacker restarts (SD ≈ 0.05, spread ≈ 0.14 AUC), motivating multi-restart reporting."
- **PROHIBITED OVERCLAIM**: presenting any single restart as the privacy figure.

## 4. Identity recoverable from either spatial band

- **CLAIM**: Both coarse (LP) and fine (HP) spatial bands alone retain enough identity for adaptive recovery.
- **SUPPORTING EXPERIMENT**: STEP 4A H1 band diagnostic (`05A_H1_BAND_DIAGNOSTIC.md`).
- **SPLIT**: VALIDATION DIAGNOSTIC (fresh adaptive attacker, 1 restart per band, seed 42).
- **STATISTIC**: LP adaptive val AUC 0.8705, HP 0.8659, Δ=0.0046 (≪ material threshold).
- **LIMITATION**: VALIDATION only, n=1 per band; not an official TEST privacy estimate.
- **SAFE WORDING**: "Identity remains highly recoverable from either coarse or fine spatial band alone."
- **PROHIBITED OVERCLAIM**: "Identity is proven to be distributed across all channels" or any "X% of identity remains".

## 5. Utility destruction from band isolation

- **CLAIM**: Isolating either band destroys diagnostic utility while not reducing adaptive Re-ID.
- **SUPPORTING EXPERIMENT**: STEP 4A utility-by-band (`05A_artifacts/utility_by_band.json`) + frozen triage.
- **SPLIT**: VALIDATION DIAGNOSTIC (frozen model landscape; not used in H1 rule).
- **STATISTIC**: classification AUC14 — reference 0.7938, LP 0.6210, HP 0.6756; Dice — reference 0.9550, LP 0.8640, HP 0.9111.
- **LIMITATION**: frozen-model landscape, validation images; not a TEST utility estimate.
- **SAFE WORDING**: "Restricting either band causes substantial diagnostic utility loss while not reducing adaptive Re-ID."
- **PROHIBITED OVERCLAIM**: treating these as TEST utility numbers or as a privacy outcome.

## 6. No material ≥0.10 effect for H2/H3

- **CLAIM**: Two utility-preserving single-channel normalizations did not produce a detectable material adaptive Re-ID reduction ≥0.10 AUC.
- **SUPPORTING EXPERIMENT**: STEP 4B (seed 42) + STEP 4D E1 extension (seeds 42/43/44), `06A_E1_H2H3_RESTART_EXTENSION.md`.
- **SPLIT**: VALIDATION DIAGNOSTIC (n=3 per arm vs reference n=10).
- **STATISTIC**: H2 intensity reduction 0.0048, 95% CI [−0.071, +0.080]; H3 border reduction 0.0204, 95% CI [−0.023, +0.064]; both upper bounds below +0.10.
- **LIMITATION**: VALIDATION only, small n; absence of a ≥0.10 effect is not proof of zero effect.
- **SAFE WORDING**: "Two utility-preserving single-channel normalizations did not produce a detectable material adaptive Re-ID reduction of ≥0.10 AUC."
- **PROHIBITED OVERCLAIM**: "Intensity carries no identity" / "Border/FOV carries no identity" / "CAA is statistically falsified".

## 7. Single-axis suppression conclusion

- **CLAIM**: Residual identity appears redundantly recoverable; single-axis suppression is not a promising anonymization strategy under a freshly retrained attacker.
- **SUPPORTING EXPERIMENT**: synthesis of STEP 4A (H1), STEP 4B/4D E1 (H2/H3) — all mechanism diagnostics.
- **SPLIT**: VALIDATION DIAGNOSTIC (all mechanism evidence).
- **STATISTIC**: LP/HP ≈ reference and no H2/H3 reduction reaches ≥0.10 (see matrix rows 4 and 6).
- **LIMITATION**: diagnostic-only; does not rule out other unexplored axes; no TEST evidence for any method.
- **SAFE WORDING**: "These results are consistent with residual identity being redundantly recoverable and indicate that single-axis suppression is not a promising anonymization strategy under a freshly retrained attacker."
- **PROHIBITED OVERCLAIM**: "Identity is proven to be distributed across all channels."

## 8. Historical comparator non-comparability

- **CLAIM**: Historical legacy (~0.635) and external published (~0.577) Re-ID figures are not comparable to the corrected baseline TEST.
- **SUPPORTING EXPERIMENT**: protocol documents (`01_ADAPTIVE_REID_PROTOCOL.md`, `01B_PROTOCOL_AMENDMENT.md`, `03D_CORRECTED_BASELINE_REID.md`).
- **SPLIT**: HISTORICAL / NON-COMPARABLE (legacy operator, different protocol / external paper).
- **STATISTIC**: corrected TEST mean 0.739194 vs historical legacy ≈0.635 vs published ≈0.577 (context only).
- **LIMITATION**: different operators, validity criteria, restart policies, and test protocols.
- **SAFE WORDING**: "Legacy and external figures were produced under different operators and protocols; the scientifically valid primary comparison is against the corrected reproduced baseline measured under this protocol."
- **PROHIBITED OVERCLAIM**: any inferential comparison among corrected 0.739, legacy ~0.635, published ~0.577.

---

## Summary of split classification

| Split | Rows | Usage |
| ----- | ---- | ----- |
| TEST | 2, 3 | primary privacy + utility estimates |
| VALIDATION DIAGNOSTIC | 4, 5, 6, 7 | mechanism / diagnostic only |
| unit test + VALIDATION smoke | 1 | operator correctness |
| HISTORICAL / NON-COMPARABLE | 8 | context only |

*Evidence assembly only. Experimental phase closed at `b546155`.*