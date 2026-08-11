# 03D — Corrected Canonical Baseline: Confirmatory Adaptive Re-ID Evaluation (STEP 3B)

> Status: **COMPLETE — PASS** (2026-08-11). First confirmatory privacy measurement of the
> corrected PriCheXy-Net canonical baseline under the frozen protocol.

---

# 1. Objective

Establish `B_corrected` — the confirmatory adaptive Re-ID AUC of the frozen corrected
PriCheXy-Net baseline — under the authoritative protocol
(`01_ADAPTIVE_REID_PROTOCOL.md`, `01B_PROTOCOL_AMENDMENT.md`). 10 numerically valid
completed attacker-A restarts, official test evaluated exactly once per valid run,
representative frozen before TEST. No method development, no C2/C4, no tuning.

# 2. Frozen corrected generator

- Path: `networks/corrected_baseline/generator_lowest_total_loss_corrected.pth`
- SHA-256: `8a489eec036679f6775a2bf0b21fb112f1f968490a513b1f9b6dc012d4e384c2`
- Resolved from `research_agent/03B_generator_provenance_baseline_corrected.json`; actual
  SHA matches frozen provenance exactly. Loadable. `transform_mode=corrected`,
  `mu=0.01`, `stochastic_lambda=0.0`.
- Config hash (`config_anonymization_baseline_corrected.json`) `f5b83f4a…` matches
  provenance; train/val pair hashes match.
- **Immutability:** SHA-256 recomputed before and after STEP 3B — identical
  (`8a489eec…`). Generator was never modified, retrained, fine-tuned, or overwritten.

# 3. Git/environment

- `pwd`: `/home/minhtt/Neups_workshop`
- Branch: `main`; HEAD before run: `1c54435846af6f0ca30ef668464032e727b5aed6`
  (same as the git commit recorded in every arm provenance record)
- Working tree clean before execution.
- GPU: NVIDIA GeForce RTX 5070 Ti (100% util, ~10.5 GB VRAM during attacker training).
- Python: `.venv` (cpython 3.10).

# 4. Confirmatory attacker protocol

Canonical Attacker A (frozen, unchanged):
- ResNet-50 Siamese, ImageNet init, shared weights
- BCEWithLogitsLoss on balanced pairs
- Adam lr = 1e-4, batch size 16, image size 256
- Max 100 epochs, early stopping patience 5 on validation loss
- Checkpoint = lowest validation loss; ties to earlier epoch
- Transform on-the-fly with `transform_mode=corrected`, `mu=0.01`, `stochastic_lambda=0.0`

Schedule (frozen): target 10 numerically VALID restarts; initial seeds 0–9; replacements
10–14 ascending **only** for `NUMERICALLY_INVALID`; max 15 attempts. Near-chance runs are
never replaced.

# 5. Attempted restart table

| Seed | Val epochs | Termination | Best val loss | Best val AUC | State |
|---:|---:|---|---:|---:|---|
| 0 | 14 | early_stopping | 0.4436 | 0.8706 | VALID |
| 1 | 12 | early_stopping | 0.4341 | 0.8743 | VALID |
| 2 | 12 | early_stopping | 0.5158 | 0.8054 | VALID |
| 3 | 14 | early_stopping | 0.5283 | 0.7939 | VALID |
| 4 | 13 | early_stopping | 0.5108 | 0.8249 | VALID |
| 5 | 13 | early_stopping | 0.4294 | 0.8691 | VALID |
| 6 | 13 | early_stopping | 0.4517 | 0.8659 | VALID |
| 7 | 11 | early_stopping | 0.5178 | 0.8077 | VALID |
| 8 | 13 | early_stopping | 0.4200 | 0.8776 | VALID |
| 9 | 12 | early_stopping | 0.5260 | 0.7927 | VALID |

10 attempts, 10 VALID, 0 `NUMERICALLY_INVALID`, 0 replacements used.

# 6. Execution-health results

- `n_attempted = 10`, `n_valid = 10`, `n_numerically_invalid = 0`.
- No NaN/Inf, no OOM, no crash, no missing/corrupt checkpoint, no infra termination in
  any run. Every run terminated via `early_stopping`.
- Each run's `training_diagnostics.json` (validation-only, schema-checked, no test fields)
  persisted under `archive/adaptive_reid_baseline_corrected_confirmatory/runs/retrain_snn_seedN/`.

# 7. Near-chance results

- `n_near_chance = 0` of 10. No seed met `best_val_loss >= 0.68 AND best_val_auc <= 0.55`.
- Not applicable to interpretation: no escalation (see §14).

# 8. Representative selection before TEST

Selection on **validation stats only**, before any official test evaluation
(enforced by Stage D freeze):
- Best validation AUC across the 10 valid runs: sorted
  {0.7927, 0.7939, 0.8054, 0.8077, 0.8249, 0.8659, 0.8691, 0.8706, 0.8743, 0.8776}.
- Median best validation AUC = (0.8249 + 0.8659)/2 = **0.8454**.
- Closest = seed **4** (0.8249, |Δ| = 0.0205; seed 6 at 0.8659 is also Δ=0.0205 → tie
  broken by **lower seed** → 4).
- Representative frozen at `2026-08-11T16:25:21Z` (Stage D provenance
  `arm_provenance_stageD.json`, `pair_test_hash=WITHHELD_TEST_SET_LOCK`).

# 9. Official TEST firewall/order evidence

- Stages A–D ran with `pair_hashes` = train+val only; the official test pair file was
  **not opened** until Stage E.
- Stage D provenance (`arm_provenance_stageD.json`) records the representative with
  `pair_test_hash = WITHHELD_TEST_SET_LOCK` before any test access.
- Stage E then opened `image_pairs/image_pairs_testing_5000.txt` (frozen,
  SHA-256 `87e52830…`) and evaluated each VALID attacker exactly once.
- Final `arm_provenance.json` records the real `pair_test_hash` and the
  `representative_selection_timestamp` proving representative was fixed before TEST.
- No test-derived quantity entered training, checkpointing, seed handling, validity, or
  representative selection.

# 10. Per-seed test AUC

| Seed | Test AUC |
|---:|---:|
| 0 | 0.7847 |
| 1 | 0.7780 |
| 2 | 0.7211 |
| 3 | 0.6601 |
| 4 | 0.7233 |
| 5 | 0.8037 |
| 6 | 0.7103 |
| 7 | 0.7175 |
| 8 | 0.8032 |
| 9 | 0.6902 |

All AUC ∈ [0,1], exactly 10 values, no missing result, no duplicate evaluation, each
`test_metrics.json` marks `valid_for_scientific_reporting=True`, `stub=False`,
`synthetic=False`, `evaluated_test=True` in run state.

# 11. Mean / sample SD / median / maximum

Computed over **all 10** numerically valid completed attackers (near-chance runs included;
here none):

| Estimand | Value |
|---|---:|
| **Mean Re-ID AUC** | **0.7392** |
| Sample SD (ddof=1) | 0.0498 |
| Median Re-ID AUC | 0.7222 |
| Max Re-ID AUC (worst case) | 0.8037 |
| Min Re-ID AUC | 0.6601 |

No "successful-attacker-only" mean computed.

# 12. Pair-sampling diagnostic

Deferred to STEP 3C / paper-table construction (protocol §12.2 and amendment §6; the
pair-level bootstrap is a secondary diagnostic, labeled
`PAIR-SAMPLING DIAGNOSTIC — NOT PATIENT-LEVEL UNCERTAINTY`, and is not needed for the
canonical STEP 3B estimands). Patient-cluster bootstrap is **not** implemented.

# 13. Historical-context comparison

Descriptive only — NOT inferential (operator, restart protocol, validity handling, and
attack implementation differ from the sources):

| Context | Re-ID AUC |
|---|---:|
| **B_corrected (this arm, STEP 3B)** | **0.739 ± 0.050 (median 0.722, max 0.804)** |
| Historical legacy baseline (approx.) | ~0.635 |
| Original PriCheXy-Net paper (approx.) | ~0.577 |

No statistical test was run against either historical figure. STEP 3B establishes
`B_corrected`; method comparison begins only after the full baseline profile exists.

# 14. Escalation status

- Near-chance trigger: `n_near_chance >= 5 of 10`. Observed: **0 of 10**.
- **ESCALATION NOT TRIGGERED.** No Attacker B and no extended-budget Attacker A required.

# 15. Integrity checks

| Check | Result |
|---|---:|
| All 10 test AUC ∈ [0,1] | PASS |
| Exactly 10 valid test AUCs | PASS |
| No missing result (10/10 test_metrics.json) | PASS |
| No duplicate test evaluation (evaluated_test=True, 1 per seed) | PASS |
| Generator SHA-256 identical across runs | PASS (`8a489eec…`) |
| Pair hashes identical across runs (train/val/test) | PASS |
| No stub/synthetic markers | PASS (`contains_stub_or_synthetic_metrics=false`) |
| `valid_for_scientific_reporting` | true for all 10 |
| Frozen generator unchanged after run (immutability) | PASS |

# 16. Runtime/GPU

- Attacker training: ~5 h 20 m wall (17:50 → 23:10), 10 seeds ≈ 26–35 min/seed,
  early-stopped at 11–14 epochs of 100.
- Stage E test evaluation: ~10 min for all 10 runs (5 000 pairs each).
- GPU: RTX 5070 Ti, ~10.5 GB VRAM, 100% util during training.

# 17. Problems/deviations

| # | Problem | Resolution |
|---|---------|------------|
| P-1 | Stock runner `run_adaptive_reid_arm.py` Stage E only emits stub metrics (`require_real_test_eval` raises for real runs) | Implemented real Stage E in `run_3b_confirmatory.py` (loads frozen test pairs, `test_snn`, sklearn ROC-AUC) |
| P-2 | First Stage E run wrote `test_auc` but `summarize_arm` reads `auc` → summary empty | Fixed driver to emit both keys; re-ran Stage E idempotently (10 evals reused, not recomputed) |
| P-3 | Legacy seed-0 canonical path from STEP 3A2 smoke | Confirmatory arm uses a distinct `arm_id` → different run signature → fresh training of seed 0 (documented decision; see below) |

**Seed-0 reuse decision (protocol "IMPORTANT — STEP 3A2 SEED 0"):** the STEP 3A2 smoke run
is treated as **integration smoke only**; the confirmatory arm runs a **fresh** seed 0.
This is protocol-preferred, and it is also mechanically enforced: the confirmatory arm
uses `arm_id=baseline_corrected_confirmatory` (vs smoke `arm_smoke_3a2_corrected_seed0`),
so `run_signature` differs and `reuse_completed_run` cannot reuse the smoke run. The
decision was not based on validation or test AUC.

# 18. STEP 3B verdict

PASS criteria:

1. Exact frozen corrected generator — **PASS** (path + SHA-256, immutability verified).
2. 10 numerically valid completed attackers — **PASS** (10/10).
3. Replacement only for objective execution failure — **PASS** (0 replacements, 0 invalid).
4. Representative frozen before TEST — **PASS** (seed 4, Stage D provenance, test hash
   withheld until Stage E).
5. Official TEST evaluated exactly once per valid attacker — **PASS**.
6. No test-derived selection — **PASS** (validation-only Stage C; enforced by Stage D).
7. Near-chance attackers retained — **PASS** (0 present; policy would retain any).
8. Mean/sample-SD/median/max over all 10 valid runs — **PASS** (0.7392 / 0.0498 / 0.7222 / 0.8037).
9. No synthetic/stub metric — **PASS**.
10. Complete provenance — **PASS** (arm_id, git commit, generator path+hash, transform
    settings, protocol/frozen/pair hashes, attempted/valid/invalid/near-chance seeds,
    attacker checkpoints, representative, timestamps).

**STEP 3B CORRECTED BASELINE RE-ID: PASS**
