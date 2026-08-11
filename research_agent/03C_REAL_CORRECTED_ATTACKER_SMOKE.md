# 03C — REAL Corrected-Baseline Attacker Smoke (STEP 3A2)

> Status: **COMPLETE — REAL SMOKE PASS** (2026-08-11).
> Exactly ONE real adaptive attacker (seed 0) trained and validated against the frozen
> corrected generator; Stages A–D only. No Stage E test evaluation, no test split access.
> Provenance, idempotency, and D-1 content pinning all verified.

---

## 1. Hard claims and verdict summary

| # | Claim | Status |
|---|-------|--------|
| 1 | Exactly ONE attacker seed (0) trained, no replacement seed, no restart schedule | **CONFIRMED** (§4) |
| 2 | Generator used is the frozen corrected checkpoint (path + SHA-256 pinned) | **CONFIRMED** (§4, §5) |
| 3 | `transform_mode='corrected'`, `mu=0.01`, `stochastic_lambda=0.0` reached the training process | **CONFIRMED** (§5) |
| 4 | Stages A–D only; **no Stage E**, no test metrics, test pair file never opened | **CONFIRMED** (§6) |
| 5 | Attacker run is real, numerically VALID, checkpoint loadable, weights changed | **CONFIRMED** (§5) |
| 6 | Re-running the arm reuses the completed run — zero additional training invocations | **CONFIRMED** (§7, Task 11) |
| 7 | Mutating the generator bytes changes the run signature and forces retraining (D-1) | **CONFIRMED** (§7, Task 12) |
| 8 | **Final verdict** | **STEP 3A2 REAL SMOKE: COMPLETE / PASS** |

**What was NOT done (and why):** the confirmatory 10-attacker arm was not launched (out of
scope for the smoke), and **no Stage E test evaluation was performed** — the smoke is
stages A–D only, and no test AUC is claimed anywhere.

---

## 2. Preconditions — clean tree, frozen artifact hashes

| Artifact | SHA-256 |
|---|---|
| `research_agent/01_ADAPTIVE_REID_PROTOCOL.md` | `c3aa381ea2136d89d03f6a409a36edd2e1cb5d8f4986b3ae3846028f61cd6741` |
| `research_agent/01B_PROTOCOL_AMENDMENT.md` | `05fcc9fd4cb4181acd9f40603707cd259987a0d1d3750d0f3463fc1fe78b0865` |
| `research_agent/topk_frozen_list.csv` | `4ebb6e15786b7c25eb4220521e5d70cf03ceb8f7ca480581dd89ef3945b24d44` |
| `image_pairs/image_pairs_training_10000.txt` | `3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268` |
| `image_pairs/image_pairs_validation_2000.txt` | `9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7` |
| `networks/corrected_baseline/generator_lowest_total_loss_corrected.pth` | `8a489eec036679f6775a2bf0b21fb112f1f968490a513b1f9b6dc012d4e384c2` |

- Working tree clean before execution (HEAD `ebc4b70`), after committing the STEP 3A1
  regression-test fix and the smoke driver.
- GPU: NVIDIA GeForce RTX 5070 Ti (100% util, ~10.5 GB VRAM during attacker training).
- `image_pairs/image_pairs_testing_5000.txt` was **NOT opened** by this execution path
  (see §6).

---

## 3. What ran

Single-seed smoke driver `run_3a2_smoke.py` reuses the REAL protocol driver code
(`run_adaptive_reid_arm.py`, `adaptive_reid.*`):

1. `run_signature` — deterministic arm identity incl. D-1 generator content hash.
2. `_train_once` — launches the real `retrain_SNN.py` (AgentSiameseNetwork, ResNet-50
   Siamese, Adam lr 1e-4, bs 16, 256×256, max 100 epochs, early-stopping patience 5).
3. `health.classify_run_health` — Stage B validity + near-chance.
4. `pipeline.ArmPipeline` — Stage C representative selection (validation-only),
   Stage D representative persist.
5. `write_arm_provenance` + frozen top-k list persist — provenance record.

Out dir: `archive/adaptive_reid_smoke_3a2/` (git-ignored; artifacts are content-addressed
by the hashes in this report and in `arm_provenance.json`).

---

## 4. Stage A — attacker training (seed 0 only)

| Item | Value |
|---|---|
| Attacker seeds attempted | `[0]` (exactly one; no schedule, no replacement) |
| Training invocations (first run) | **1** |
| Checkpoint | `archive/retrain_snn_seed0/retrain_snn_seed0_best_network.pth` |
| Epochs completed | 14 (early stopping, patience 5) |
| Termination reason | `early_stopping` |
| Best validation loss | `0.42763` (epoch 8) |
| Best validation AUC | `0.87666` (epoch 8) |
| Any NaN/Inf | false |
| Checkpoint exists / loadable | true / true |
| Weights changed from init | true |
| Run state | `VALID`, near-chance = false |

The legacy Aug-10 artifact that previously occupied the canonical attacker path was
stashed aside (untracked, backup under `/tmp/opencode/legacy_retrain_snn_seed0_bak`)
so the smoke is unambiguous; the training then wrote a fresh seed-0 attacker.

---

## 5. Generator binding (D-1) — what the attacker actually saw

`resolved_config.json` written by the training process records:

| Key | Value |
|---|---|
| `transform_mode` | `corrected` |
| `mu` | `0.01` |
| `stochastic_lambda` | `0.0` |
| `perturbation_model_file` | `networks/corrected_baseline/generator_lowest_total_loss_corrected.pth` |
| `generator_checkpoint_hash` | `8a489eec…384c2` (matches §2) |
| `evaluate_test_after_training` | `false` |
| `pair_train_path` | `image_pairs/image_pairs_training_10000.txt` |
| `pair_validation_path` | `image_pairs/image_pairs_validation_2000.txt` |
| `max_epochs` / `early_stopping` | `100` / `5` |
| `seed` | `0` |

The same generator hash is bound into the run signature, the training diagnostics, and
the arm provenance record. No config/hash references the test pair file.

---

## 6. Test-set lock — stages A–D only

| Check | Result |
|---|---|
| `evaluate_test_after_training` in attacker config | `false` → `test_loader = None`, `testing_evaluation()` never called |
| Test pair file opened by this execution path | **NO** |
| `pair_hashes` computed for | train + validation pair files only |
| `pair_test_hash` in arm provenance | `WITHHELD_TEST_SET_LOCK` (explicit marker, not a real digest) |
| Test metrics in diagnostics / summary | `None` / no `test_auc` fields |
| Stage E worker | disabled (`evaluate_test=None`) |

`arm_summary.json`: `scientific_summary_available=false`, `test_auc_values=[]`,
`max/mean/median/std_test_auc=null`, `contains_stub_or_synthetic_metrics=false`.

---

## 7. Protocol checks

### Task 11 — idempotent reuse (zero retraining on second invocation)

- 1st invocation: `training invoked: True`, count-file `= [0]` (one real training).
- 2nd invocation (no `--force`): `seed0 reused (no training): True`,
  `training invoked: False`, count-file unchanged `= [0]` → **0 additional training
  invocations**.

### Task 12 — generator content-pin mutation check (D-1)

A byte-flipped copy of the corrected checkpoint was created (different bytes,
SHA-256 `874c7ae8…` vs original `8a489eec…`):

- `run_signature` with the mutated checkpoint → **different** signature
  (`generator_checkpoint_hash` differs).
- `reuse_completed_run` against the completed run with the mutated signature → **refused**
  (`None`): a mutated generator can never silently reuse the run.
- `reuse_completed_run` with the true signature → **accepted** (record returned).

---

## 8. Provenance record

`archive/adaptive_reid_smoke_3a2/arm_provenance.json` (key fields):

- `arm_id`: `arm_smoke_3a2_corrected_seed0`
- `git_commit`: `ebc4b7094915e3af4ad423eb025dc1d2b93b8bae`
- `transform_mode`: `corrected`; `mu`: `0.01`; `stochastic_lambda`: `0.0`
- `generator_checkpoint_hash`: `8a489eec036679f6775a2bf0b21fb112f1f968490a513b1f9b6dc012d4e384c2`
- `attacker_architecture`: `ResNet-50 Siamese`
- `attacker_hyperparameters`: `{learning_rate: 0.0001, batch_size: 16, max_epochs: 100, early_stopping: 5}`
- `attacker_seeds_attempted`: `[0]`; `run_states`: `{"0": "VALID"}`; `near_chance_flags`: `{"0": false}`
- `representative_attacker_seed`: `0`
- `representative_selection_criterion`: best-validation-AUC closest to median, tie=smaller seed
- pair hashes: train `3c535eed…`, validation `9e33a081…`, test `WITHHELD_TEST_SET_LOCK`
- protocol hashes: 01 `c3aa381e…`, 01B `05fcc9fd…`; frozen top-k `4ebb6e15…`
- `schedule_name`: `smoke_single_seed`

Artifacts (all under `archive/adaptive_reid_smoke_3a2/`, git-ignored):
`arm_provenance.json`, `arm_summary.json`, `topk_frozen_list.csv`,
`runs/retrain_snn_seed0/{run_signature.json, run_state.json, training_diagnostics.json}`.

---

## 9. Regression suite status

| Test | Result |
|---|---|
| `test_grad_accum.py` | PASS |
| `test_adaptive_reid_protocol.py` (STEP 2B + 2B.1) | PASS |
| `test_operator_repair.py` (STEP 1B review) | PASS (after test12b updated to admit the intentional corrected config) |

---

## 10. Failures discovered and fixes

| # | Failure | Status | Fix |
|---|---------|--------|-----|
| F-1 | `test_operator_repair.py` TEST 12b failed preflight because the new STEP 3A1 config carries an intentional `transform_mode: "corrected"` | **FIXED** | test12b now counts corrected configs separately and asserts `resolve_transform_mode("corrected") == "corrected"`; rerun PASS. Committed in `ebc4b70`. |
| F-2 | Legacy Aug-10 artifact occupied the canonical attacker checkpoint path, making the smoke ambiguous | **FIXED** | Stashed untracked dir aside to `/tmp/opencode/legacy_retrain_snn_seed0_bak`; fresh attacker trained. |
| F-3 | `run_3a2_smoke.py` provenance writer requires `args.mode` | **FIXED** | Driver defines `--mode smoke_single_seed`; recorded in provenance. |

No scientific/infrastructure defect in the real training path was found.

---

## 11. Final verdict

**STEP 3A2 REAL SMOKE: COMPLETE / PASS**

Exactly one real adaptive attacker (seed 0) was trained and validated end-to-end against
the frozen corrected generator, with full provenance (D-1 content pinning, operator
provenance, idempotent reuse, test-set lock). The run is numerically VALID; no test AUC
was computed. Per the task scope, the run **stops here** — the confirmatory arm is a
separate step.
