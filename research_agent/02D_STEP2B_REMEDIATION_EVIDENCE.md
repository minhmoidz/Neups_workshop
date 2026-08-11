# 02D — STEP 2B.1 Remediation Evidence

> Scope: STEP 2B.1 scientific-review remediation of the STEP 2B implementation.
> No training, no real attacker runs, no C2/C4/method experiments. All evidence below is
> from the frozen authoritative protocol documents, the implementation regression suite,
> and live integration smoke checks.

---

## 1. Authoritative files are present and hash-verified

At the start of this pass the three authoritative frozen artifacts existed in the
workspace (untracked); their SHA-256 were recorded **before** any code change and
re-verified at the end:

| Artifact | SHA-256 |
|---|---|
| `research_agent/01_ADAPTIVE_REID_PROTOCOL.md` | `c3aa381ea2136d89d03f6a409a36edd2e1cb5d8f4986b3ae3846028f61cd6741` |
| `research_agent/01B_PROTOCOL_AMENDMENT.md` | `05fcc9fd4cb4181acd9f40603707cd259987a0d1d3750d0f3463fc1fe78b0865` |
| `research_agent/topk_frozen_list.csv` | `4ebb6e15786b7c25eb4220521e5d70cf03ceb8f7ca480581dd89ef3945b24d44` |

These are the **source of truth** for the implementation; none were modified or
regenerated during remediation. They are now tracked in Git (see §16).

## 2. Frozen Top-k list integrity (01B amendment §2, protocol §9.2)

Validated `topk_frozen_list.csv` (500 rows + header):

- 500 rows, **500 unique `patient_id`**;
- `gallery_image != probe_image` for all 500 pairs;
- `gallery_followup != probe_followup` for all 500 pairs (preferred distinct follow-up);
- every gallery/probe image resolves into the official NIH **test fold**
  (`image_pairs/test_list.txt`): 0 entries missing;
- columns: `patient_id, gallery_image, gallery_followup, probe_image, probe_followup`.

The runner now **reuses** this authoritative file (only deterministic rebuild as a
fallback if it is absent); it is never regenerated per arm.

## 3. BLOCKER 1 — validation accuracy used the wrong boundary

**Finding:** `validate_snn(..., return_metrics=True)` passed the Siamese head's **logits**
to an accuracy helper that thresholded at 0.5 on the raw-logit scale, i.e. the operating
point was silently wrong (protocol R-1: accuracy must use `p = sigmoid(logit) >= 0.5`,
≡ `logit >= 0.0`).

**Fix:** `adaptive_reid/metrics.py` adds `logistic_sigmoid`,
`compute_accuracy_from_logits`, `validation_metrics_from_logits`; `validate_snn` routes
through the logits-aware estimator. AUC (rank-based) is invariant to the sigmoid.

**Regression (`test_A2_accuracy_boundary_is_sigmoid_logit`):**
logits `[-3.0, -0.2, 0.2, 0.3, 0.6, 2.0]`, labels `[0,0,1,1,1,1]`:
- corrected accuracy = **1.0**;
- the old buggy rule (`logits > 0.5`) gives **4/6 = 0.6667**;
- exact boundary: `logit 0.0` → sigmoid 0.5 → positive class; `logit -1e-9` → negative.

## 4. BLOCKER 2 — double training eliminated

**Finding:** `main()` trained every seed inside `restarts.run_schedule(...)` **and then**
called `pipeline.stage_a_train_all(attempts)`, which re-invoked the training worker on
every seed — two training invocations per restart.

**Fix:** the redundant `stage_a_train_all` call was removed. The scheduler is the only
thing that trains; each seed is attempted exactly once, and the final health state drives
replacement. `pipeline.ArmPipeline` retains its stage abstraction for the tests.

**Regression (`test_D2_single_training_invocation_per_seed`):** a counting worker is
invoked 3 times during the schedule and **0 additional times** across stages B/C/D.

## 5. BLOCKER 2A — replacement policy on final state, invalid-only

**Fix (already correct in `restarts.run_schedule`, now regression-locked):**

- replacement triggers **only** on `NUMERICALLY_INVALID`;
- completed `VALID_NEAR_CHANCE` runs are **never** replaced;
- replacement seeds are strictly ascending after the initial set.

**Regression (`test_E2_confirmatory_ascending_replacement_on_invalid_only`):**
initial seeds 0–9 with seeds 5 and 8 invalid → attempts `[0..9, 10, 11]`, exactly 10 valid
runs, both replacements ascending.

## 6. BLOCKER 2B — idempotent reuse of completed runs

**Fix:** `run_adaptive_reid_arm.run_signature(seed, args, cfg, pair_hashes, protocol_documents,
frozen_artifacts)` produces a deterministic identity for one attempt (arm_id, seed,
transform mode, mu, stochastic lambda, checkpoint path, pair-file hashes, attacker
hyperparameters, protocol-doc hashes, frozen-artifact hashes, stub flag).
`reuse_completed_run(...)` returns the persisted `training_diagnostics.json` iff the
signature, `run_state.json`, and (for real runs) the checkpoint all exist and match.
`--force` is the explicit re-train override.

**Regression (`test_D3_idempotent_reuse_of_completed_run`):** empty dir → must train;
completed dir with matching signature → reused; changed protocol-doc hash → reuse
invalidated (must train again).

## 7. BLOCKER 3 — no fabricated test metrics

**Finding:** Stage E unconditionally emitted `{'auc': 0.55 + 0.02*(seed % 4)}` — a
fabricated number that could enter scientific summaries (R-11).

**Fix:**
- `require_real_test_eval(stub=False)` raises **`NotImplementedError`** — a real
  (non-stub) Stage E has no implementation and refuses to invent an AUC;
- `stub_test_metrics(seed)` returns only stub-mode metrics carrying `stub: true`,
  `synthetic: true`, `valid_for_scientific_reporting: false`;
- `summary.summarize_arm` excludes non-scientific metrics from mean/median/max/SD and
  flags `contains_stub_or_synthetic_metrics` / `scientific_summary_available`.

**Regression:** `test_E3_stub_markers_and_nonstub_not_implemented` (markers present;
non-stub raises); `test_J3_summary_refuses_synthetic_metrics` (all-synthetic → scientific
fields `None`; mixed → only the real value enters aggregates).

## 8. AMENDMENT 1 — sample SD `ddof=1`

**Fix:** `summary.summarize_arm` uses `np.std(ddof=1)` (sample SD) per protocol
§6.3/§12/R-10; `n < 2` returns `None` instead of NaN.

**Regression (`test_J2_sample_sd_uses_ddof1`):** with `[0.50, 0.52, 0.60, 0.70]`,
`std_test_auc == arr.std(ddof=1)` and `!= arr.std(ddof=0)`; single-element arm → `None`.

## 9. AMENDMENT 2 — protocol docs and frozen list in provenance + Git

**Fix:**
- `provenance.build_arm_provenance` accepts `protocol_documents` and `frozen_artifacts`
  (path → sha256) and writes both to `arm_provenance.json` (R-7/R-12);
- `run_adaptive_reid_arm.py` hashes the two protocol markdown files and the frozen
  Top-k CSV on every arm;
- the three authoritative files are committed to the repository.

**Regression:** `test_K2_provenance_protocol_and_frozen_hashes`;
end-to-end `test_Z_runner_stub_end_to_end` asserts both hashes are non-empty in the
provenance written by a real (stub) run.

## 10. R-9 — scientifically resolved, not implemented

The patient-clustered bootstrap proposal is **withdrawn** (01B amendment §6) because the
verification data are dyadic and negative pairs span two identities.
`adaptive_reid/bootstrap.py`:

- documents the ambiguity and the withdrawal;
- defines `PAIR_BOOTSTRAP_LABEL = "PAIR-SAMPLING DIAGNOSTIC — NOT PATIENT-LEVEL UNCERTAINTY"`;
- contains **no** patient-cluster resampling implementation (the former
  `PatientClusterResampler` class was removed);
- `report_R9_final_policy` records `R9_status = WITHDRAWN_PATIENT_CLUSTER_BOOTSTRAP`,
  the dyadic reason, restart-SD ddof=1, and the label.

**Regression:** `test_N_R9_final_policy` (status, label, absence of `PatientClusterResampler`).

## 11. Test infrastructure — CUDA skip guards

`test_grad_accum.py` (both CUDA-only tests) and the grad-accum sub-check in
`test_adaptive_reid_protocol.py` are guarded: on a host without CUDA they PASS by
skipping (plain script prints `[SKIP]`; under pytest `@pytest.mark.skipif` would apply);
on CUDA they run fully.

## 12. Full regression suite results

```
STEP 2B + STEP 2B.1 REMEDIATION TEST SUITE: PASS   (36 checks, incl. new A2/D2/D3/E2/E3/J2/J3/K2/N/Z)
STEP 1B REVIEW REMEDIATION: PASS                   (test_operator_repair.py)
ALL GRADIENT-ACCUMULATION REGRESSION TESTS PASSED  (test_grad_accum.py)
```

All new remediation regressions listed in §§3–10 pass in the same run.

## 13. Live integration — `validate_snn` backward compatibility

On GPU (venv, torch 2.7.0+cu128):
- the default path still returns a bare float validation loss;
- `return_metrics=True` returns `(loss, {'auc', 'accuracy'})` with the accuracy computed
  from the logits via the sigmoid boundary.

## 14. Runner stub end-to-end smoke

`python run_adaptive_reid_arm.py --mode confirmatory --arm_id arm_smoke --mu 0.01
--transform_mode legacy --out_dir ... --stub --stage a_e` produced:

- 12 attempts (10 initial + 2 replacements for the stub-infrastructure invalids),
  10 valid, 2 `NUMERICALLY_INVALID`, 0 near-chance;
- representative seed 1 (validation-median selection, seed tie-break);
- `arm_summary.json`: `contains_stub_or_synthetic_metrics: true`,
  `mean/max/median/std_test_auc: null`, `scientific_summary_available: false` — synthetic
  stub values never entered scientific fields;
- `arm_provenance.json`: non-empty `protocol_documents` (both files) and
  `frozen_artifacts` (Top-k CSV) hashes.

## 15. Files changed by the remediation

| File | Change |
|---|---|
| `adaptive_reid/metrics.py` | `logistic_sigmoid`, `compute_accuracy_from_logits`, `validation_metrics_from_logits` |
| `utils/utils.py` | `validate_snn` metrics route through the logits-aware estimator (accuracy boundary) |
| `adaptive_reid/summary.py` | `ddof=1` sample SD; synthetic/stub metrics excluded; `n<2` → `None` |
| `run_adaptive_reid_arm.py` | single-training flow (no re-train); idempotent reuse + `--force`; `NotImplementedError` for non-stub Stage E; stub markers; provenance hashes; authoritative Top-k reuse |
| `adaptive_reid/provenance.py` | `protocol_documents` + `frozen_artifacts` fields |
| `adaptive_reid/bootstrap.py` | R-9 withdrawn; pair-label constant; `PatientClusterResampler` removed |
| `adaptive_reid/__init__.py` | docstring reflects R-9 final policy |
| `test_adaptive_reid_protocol.py` | new regressions A2/D2/D3/E2/E3/J2/J3/K2/N/Z + CUDA-skip guard |
| `test_grad_accum.py` | CUDA skip guards |
| `research_agent/01_ADAPTIVE_REID_PROTOCOL.md` | tracked (unchanged content) |
| `research_agent/01B_PROTOCOL_AMENDMENT.md` | tracked (unchanged content) |
| `research_agent/topk_frozen_list.csv` | tracked (unchanged content) |
| `research_agent/02_STEP2B_ATTACKER_PROTOCOL_IMPLEMENTATION.md` | §16 updated, §18 remediation section |
| `research_agent/02D_STEP2B_REMEDIATION_EVIDENCE.md` | this file |
| `research_agent/STEP2B1_REMEDIATION.diff` | full diff artifact vs HEAD~ |

## 16. Repository-state verification

At the remediation commit the three authoritative files appear in `git ls-files`, the
worktree is clean, and the SHA-256 below match §1 (recomputed from the tracked blobs):

```
c3aa381ea2136d89d03f6a409a36edd2e1cb5d8f4986b3ae3846028f61cd6741  01_ADAPTIVE_REID_PROTOCOL.md
05fcc9fd4cb4181acd9f40603707cd259987a0d1d3750d0f3463fc1fe78b0865  01B_PROTOCOL_AMENDMENT.md
4ebb6e15786b7c25eb4220521e5d70cf03ceb8f7ca480581dd89ef3945b24d44  topk_frozen_list.csv
```

**STEP 2B.1 REMEDIATION: PASS**
