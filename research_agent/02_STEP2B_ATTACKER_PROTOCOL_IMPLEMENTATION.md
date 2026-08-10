# 02 — STEP 2B: Attacker Protocol Implementation

> Status: **DONE (PASS WITH R-9 BLOCKED)** — 2026-08-11.
> Infrastructure + tests only. No training / no 10-attacker runs / no method experiments.

---

## 1. Protocol documents implemented

- `research_agent/01_ADAPTIVE_REID_PROTOCOL.md` — **NOT PRESENT in workspace/remote/history**
  at implementation time (checked worktree, `git log --all`, `origin/main`).
- `research_agent/01B_PROTOCOL_AMENDMENT.md` — same.

Per operator decision (recorded in the conversation), the **STEP 2B task specification
itself** was used as the authoritative frozen protocol: it contains every constant,
schedule, API constraint, and stage rule that the missing documents reference. All
cross-references below map to the task's Part numbers. R-9 was resolved exactly as the
task mandates ("do not guess"): the pair schema makes patient-cluster bootstrap
scientifically ambiguous, so R-9 is **BLOCKED FOR SCIENTIFIC CLARIFICATION** and does
not block the rest of STEP 2B.

## 2. Files changed

| File | Role |
|---|---|
| `utils/utils.py` | `validate_snn(..., return_metrics=False)` — backward-compatible validation metrics |
| `adaptive_reid/__init__.py` | package |
| `adaptive_reid/constants.py` | named constants (CHANCE_LOSS, 0.68/0.55, schedules, Top-k) |
| `adaptive_reid/metrics.py` | pure AUC/accuracy helpers (Part 1) |
| `adaptive_reid/diagnostics.py` | training diagnostics JSON, separate run_state/test_metrics filenames (Part 2/7) |
| `adaptive_reid/health.py` | `classify_run_health` (Part 3) |
| `adaptive_reid/weights.py` | parameter hash / `weights_changed` (Part 4) |
| `adaptive_reid/restarts.py` | screening/confirmatory schedules + replacement driver (Part 5) |
| `adaptive_reid/selection.py` | representative selection (validation only) (Part 6) |
| `adaptive_reid/summary.py` | arm summary aggregation (Part 8) |
| `adaptive_reid/bootstrap.py` | R-9 patient-cluster bootstrap (BLOCKED) + pair-sampling diagnostic (Part 9) |
| `adaptive_reid/topk.py` | frozen Top-k list (Part 10) |
| `adaptive_reid/determinism.py` | per-arm determinism check (Part 11) |
| `adaptive_reid/provenance.py` | arm provenance record (Part 12) |
| `adaptive_reid/pipeline.py` | staged pipeline A–E (Part 6) |
| `run_adaptive_reid_arm.py` | arm driver CLI (stub mode = no GPU, production mode = real runs) |
| `eval_topk_adaptive.py` | Top-1/5/MRR with representative adaptive attacker + proxy mode (Part 10) |
| `test_adaptive_reid_protocol.py` | STEP 2B regression suite (Part 15) |
| `research_agent/02_STEP2B_ATTACKER_PROTOCOL_IMPLEMENTATION.md` | this report |
| `research_agent/STEP2B_ATTACKER_PROTOCOL.diff` | full diff artifact vs an existing base commit |

## 3. Validation metrics implementation (Part 1)

`validate_snn` gains an **optional** `return_metrics=False`.

- `False` (default): returns `validation_loss` exactly as historically — every existing
  caller (`agents/AgentSiameseNetwork.py`) is untouched and bit-compatible.
- `True`: returns `(validation_loss, {'auc': float, 'accuracy': float})`.

Collection details:
- `y_true` = validation labels; `y_scores` = continuous logits (before sigmoid) from the
  same network forward. ROC-AUC is computed on continuous scores via
  `adaptive_reid.metrics.compute_auc` (never on thresholded predictions).
- Accuracy uses the canonical binary threshold 0.5 on the probability scale
  (`metrics.compute_accuracy`).
- Only the existing validation split is iterated — the test split is never touched and
  validation ordering / pair files are not altered.

Unit tests (`test_adaptive_reid_protocol.py`):
- perfect ranking → AUC 1, reversed → AUC 0, all-tied → AUC 0.5;
- accuracy calculation on a known case;
- empty / NaN / single-class / shape-mismatch inputs raise `ValueError`.

## 4. Run-health and near-chance implementation (Part 3)

`health.classify_run_health(training_diagnostics)` returns `(state, near_chance)`.

- `NUMERICALLY_INVALID` fires ONLY on objective execution failures:
  NaN/Inf, checkpoint missing, checkpoint unloadable, weights unchanged from
  initialization, `termination_reason == 'infrastructure_failure'`, any illegal
  termination path, zero completed epochs.
- Performance level NEVER triggers exclusion. A completed low-performance run is
  `VALID` (near_chance set to True when `best_validation_loss >= 0.6800` **and**
  `best_validation_auc <= 0.55`), and stays in every final estimand.

API safety (Part 13 #1): the signature takes exactly one positional parameter,
`training_diagnostics`, so test AUC/predictions/labels cannot be passed. Regression
tests introspect the signature and assert the validity JSON contains no test field.

## 5. Restart driver (Part 5)

`adaptive_reid/restarts.py`:

- `ScreeningSchedule`: target 3, initial seeds 0,1,2, max replacements 2, max attempts 5.
- `ConfirmatorySchedule`: target 10, initial seeds 0..9, max infrastructure
  replacements 5, max attempts 15.
- `run_schedule` runs initial seeds first, then replacement seeds as the **next integers
  in ascending order** (never user-selected after outcomes). Replacement occurs only when
  `state == NUMERICALLY_INVALID`; a completed `VALID + near_chance=True` is **never**
  replaced. Every attempted run is recorded in the returned attempts list.

Note: the screening mode's replacement budget (2) equals the initial seeds count, so the
driver breaks once 2 replacements are consumed and < target valid runs remain — strictly
bounded by max_attempts = 5 either way.

## 6. Train / validation / test stage separation (Part 6)

`adaptive_reid/pipeline.py` `ArmPipeline` with explicit stages:

- STAGE A train+validate all restarts → persist per-run `training_diagnostics.json`,
  checkpoint, validation curves.
- STAGE B classify validity + near-chance.
- STAGE C select representative via `selection.select_representative` using **validation
  AUC only**.
- STAGE D persist representative identity (`is_representative` marker + provenance
  `representative_attacker_seed`).
- STAGE E evaluate frozen TEST pairs once per completed attacker.

The driver CLI gating `--stage a_d` (representative frozen, **no** test eval) vs
`--stage a_e` (test eval after Stage D) makes accidental test-derived selection
structurally impossible. Legacy `run_snn_multiseed.py` / `AgentSiameseNetwork` untouched.

## 7. Representative attacker selection (Part 6, Stage C)

`selection.select_representative`:
- computes each valid restart's **best validation AUC** (`max` over its per-epoch
  validation AUC series);
- picks the restart closest to the **median** of those best-validation-AUCs;
- tie → smaller `attacker_seed`.

Signature has no test-derived parameter; a regression test (Part 13 #2) demonstrates the
function neither accepts nor uses test AUC (introspects the signature).

## 8. Final privacy aggregation (Part 8)

`summary.summarize_arm(attempts)` over EXACTLY the numerically valid completed attacks:
`n_attempted`, `n_numerically_invalid`, `n_valid`, `n_near_chance`, `mean/std/median/max
test AUC`, `representative_attacker_seed`. Test AUC comes only from
per-attempt `test_metrics['auc']` written by Stage E.

- Near-chance runs are never dropped; no "successful-attacks-only" mean is computed.
- Regression test uses synthetic runs `[0.50, 0.52, 0.61, 0.68, ...]` and asserts 0.50
  and 0.52 are included in every aggregate.

## 9. Patient-clustered bootstrap (Part 9, R-9 BLOCKED)

Inspected the actual pair-file schema (`image_pairs/image_pairs_validation_2000.txt`):
each line is `<image1> <image2> <label>`; positive pairs always share one patient
identity, negative pairs span **two** patient identities.

A patient-cluster (identity) bootstrap resamples patient identities; a **positive** pair
belongs unambiguously to one cluster, but a **negative** pair spans two clusters and the
frozen protocol does not define which cluster owns the pair under resampling. Per the
protocol's explicit rule this ambiguity is **not guessed**:

- `bootstrap.patient_cluster_bootstrap_is_ambiguous` → True for the frozen pair files;
- `bootstrap.report_R9_ambiguity` returns
  `R9_status: BLOCKED_FOR_SCIENTIFIC_CLARIFICATION` with the exact ambiguity text and
  observed pair statistics (positive=1000 same-patient, negative=1000 two-patient for
  validation);
- the historical pair-level bootstrap is retained and labelled a **pair-sampling
  diagnostic** (`PatientClusterResampler` refuses to invent a rule).

## 10. Top-k infrastructure (Part 10)

`adaptive_reid/topk.py` builds the frozen gallery/probe metadata
(`topk_frozen_list.csv`, patient_id / gallery_image / gallery_followup / probe_image /
probe_followup) from `chexnet/nih_labels.csv`, which provably contains `Follow-up #`
per image (verified: test fold has 2797 patients, 1424 with ≥2 images; for N=10 the
built list gets different follow-ups for 100% of selected pairs).

- N=500, selection seed 42, CLEAN gallery + ANONYMIZED probe.
- Built once per arm and reused by every arm (`save_frozen_topk_list` upstream of any
  headline computation).
- `eval_topk_adaptive.py` computes Top-1 / Top-5 / MRR using the **representative
  adaptive attacker** branch embedding; `--proxy_resnet` runs the older frozen ImageNet
  evaluation as an explicitly-named proxy mode.
- No headline Top-k results computed in STEP 2B (infrastructure only).

## 11. Determinism checks (Part 11)

`adaptive_reid/determinism.check_deterministic(generate, ...)` runs the same input
twice with the same checkpoint/config and requires bit-identical output; `arm_is_stochastic`
flags the arm when outputs differ. Unit-tested on deterministic and stochastic mocks.
The full test-set determinism run is deferred (as required) to a later step.

## 12. Provenance (Part 12)

`adaptive_reid/provenance.build_arm_provenance` writes `arm_provenance.json` with:
arm_id, git_commit, transform_mode, generator_checkpoint_path + hash, mu, stochastic
lambda, attacker architecture + hyperparameters, seeds attempted, pair file paths +
hashes, representative seed + selection criterion, run states, near-chance flags,
timestamps, schedule name. `transform_mode` and generator hash are explicit so legacy vs
corrected (and different checkpoints) cannot be confused.

## 13. Anti-leakage guarantees (Part 13)

1. run-health classifier cannot receive test AUC — signature allows only `training_diagnostics`.
2. representative selector cannot receive test AUC — signature introspected in tests.
3. restart replacement logic cannot inspect test AUC — `run_schedule` only reads `state`.
4. near-chance runs are never replaced merely for low AUC — tested.
5. test stage cannot alter representative selection — Stage D precedes Stage E; tests
   verify stage ordering and that `stage_d` runs before any test evaluation.
6. test pair files are not touched during development/screening logic — the pipeline
   passes them only to Stage E's worker.

## 14. Backward compatibility (Part 14)

- `utils.validate_snn` default path returns float (verified live on GPU: 0.6967 for both
  old-style float return and the metrics mode).
- `test_operator_repair.py` → PASS (STEP 1B REVIEW REMEDIATION: PASS).
- `test_grad_accum.py` → ALL PASSED.
- `utils.deform` / `pretrain` / `preval` still route through `build_sampling_grid`;
  `resolve_transform_mode(None) == 'legacy'`.
- Old pair files, archived results, legacy scripts unchanged.

## 15. Full automated test results

```
STEP 2B PROTOCOL TEST SUITE: PASS
```

All 28 checks in `test_adaptive_reid_protocol.py` pass (A–P + stage-ordering X
coverage), on CPU where possible plus the live GPU `validate_snn` backward-compat check
above.

## 16. Remaining ambiguities / blockers

**R-9 — patient-cluster bootstrap: BLOCKED FOR SCIENTIFIC CLARIFICATION.**

- Pair schema: negative pairs span two patient identities; the frozen protocol does not
  define the cluster-membership rule for such pairs under patient-cluster resampling.
- No clustering rule was invented. A clean, unambiguous patient-cluster bootstrap is not
  implementable from the current pair schema without a scientific decision.
- All other parts (validation metrics, diagnostics, health/run-state, schedules, staged
  pipeline, representative selection, aggregation, determinism, Top-k, provenance,
  anti-leakage) are implemented and tested.
- Therefore the overall step status is **PASS WITH R-9 BLOCKED** (a single blocked
  subtask per the protocol's own escape hatch).

## 17. Exact reproduction commands

```bash
# STEP 2B regression suite (CPU; the grad-accum sub-test needs a GPU/CUDA)
python test_adaptive_reid_protocol.py

# Pre-existing STEP 1 regressions
python test_operator_repair.py
python test_grad_accum.py

# Driver smoke runs (stub mode: no GPU training)
python run_adaptive_reid_arm.py --mode screening    --arm_id arm_s  --mu 0.01 \
    --transform_mode corrected --out_dir /tmp/arm_s --stub --stage a_d
python run_adaptive_reid_arm.py --mode confirmatory --arm_id arm_c  --mu 0.01 \
    --transform_mode legacy --out_dir /tmp/arm_c --stub --stage a_e

# Production arm (AFTER the strict stop is lifted): drop --stub and pass --checkpoint
python run_adaptive_reid_arm.py --mode confirmatory --arm_id arm_corrected_mu0.01 \
    --checkpoint ./archive/.../generator_lowest_total_loss.pth \
    --transform_mode corrected --mu 0.01 --out_dir ./archive/adaptive_reid_arms/... --stage a_d
```