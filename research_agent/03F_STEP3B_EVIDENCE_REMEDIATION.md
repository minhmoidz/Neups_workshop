# 03F — STEP 3B Evidence Remediation

> Status: **COMPLETE** (2026-08-12). Purely an evidence/provenance remediation of the
> already-executed STEP 3B official TEST evaluation. No test images were opened, no
> inference run, no attacker retrained, no `test_metrics.json` overwritten, no result
> changed. The ten per-seed TEST AUCs are byte-identical to the STEP 3B run.

---

# 1. Audit failure being remediated

`research_agent/03E_STEP3B_SCIENTIFIC_AUDIT.md` was **not present locally** at remediation
time. Per the task, the audit is recorded here from the task brief, which states:

> STEP 3B SCIENTIFIC AUDIT: FAIL

The audit did **not** find numerical evidence that the reported result is wrong. It found
three **auditability blockers**:

1. Runtime JSON artifacts (arm provenance, per-seed diagnostics/test metrics, summaries)
   were excluded by `.gitignore` — the entire `archive/` tree is ignored
   (`.gitignore:19`), so the raw evidence was never tracked.
2. The committed summary `research_agent/03D_corrected_baseline_reid_summary.json` was
   **hand-assembled** rather than emitted through the audited `summarize_arm()` firewall.
3. `run_3b_confirmatory.py`, which implements the real Stage E path, has **never been
   scientifically reviewed** — and, as this remediation documents, was not in Git at the
   execution commit.

This patch addresses all three **without** touching the scientific measurement.

## Remediation commits / context

- Starting state: HEAD = `bd57f2b33d4a3deba167768a296f81635a0dd6e1` (STEP 3B commit),
  branch `main`, working tree clean, upstream in sync.
- The original runtime artifacts existed on the execution machine
  (`archive/adaptive_reid_baseline_corrected_confirmatory/`). They were inventoried,
  verified, and copied **byte-for-byte** into a tracked evidence bundle.

---

# 2. Original runtime artifact inventory

Original arm directory: `archive/adaptive_reid_baseline_corrected_confirmatory/` (not
recreated; the existing arm is the one used for STEP 3B).

## Arm level

| Artifact | Present | Bytes |
|---|---|---|
| `arm_provenance.json` | yes | 2 292 |
| `arm_provenance_stageD.json` | yes | 2 250 |
| `arm_summary.json` | yes | 621 |
| `arm_summary_stageD.json` | yes | 383 |
| `attacker_checkpoint_paths.json` | yes | 712 |

## Per-attempt (seeds 0–9)

Every attempted seed directory
`archive/adaptive_reid_baseline_corrected_confirmatory/runs/retrain_snn_seedN/` contains:

| File | Present (all 10 seeds) |
|---|---|
| `training_diagnostics.json` | yes |
| `run_state.json` | yes |
| `test_metrics.json` | yes |
| `run_signature.json` | yes |

No replacement attempts were needed: the confirmatory schedule issued seeds `0..9` and all
10 completed VALID, so seeds 10–14 were never used. There are no failed/invalid attempts
to preserve.

Attempted seeds per `arm_provenance.json`: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.

---

# 3. Evidence bundle layout

Tracked under `research_agent/03D_artifacts/`:

```
03D_artifacts/
├── MANIFEST.json                     # SHA-256 manifest, 45 files
├── STAGE_E_CODE_PROVENANCE.json      # Stage-E driver provenance (T13)
├── checkpoint_references.json        # generator + 10 attacker paths/SHAs (T5)
├── per_seed_verification.json        # 10-row compliance table (T2/T10/T11)
├── arm_provenance.json               # byte-for-byte copy
├── arm_provenance_stageD.json        # byte-for-byte copy
├── arm_summary.json                  # original STEP 3B runtime summary (byte-for-byte)
├── arm_summary_stageD.json           # byte-for-byte copy
├── arm_summary_machine.json          # NEW — emitted through audited summarize_arm()
├── attacker_checkpoint_paths.json    # byte-for-byte copy
└── seed_0/ … seed_9/
    ├── training_diagnostics.json
    ├── run_state.json
    ├── test_metrics.json
    └── run_signature.json
```

All per-seed JSONs are byte-for-byte copies of the runtime originals — **not** rebuilt
from reports, summaries, console output, or memory.

---

# 4. Byte-for-byte SHA verification

`research_agent/03D_artifacts/MANIFEST.json` records, for every one of the 45 copied
artifacts: `original_path`, `tracked_copy_path`, `byte_size`, `sha256`, and
`sha256_match_original`.

- At copy time: **45/45 `sha256_match_original = true`**.
- Independent re-verification after copying (recomputing `sha256` of both sides):
  **45/45 match, 0 mismatches**.

A mismatch is BLOCKING; none was found.

---

# 5. Per-seed health evidence

For each of the 10 valid seeds, `run_state.json` records `state=VALID`, `near_chance=false`,
`evaluated_test=true`. `adaptive_reid.health.classify_run_health` was independently
re-invoked on the persisted `training_diagnostics.json` of every seed — all 10 reproduce
`(VALID, False)` exactly, matching the recorded `run_state.json`:

| Seed | run_state | health re-classification | termination | any_nan_inf | ckpt exists/loadable | weights changed |
|---:|---|---|---|---|---|---|
| 0 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 1 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 2 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 3 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 4 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 5 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 6 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 7 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 8 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |
| 9 | VALID | VALID / near=False | early_stopping | false | yes/yes | true |

Every `training_diagnostics.json` satisfies the canonical schema (no missing required
fields, no forbidden test-derived fields).

---

# 6. Per-seed generator consistency

Every recorded generator digest equals the frozen STEP 3A1 digest
`8a489eec…c384c2` — checked independently in three places per seed
(`training_diagnostics.json`, `test_metrics.json`, `run_signature.json`):

| Seed | generator_checkpoint_hash (recorded) | matches frozen 8a489eec… |
|---:|---|---|
| 0 | 8a489eec…c384c2 | ✅ |
| 1 | 8a489eec…c384c2 | ✅ |
| 2 | 8a489eec…c384c2 | ✅ |
| 3 | 8a489eec…c384c2 | ✅ |
| 4 | 8a489eec…c384c2 | ✅ |
| 5 | 8a489eec…c384c2 | ✅ |
| 6 | 8a489eec…c384c2 | ✅ |
| 7 | 8a489eec…c384c2 | ✅ |
| 8 | 8a489eec…c384c2 | ✅ |
| 9 | 8a489eec…c384c2 | ✅ |

All 10 = ✅. The live frozen generator file `networks/corrected_baseline/
generator_lowest_total_loss_corrected.pth` was re-hashed for this task and still equals
`8a489eec…c384c2` (bytes read only; no inference). Protocol-document hashes recorded in
the per-seed diagnostics match the R-12 canonical pair
(`01_ADAPTIVE_REID_PROTOCOL.md` + `01B_PROTOCOL_AMENDMENT.md`).

This closes the Scientist's concern that "arm-level generator agreement is not enough":
**every attacker is individually checkable against the same frozen generator digest.**

---

# 7. Representative Stage-D runtime evidence

## Recorded Stage D state

- `arm_provenance_stageD.json`: `representative_attacker_seed = 4`,
  `pair_test_hash = WITHHELD_TEST_SET_LOCK`, `scientific_summary_available=false`,
  `test_auc_values=[]` — i.e. the Stage-D record contains **no test result**.
- `arm_summary_stageD.json`: `mean/max/median/std_test_auc = null`, `test_auc_values = []`.
- `arm_provenance.json` (final): `representative_attacker_seed = 4`, real
  `pair_test_hash = 87e52830…`.

## Ordering evidence (runtime, not just code structure)

Run timeline reconstructed from the on-disk logs (`/tmp/opencode/`):

| Log | mtime (UTC) | Content | Test evaluations |
|---|---|---|---|
| `3b_confirmatory.log` | 16:09:41 | stages A–D background run | 0 |
| `3b_stageE.log` | 16:15:30 | **first Stage E invocation** — prints `STAGE D FROZEN: representative_attacker_seed = 4` (line 6) then 10× `Testing----->` | **10 (the real evaluations)** |
| `3b_stageE2.log` | 16:25:21 | idempotent re-invocation after the summarizer-key fix | **0** (all 10 test_metrics reused, never recomputed) |

- In `3b_stageE.log`, `STAGE D FROZEN` (line 6) appears **before** the first `Testing----->`
  (line 22): the representative was frozen before Stage E evaluated the test set.
- In `3b_stageE2.log` there are **zero** `Testing----->` markers: the reuse guard
  (run_3b `evaluate_test_real`) found the existing `test_metrics.json` (unchanged
  generator digest) and returned the recorded results without a single re-evaluation.

## Honest caveat about the persisted Stage-D timestamp

The persisted `arm_provenance_stageD.json` file **mtime is 16:25:21 UTC**, which is
**later** than the per-seed test-metric timestamps (16:13:20–16:15:30 UTC). This is an
artifact of the idempotent second invocation (16:25:21) re-writing the Stage-D provenance
file when it re-ran. It does **not** mean Stage E ran before Stage D: the run that
actually evaluated TEST (`3b_stageE.log`, 16:13–16:15) printed `STAGE D FROZEN: seed=4`
at line 6, before any `Testing----->`, and `evaluate_test_real` calls
`verify_stage_e_generator_hash` (which requires the Stage-A–D diagnostics) before any
evaluation. Both the log line-order and the Stage-D record's `WITHHELD_TEST_SET_LOCK`
support Stage-D-before-E; the file mtime alone does not. This is reported exactly as
observed, for Scientist judgement.

---

# 8. Per-seed real TEST records

Each `test_metrics.json` records `n_pairs=5000`, `stub=false`, `synthetic=false`,
`valid_for_scientific_reporting=true`, the real `test_auc`, and
`evaluation_timestamp`:

| Seed | test_auc | n_pairs | evaluated_test (run_state) | evaluation_timestamp (UTC) |
|---:|---:|---:|---:|---|
| 0 | 0.78469120 | 5000 | true | 2026-08-11T16:13:20 |
| 1 | 0.77798128 | 5000 | true | 2026-08-11T16:13:35 |
| 2 | 0.72106304 | 5000 | true | 2026-08-11T16:13:49 |
| 3 | 0.66013752 | 5000 | true | 2026-08-11T16:14:04 |
| 4 | 0.72330368 | 5000 | true | 2026-08-11T16:14:18 |
| 5 | 0.80366208 | 5000 | true | 2026-08-11T16:14:33 |
| 6 | 0.71028000 | 5000 | true | 2026-08-11T16:14:47 |
| 7 | 0.71749968 | 5000 | true | 2026-08-11T16:15:01 |
| 8 | 0.80315408 | 5000 | true | 2026-08-11T16:15:16 |
| 9 | 0.69016736 | 5000 | true | 2026-08-11T16:15:30 |

Timestamps increase monotonically with seed and are spaced ~14 s apart — consistent with
one sequential real evaluation pass.

---

# 9. Synthetic/stub firewall evidence

Machine-emitted summary (see §10) independently reports:

- `contains_stub_or_synthetic_metrics = false`
- `n_stub_or_synthetic_test_metrics = 0`
- `scientific_summary_available = true`

Direct record check: all 10 `test_metrics.json` files have `stub=false`,
`synthetic=false`, `valid_for_scientific_reporting=true`. No stub/synthetic marker exists
in any contributing record. These values were derived by `summarize_arm()` from the
records — they were **not** injected by hand.

---

# 10. Machine-emitted arm summary

`research_agent/03D_artifacts/arm_summary_machine.json` is the **direct output** of the
audited `adaptive_reid.summary.summarize_arm()` invoked on the persisted records
(reconstructed via `research_agent/emit_arm_summary_machine.py`), with the canonical
Stage-E reuse adapter (adds the `auc` alias from `test_auc`, exactly as
`run_3b_confirmatory.py:evaluate_test_real` does). Contents:

```json
n_attempted                     10
n_valid                         10
n_numerically_invalid           0
n_near_chance                   0
n_stub_or_synthetic_test_metrics 0
contains_stub_or_synthetic_metrics false
scientific_summary_available    true
representative_attacker_seed    4
mean_test_auc                   0.7391939919999999
std_test_auc                    0.04984744373953724
median_test_auc                 0.7221833599999999
max_test_auc                    0.8036620800000001
test_auc_values                 [0.7846912, 0.77798128, 0.72106304, 0.66013752,
                                 0.72330368, 0.80366208, 0.71028, 0.71749968,
                                 0.80315408, 0.69016736]
```

`research_agent/03D_corrected_baseline_reid_summary_auditable.json` is the new authoritative
pointer: it links the machine summary and manifest, records the Stage-E driver SHA-256 and
the generator digest, and reads its load-bearing numbers from
`arm_summary_machine.json` (no hand-written scientific validity assertion).

---

# 11. Independent arithmetic

Plain-Python recomputation (no numpy, no `adaptive_reid`) of the 10 recorded TEST AUCs:

| Estimand | Independent value | Reported (STEP 3B / machine) | |Δ| |
|---|---:|---:|---:|
| mean | 0.739193992000000 | 0.7391939919999999 | 1.1e-16 |
| sample SD (ddof=1) | 0.049847443739537 | 0.04984744373953724 | 0 |
| median | 0.722183360000000 | 0.7221833599999999 | 0 |
| max | 0.803662080000000 | 0.8036620800000001 | 0 |

All within floating-point epsilon. **Machine records agree with the old summary**; no
discrepancy to report, and no per-seed metric was modified.

---

# 12. Test-pair consistency

All 10 seeds record `pair_test_hash = 87e528308507ada349d478d0c16e85f858dba8fb613abbca4857ffc437490fe2`,
equal to the canonical testing-pair digest, and `pair_test_path =
image_pairs/image_pairs_testing_5000.txt`.

The pair metadata file itself was re-hashed (metadata only, no image inference) and still
equals `87e52830…`: **verified**.

---

# 13. Single-evaluation evidence

What the artifacts demonstrate:

1. Each `test_metrics.json` has exactly one `evaluation_timestamp` and the `run_state.json`
   flips `evaluated_test=false → true` at most once per seed.
2. Log evidence: exactly **10 `Testing----->` markers** in the run that evaluated TEST
   (`3b_stageE.log`), i.e. one pass over 10 seeds; the re-invocation
   (`3b_stageE2.log`) has **0** markers (reuse, not re-evaluation).
3. `evaluate_test_real` only writes `test_metrics.json` when it does not already exist
   with a matching generator digest; otherwise it returns the recorded result.

Honest limitation: **the persisted artifact schema does not contain an explicit
evaluation-counter field.** One-evaluation-per-attacker is therefore corroborated by the
single timestamp per file + the single pass of 10 `Testing----->` markers in the run that
produced the metrics + the reuse guard logic, but it is **NOT directly provable from the
schema alone**. This is stated for Scientist judgement rather than overstated as fact.

---

# 14. Real Stage-E driver provenance

`research_agent/03D_artifacts/STAGE_E_CODE_PROVENANCE.json`:

- `path`: `run_3b_confirmatory.py`
- `sha256`: `e9f578dac91c8e8f080faceba360f13c92622dabb81079a5c2e08219e2e0b5b6`
- `git_commit_first_introduced`: `bd57f2b33d4a3deba167768a296f81635a0dd6e1`
- `step3b_execution_commit_recorded_in_arm_provenance`: `1c54435846af6f0ca30ef668464032e727b5aed6`
- `committed_version_equals_disk_version`: true (verified by clean `git diff HEAD`)

**Important honesty finding:** `run_3b_confirmatory.py` was **not** in Git at the
execution commit `1c54435`; it was first committed at `bd57f2b` after the run. The
committed blob is byte-identical to the file on disk (working tree clean), so the code a
Scientist reviews is the code that produced the final summary — but the **exact** uncommitted
bytes that ran during the first Stage-E pass (16:13–16:15, the pass that actually computed
the AUCs from images) are **not recoverable from Git**. The committed driver reproduces the
summary from the persisted records (verified: machine summary == original summary), and its
`evaluate_test_real` reuse path is exactly what consumed those records at 16:25:21.

The driver was **not rewritten** for this remediation.

---

# 15. New Stage-E regression tests

`test_stage_e_confirmatory.py` (synthetic/mock data only; no real TEST data, no inference):

| # | Guarantee | Status |
|---|---|---|
| 1 | Stage E cannot execute before representative Stage D is frozen | PASS |
| 1b | Stage E without a real evaluator raises | PASS |
| 2 | Stage E verifies generator content hash before evaluation (D-1) | PASS |
| 3 | Stage E consumes the fixed test pair configuration | PASS |
| 4 | Real evaluator writes `stub=false, synthetic=false, VFSR=true` | PASS |
| 5 | Stub/synthetic records are never emitted as scientific (R-11) | PASS |
| 6 | Stage E does not retrain or modify the attacker checkpoint | PASS |
| 7 | Stage E does not change the representative seed | PASS |
| 8 | Existing immutable test metrics are reused, not overwritten | PASS |
| 8b | Stale generator digest prevents reuse (hard failure) | PASS |
| 9 | Stale generator digest causes hard failure | PASS |
| 10 | Emitted schema compatible with `summarize_arm()` (numbers match) | PASS |

Run: `PYTHONPATH=. .venv/bin/python test_stage_e_confirmatory.py` → **PASS**.

The pre-existing suites were also re-run: `test_adaptive_reid_protocol.py`,
`test_operator_repair.py`, `test_grad_accum.py` — see §16.

---

# 16. Files changed

Added (all tracked, no `.pth`, no broad `.gitignore` change):

- `research_agent/03D_artifacts/` — 45 byte-for-byte evidence copies + manifest
- `research_agent/03D_artifacts/MANIFEST.json`
- `research_agent/03D_artifacts/STAGE_E_CODE_PROVENANCE.json`
- `research_agent/03D_artifacts/checkpoint_references.json`
- `research_agent/03D_artifacts/per_seed_verification.json`
- `research_agent/03D_artifacts/arm_summary_machine.json`
- `research_agent/emit_arm_summary_machine.py`
- `research_agent/03D_corrected_baseline_reid_summary_auditable.json`
- `test_stage_e_confirmatory.py`
- `research_agent/03F_STEP3B_EVIDENCE_REMEDIATION.md` (this report)

Unchanged: `run_3b_confirmatory.py` (byte-identical to `bd57f2b`), all ten
`test_metrics.json`, all ten `training_diagnostics.json`, `arm_summary.json`, `arm_provenance*`.

---

# 17. Remaining audit limitations

1. **Stage-E driver not in Git at execution commit.** The first-stage-E-pass bytes that
   computed the AUCs from images are not recoverable from Git; the committed driver
   reproduces the summary from persisted records and is byte-identical to the on-disk file
   that produced the final summary. A Scientist can review the committed code but not the
   transient first-pass variant.
2. **Single-evaluation count is not schema-provable** (§13) — it is strongly corroborated
   by timestamps + log markers + reuse guard, but there is no explicit counter field.
3. **Persisted Stage-D file mtime postdates the test metrics** due to the idempotent
   re-invocation re-writing it (§7); the Stage-D-before-E ordering rests on log line-order
   and the `WITHHELD_TEST_SET_LOCK` record, not on that mtime.
4. **Checkpoint `.pth` files are referenced, not committed** (paths + SHA-256 in
   `checkpoint_references.json`); their integrity at a future time depends on the archive
   surviving, or re-pinning from the recorded hashes.
5. The hand-assembled original summary remains on disk as the historical STEP 3B output;
   the auditable pointer supersedes it for load-bearing claims.

---

# 18. Final remediation verdict

All three auditability blockers are addressed without touching the measurement:

1. ✅ Runtime JSON evidence is now **tracked** (byte-for-byte, SHA-verified) under
   `research_agent/03D_artifacts/`, no `.gitignore` change, no `.pth` committed.
2. ✅ The scientific summary is now **machine-emitted** through the audited
   `summarize_arm()` (`arm_summary_machine.json`) with an auditable pointer; the original
   hand-assembled summary is retained as history only.
3. ✅ The real Stage-E implementation is now **reviewable** (committed SHA + provenance +
   a 12-case synthetic regression suite) and was **not rewritten**.

No new scientific measurement was produced; the ten TEST AUCs are unchanged and immutable.

**STEP 3B.1 EVIDENCE REMEDIATION: PASS**
