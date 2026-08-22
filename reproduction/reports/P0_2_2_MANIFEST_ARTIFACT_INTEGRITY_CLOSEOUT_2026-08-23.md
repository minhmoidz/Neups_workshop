# P0.2.2 MANIFEST–ARTIFACT INTEGRITY HOTFIX — CPU-ONLY CLOSEOUT

**Date:** 2026-08-23
**Status:** CPU-only synthetic-source hotfix. No training/evaluation/checkpoint/image/pair-file loading, no GPU/CUDA, no Direction B evaluation, no approval manifest, no scientific loop. PASS means only `READY_FOR_EXTERNAL_RE_REVIEW`.

---

## 1. SOURCE IDENTITY

- Starting: branch `review/p0-2-1-source-closeout-20260822`, commit `5cd7af911e15a70199699a6095e78ee78ba703e2` (parent `4ad3aa3…` verified).
- New isolated worktree `/tmp/p0-22-integrity-worktree`, new branch `review/p0-2-2-manifest-artifact-integrity-20260823`.
- Direction B process: NOT_RUNNING (checked; never touched).
- `CUDA_VISIBLE_DEVICES=""` for every Python run; `torch.cuda.is_initialized()` asserted False throughout.

## 2. EXACT CHANGED-FILE LIST (8)

```text
M reproduction/p0_bridge/protocol_v1.json
M reproduction/p0_bridge/manifest_io.py
M reproduction/p0_bridge/run_p0_bridge.py
M reproduction/p0_bridge/generator_guard.py
M reproduction/p0_bridge/tests/test_manifest_io.py
M reproduction/p0_bridge/tests/test_runner_gate.py
M reproduction/p0_bridge/tests/test_generator_guard.py
A reproduction/reports/P0_2_2_MANIFEST_ARTIFACT_INTEGRITY_CLOSEOUT_2026-08-23.md
```

`seed_contract.py` and `deterministic_sampler.py` untouched (forbidden by tasking). The three previous reports untouched (hashes re-verified at §7).

## 3. FAILING-BEFORE COUNTEREXAMPLES (reproduced on `5cd7af9` with production functions)

| # | Counterexample | Result on 5cd7af9 |
|---|---|---|
| CE1 | complete 10-identity screen aggregate with NO `predictions.parquet`/`attacker_best.pth` on disk | ACCEPTED (count=10) |
| CE2 | `epoch_order_hashes={}` for every arm/seed | ACCEPTED |
| CE3 | extra unlocked key in `derived_seeds` | ACCEPTED |
| CE4 | undeclared arm directory under runs root | never flagged (failed later on unrelated missing identity; "ROGUE" absent from message) |
| CE5 | final-component symlink AND parent-directory symlink into `verify_artifact_sha()` | BOTH ACCEPTED (hashed through symlinks) |
| CE6 | `verify_all_artifacts()` hashing `generator_path` bytes while ignoring `execution_path` bytes | ACCEPTED (provenance GOOD / execution EVIL passed) |
| CE7 | generator parameter mutated in-place during `protected_forward` | UNDETECTED (weights changed, no exception) |

## 4. FIXES IMPLEMENTED

**A — manifests bound to actual output bytes (`manifest_io.py`).**
Every expected (arm, seed) now requires a real non-symlink run directory containing EXACTLY `{predictions.parquet, attacker_best.pth, run_manifest.json}`; missing/symlink/non-file/partial-temp/undeclared entries rejected; SHA-256 RECOMPUTED from actual prediction and attacker-checkpoint bytes and compared against `predictions_sha256` / `attacker_best_sha256`; all bound hash fields must be lowercase 64-char hex (`^[0-9a-f]{64}$`); outputs newer than the run manifest are rejected as post-manifest modification; atomic write + file/dir fsync retained.

**B — full paired-order contract.**
`derived_seeds` key set must EQUAL the locked protocol domain set (no missing/extra keys); `epoch_order_hashes` must be a non-empty mapping with exact epoch keys `0..stop_epoch`; every value lowercase hex64; expected hashes RECOMPUTED via production `build_permutation`/`order_hash` from locked seed + epoch + sampler schema + auditable TRAIN `pair_count` (=10000, added to the locked protocol rather than inferred from filenames); cross-arm equality still enforced. Regressions: empty, missing, extra, malformed-hex, and recomputed-mismatch order hashes all rejected.

**C — closed output-tree identity set.**
Arms/seeds derived only from the locked protocol; undeclared arm directories at the runs root rejected (CE4); undeclared seed directories and non-directory/symlink entries rejected; screen = exactly 10 identities, full = exactly 52 enforced.

**D — correct checkpoint artifact (`run_p0_bridge.py`).**
`verify_all_artifacts()` verifies `execution_path` per arm (the ACTUAL local bytes); `generator_path` remains provenance metadata but both paths must be lexically safe repository-relative paths. Regression proves U_PUBLISHED verification requests `reproduction/workspace_prichexy/networks/generator_lowest_total_loss_mu_0.01.pth` semantics (fails on exec-byte mismatch at `exec.pth`, passes once exec bytes match). The unresolved ImageNet-artifact gate fires FIRST — before any other artifact is opened or hashed.

**E — symlink rejection done correctly.**
Lexical component walk beneath the injected root rejects symlink final components AND symlink intermediate directories before any hashing; traversal (`..`) rejected lexically; open uses `O_NOFOLLOW` where supported plus `fstat` regular-file verification (TOCTOU reduction); repository escape checked via realpath confinement. Regressions: final-component symlink, parent-directory symlink, lexical escape, ordinary in-repo regular file success, byte-hash mismatch — all green.

**F — complete generator state guard (`generator_guard.py`).**
Efficient parameter identity snapshot (storage pointer + `_version` counter + dtype + shape — NO GPU→CPU byte copies per batch): detects parameter ADDITION, REMOVAL, REPLACEMENT and IN-PLACE mutation alongside buffer/training-mode checks; full byte-level model-state hash remains available for trajectory boundaries. `no_grad`-only contract unchanged; BN-drift negative control and nested-inference-tensor rejection remain green; unchanged-generator forwards succeed.

## 5. SCHEMA / VERSION DISCIPLINE

```text
implementation_revision:            P0_2_2
supersedes_protocol_sha256:         096aeeb7e54e236838ebf97c296fee3b2423dd340d9ac9869b882e2569db8a4e   (P0_2_1)
REVISED_PROTOCOL_SHA256:            528da8b471b9a2d71b49eab2485ddc0b5690d2836fd5219fa42af035284d117c
component_schemas.run_manifest:     P0_RUN_MANIFEST_V1_2
component_schemas.aggregate_manifest: P0_AGGREGATE_MANIFEST_V1_2
authorization_status:               NOT_AUTHORIZED
```

Preserved unchanged: scientific estimand (Δ = AUC_U_PUBLISHED − AUC_D_BDEV), arms (U_PUBLISHED/D_BDEV SHAs), seed sets (screen 42–46; full 42–67), delta sign, SEOI 0.03 provisional, RAW ROC-AUC definition.

## 6. TEST COMMANDS AND COUNTS

pytest remains unavailable offline; all five standalone suites executed explicitly (plus py_compile):

```text
CUDA_VISIBLE_DEVICES="" python -m py_compile <changed .py files>        → COMPILE_OK
CUDA_VISIBLE_DEVICES="" python tests/test_seed_contract.py              → ALL PASS (7)
CUDA_VISIBLE_DEVICES="" python tests/test_deterministic_sampler.py      → ALL PASS (10)
CUDA_VISIBLE_IMAGES/CUDA_VISIBLE_DEVICES="" python tests/test_manifest_io.py      → ALL PASS (12)
CUDA_VISIBLE_DEVICES="" python tests/test_generator_guard.py            → ALL PASS (12)
CUDA_VISIBLE_DEVICES="" python tests/test_runner_gate.py                → ALL PASS (21)
TOTAL: 60/60 PASS
```

Import-side-effect check green; `torch.cuda.is_initialized() == False` in every session; `git diff --check` clean; staged set == exact allowlist (verified programmatically).

## 7. UNCHANGED GOVERNED REPORTS

| Report | SHA-256 (unchanged) |
|---|---|
| P0_CANONICAL_PROTOCOL_LOCK_2026-08-21.md | `60f1b972c32f75c26a8a6aa64432efa0156312828981e9113f754822ed020b75` |
| P0_1_PROTOCOL_FEASIBILITY_CLOSEOUT_2026-08-22.md | `ccc8b87b6ba457835ba2ad0063473327ac34208e93c701bae595e8a043a4ab12` |
| P0_2_CPU_ONLY_IMPLEMENTATION_REPORT_2026-08-22.md | `c817512d3ddfcbf6d487306dcbc25193dc57106d0471a56d8eca09430ed945b3` |

## 8. LIMITATIONS

Scientific loop still intentionally unimplemented; artifact byte gates exercised only against synthetic injected roots; mtime-based post-manifest-modification detection assumes a cooperating filesystem clock; O_NOFOLLOW used where the platform provides it.

## 9. NO REAL ACCESS STATEMENT

No real artifact, dataset, image, pair file, generator/attacker/ImageNet checkpoint, or Direction B result was accessed, opened or hashed in this task; no network access; CUDA never initialized.

---

## 10. MACHINE-READABLE VERDICT

```text
P0_2_2_STATUS: READY_FOR_EXTERNAL_RE_REVIEW
P0_EXECUTION_AUTHORIZATION: NONE
GPU_AUTHORIZATION: NONE
SCIENTIFIC_LOOP: NOT_IMPLEMENTED

COUNTEREXAMPLES_REPRODUCED_BEFORE_FIX: 7/7
COUNTEREXAMPLES_ACCEPTED_AFTER_FIX: 0
MANIFEST_BYTE_BINDING: PASS
PAIRED_ORDER_CONTRACT_FULL: PASS
OUTPUT_TREE_IDENTITY_CLOSED: PASS
EXECUTION_PATH_VERIFICATION: PASS
SYMLINK_REJECTION_LEXICAL_NOFOLLOW: PASS
GENERATOR_PARAM_MUTATION_DETECTION: PASS

PROTOCOL_SCHEMA: P0_PROTOCOL_V1_1
IMPLEMENTATION_REVISION: P0_2_2
PREVIOUS_PROTOCOL_SHA: 096aeeb7e54e236838ebf97c296fee3b2423dd340d9ac9869b882e2569db8a4e
REVISED_PROTOCOL_SHA: 528da8b471b9a2d71b49eab2485ddc0b5690d2836fd5219fa42af035284d117c
IMAGENET_WEIGHT_ARTIFACT_STATUS: UNRESOLVED_BLOCKER

SYNTHETIC_CPU_TESTS: 60/60
PY_COMPILE: OK
GIT_DIFF_CHECK: CLEAN
CHANGED_PATHS: 8
GOVERNED_REPORTS_UNTOUCHED: YES
SEED_CONTRACT_AND_SAMPLER_MODIFIED: NO

CUDA_INITIALIZED: NO
REAL_ARTIFACTS_ACCESSED: NO
MODEL_TRAINING_OR_EVALUATION: NONE
ACTIVE_TRAINING_PROCESS_TOUCHED: NO
COMMIT_OR_PUSH: SEE BELOW (single normal push of this branch only)
NEXT_REQUIRED_ACTION: EXTERNAL_RE_REVIEW
```
