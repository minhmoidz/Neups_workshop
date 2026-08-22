# P0.2.1 EXTERNAL SOURCE-REVIEW CLOSEOUT — CPU-ONLY HOTFIX

**Date:** 2026-08-22
**Status:** HOTFIX + SYNTHETIC CPU TESTS ONLY. No scientific attacker loop, no real model/checkpoint, no medical image/pair file, no Direction B evaluation, no GPU/CUDA, no approval manifest created. A PASS means only `READY_FOR_EXTERNAL_RE_REVIEW`.

---

## 1. SOURCE BRANCH AND STARTING COMMIT

- Worktree `/tmp/p0-canonical-protocol-worktree`.
- Starting branch `review/p0-2-external-source-review-20260822`, HEAD `4ad3aa35444bca2432806dbc4df6443e967f5af3`, parent verified `a3513e5c…`.
- New hotfix branch created after verifying non-existence locally and remotely: `review/p0-2-1-source-closeout-20260822` (from `4ad3aa3`).
- Direction B process: **NOT_RUNNING** (completed; never touched).
- All Python under `CUDA_VISIBLE_DEVICES=""`; `torch.cuda.is_initialized()` asserted `False` in every session.

## 2. EXACT CHANGED-FILE LIST (13)

Modified (12): `p0_bridge/__init__.py`, `protocol_v1.json`, `seed_contract.py`, `deterministic_sampler.py`, `generator_guard.py`, `manifest_io.py`, `run_p0_bridge.py`, `tests/test_seed_contract.py`, `tests/test_deterministic_sampler.py`, `tests/test_generator_guard.py`, `tests/test_manifest_io.py`, `tests/test_runner_gate.py`.
Created (1): this report. Nothing else changed.

## 3. UNCHANGED PREVIOUS REPORT HASHES (before == after)

| Report | SHA-256 |
|---|---|
| P0_CANONICAL_PROTOCOL_LOCK_2026-08-21.md | `60f1b972c32f75c26a8a6aa64432efa0156312828981e9113f754822ed020b75` |
| P0_1_PROTOCOL_FEASIBILITY_CLOSEOUT_2026-08-22.md | `ccc8b87b6ba457835ba2ad0063473327ac34208e93c701bae595e8a043a4ab12` |
| P0_2_CPU_ONLY_IMPLEMENTATION_REPORT_2026-08-22.md | `c817512d3ddfcbf6d487306dcbc25193dc57106d0471a56d8eca09430ed945b3` |

## 4. BLOCKER-BY-BLOCKER CORRECTION

**A — deterministic worker contract (was: `derive_seed` called but not imported ⇒ NameError with num_workers>0).**
`deterministic_sampler.py` now imports every referenced function explicitly; the epoch permutation uses the LOCKED `train_order` domain explicitly (`train_order_seed()`); strict type validation (bool/float/negative/invalid) everywhere; worker initializer is a TOP-LEVEL picklable function (`p0_worker_init`) bound via `functools.partial`; a fresh explicit CPU `torch.Generator` is passed through DataLoader `generator=`, derived from the `dataloader_worker_base` domain + epoch (`P0_LOADERGEN_V1_1`); independent generator objects per arm with identical values; `shuffle=False`, deterministic sampler, `persistent_workers=False`; batch_size/num_workers validated. REAL `num_workers=1` synthetic iteration test passes and reproduces the exact permutation.

**B — sampler/order-hash correction.**
Schemas bumped to `P0_SAMPLER_V1_1` / `P0_ORDERHASH_V1_1`; epoch-order seed binds (schema, derived train_order seed, epoch, length). `order_hash()` rejects wrong length, duplicates, omissions, negatives, out-of-range indices and ambiguous numeric inputs (bool/float). Hash identical across arms for equal seed+epoch; computing it consumes no sampler state.

**C — generator guard strengthening.**
Rejects training=True on ANY submodule (nested Dropout/BatchNorm negative tests); detects added AND removed as well as mutated buffers; recursively rejects inference tensors inside tuple/list/dict/nested outputs; callable explicitly bound as `callable_fn(generator, *args)`; still `torch.no_grad()` only; downstream backward verified with zero generator gradients. Model-state hash bumped to `P0_MODELSTATE_V1_1`: versioned length-prefixed serialization (name/dtype/shape/byte-length + dense CPU bytes), explicit sparse/meta rejection.

**D — manifest identity validation.**
Aggregator derives arms/seeds FROM THE LOCKED PROTOCOL (caller lists removed); validates directory↔manifest identity binding, protocol schema/SHA, runner commit, per-arm generator role+SHA, TRAIN/VAL pair hashes, derived-seed bundle, initial-attacker/predictions/attacker-checkpoint hash formats, score direction, best≤stop≤max_epochs−1, status==COMPLETE, timestamps, environment provenance; enforces screen = exactly 10 identities and full = exactly 52; rejects one-arm grids, missing/extra identities, stale protocols, paired order-hash and init-hash disagreements. Per-run manifests are payload-hash sealed; the previous fake duplicate-set test was deleted and replaced by production-code regressions.

## 5. FRESH-OUTPUT / ATOMIC-WRITE CORRECTION

New `claim_run_directory()`: `<arm>/<seed>` must NOT exist at all (empty dirs rejected), exclusive creation, symlink rejection, defensive re-scan rejecting pre-existing predictions/checkpoint/manifests/temp files, concurrent-claim rejection. Atomic writes fsync the file AND the containing directory (where supported) for both run and aggregate manifests. Regression tests reproduce rejection of pre-existing `predictions.parquet`, `attacker_best.pth`, empty run directories, existing run manifests and aggregate manifests.

## 6. APPROVAL GATE CORRECTION

Blank approver rejected (normalized non-empty); naive timestamps rejected (tz-aware ISO-8601 required); `approved_output_root` must EQUAL the protocol-locked root, resolve inside the repository, no traversal/symlink escape; unknown approval fields fail closed. ACTUAL byte verification functions added with dependency-injected repository roots: per-arm generator SHA, TRAIN/VAL pair SHAs, pinned ImageNet weight artifact SHA (refuses while UNRESOLVED), runner commit, and dirty tracked worktree rejection (real synthetic git repo test). `--execute` still terminates with `SCIENTIFIC_LOOP_NOT_IMPLEMENTED` after every gate.

## 7. REVISED PROTOCOL

```text
PROTOCOL_SCHEMA:            P0_PROTOCOL_V1_1
implementation_revision:    P0_2_1
supersedes_protocol_sha256: b63f98af8e37a294b45ea6686282e5f392b4a26ff68179cf9ff86ea4a732e731
REVISED_PROTOCOL_SHA256:    096aeeb7e54e236838ebf97c296fee3b2423dd340d9ac9869b882e2569db8a4e
```
Scientific estimand, arms, seeds, SEOI (0.03 provisional) and RAW_ROC_AUC metric unchanged. Component schemas locked (seed P0_SEED_V1; sampler P0_SAMPLER_V1_1; order-hash P0_ORDERHASH_V1_1; model-state-hash P0_MODELSTATE_V1_1; run-manifest P0_RUN_MANIFEST_V1_1; approval-manifest P0_APPROVAL_MANIFEST_V1_1; loader-generator P0_LOADERGEN_V1_1). U_PUBLISHED execution path set to the established local materialized copy (`reproduction/workspace_prichexy/networks/generator_lowest_total_loss_mu_0.01.pth`). Human ratification remains REQUIRED because the protocol hash changed; authorization_status stays NOT_AUTHORIZED.

## 8. GOLDEN SEED VECTORS (hard-coded literals in tests)

```text
derive_seed(42, "attacker_weight_init")     = 3182366824493050920
derive_seed(42, "train_order")              = 1168295852399073028
derive_seed(42, "dataloader_worker_base")   = 7923226083686500895
derive_seed(67, "attacker_weight_init")     = 4526986779586776147
```

## 9. TEST EXECUTION AND RESULTS

pytest is NOT installed in this offline environment (no package index); reported accurately and all five standalone files were executed explicitly:

```text
CUDA_VISIBLE_DEVICES="" python reproduction/p0_bridge/tests/test_seed_contract.py          → ALL PASS (7)
CUDA_VISIBLE_DEVICES="" python reproduction/p0_bridge/tests/test_generator_guard.py        → ALL PASS (9)
CUDA_VISIBLE_DEVICES="" python reproduction/p0_bridge/tests/test_deterministic_sampler.py  → ALL PASS (10)
CUDA_VISIBLE_DEVICES="" python reproduction/p0_bridge/tests/test_manifest_io.py            → ALL PASS (10)
CUDA_VISIBLE_DEVICES="" python reproduction/p0_bridge/tests/test_runner_gate.py            → ALL PASS (15)
CUDA_VISIBLE_DEVICES="" python -m py_compile <all 11 modified/new .py files>               → COMPILE_OK
```

SYNTHETIC_CPU_TESTS: 51/51. The mandatory REAL `num_workers=1` DataLoader test ran and succeeded in this environment (worker process started, exact deterministic order reproduced).

Adversarial regression probes converted to production-code tests (all failing-before → fixed-after):
MISMATCHED_INTERNAL_IDENTITY_ACCEPTED → rejected via directory/manifest binding;
ONE_ARM_SCREEN_ACCEPTED → rejected (identity-count enforcement);
PREEXISTING_PREDICTION_ACCEPTED → fresh-output claim rejects stale outputs;
EMPTY_OUTPUT_AND_BLANK_APPROVER_ACCEPTED → blank approver + empty-dir rejections;
DERIVE_SEED_CALLED_BUT_NOT_IMPORTED → explicit imports + real num_workers=1 run;
plus: main-process global RNG byte-unchanged across loader construction/iteration; partial & duplicate order sequences rejected; dirty tracked worktree rejected (production git path exercised); actual artifact byte-hash mismatch rejected.

## 10. IMAGENET WEIGHT ARTIFACT STATUS

```text
IMAGENET_WEIGHT_ARTIFACT_STATUS: UNRESOLVED_BLOCKER
```
Exact torchvision weight-bundle identifier, local path and SHA-256 are not yet established; encoded as `UNRESOLVED_BLOCKER` in the locked protocol; `verify_all_artifacts` refuses execution while unresolved. No hash invented.

## 11. LIMITATIONS

The scientific loop remains unimplemented by design; artifact gates were exercised against synthetic injected roots only (never real governed artifacts); pytest unavailable offline (standalone harnesses used, structure import-compatible); full 52-identity aggregation evidence used the locked seed range on synthetic manifests.

## 12. SCIENTIFIC EXECUTION STATEMENT

No scientific model was instantiated or trained; no checkpoint loaded; no image or pair file opened; no Direction B artifact evaluated; CUDA never initialized; no approval manifest created; nothing merged; no pull request opened.

---

## 13. MACHINE-READABLE VERDICT

```text
P0_2_1_CLOSEOUT_STATUS: PASS
EXTERNAL_REVIEW_BLOCKERS_ADDRESSED: YES

PROTOCOL_SCHEMA: P0_PROTOCOL_V1_1
PROTOCOL_AUTHORIZATION_STATUS: NOT_AUTHORIZED
PREVIOUS_PROTOCOL_SHA:
b63f98af8e37a294b45ea6686282e5f392b4a26ff68179cf9ff86ea4a732e731
REVISED_PROTOCOL_SHA: 096aeeb7e54e236838ebf97c296fee3b2423dd340d9ac9869b882e2569db8a4e

WORKER_DERIVE_SEED_IMPORT: PASS
EXPLICIT_DATALOADER_GENERATOR: PASS
REAL_NUM_WORKERS_1_TEST: PASS
MAIN_GLOBAL_RNG_UNCHANGED: PASS
ORDER_HASH_COMPLETENESS_VALIDATION: PASS

ALL_SUBMODULES_EVAL_GUARD: PASS
NESTED_INFERENCE_TENSOR_REJECTION: PASS
MODEL_STATE_HASH_SCHEMA: P0_MODELSTATE_V1_1

MANIFEST_DIRECTORY_IDENTITY_BINDING: PASS
SCREEN_EXACT_IDENTITY_COUNT_10: PASS
FULL_EXACT_IDENTITY_COUNT_52: PASS
ONE_ARM_AGGREGATION_REJECTED: PASS
STALE_OUTPUT_DIRECTORY_REJECTED: PASS

APPROVER_NONEMPTY_VALIDATION: PASS
APPROVAL_TIMESTAMP_VALIDATION: PASS
OUTPUT_ROOT_PROTOCOL_BINDING: PASS
ACTUAL_ARTIFACT_HASH_GATE: PASS
DIRTY_WORKTREE_REJECTION: PASS

IMAGENET_WEIGHT_ARTIFACT_STATUS:
UNRESOLVED_BLOCKER

SCIENTIFIC_EXECUTION_LOOP:
NOT_IMPLEMENTED

SYNTHETIC_CPU_TESTS:
51/51

CUDA_INITIALIZED: NO
REAL_IMAGES_OPENED: NO
REAL_PAIR_FILES_OPENED: NO
REAL_CHECKPOINTS_LOADED: NO
DIRECTION_B_EVALUATION: NONE

P0_EXECUTION_AUTHORIZATION: NONE
P0_GPU_AUTHORIZATION: NONE
FILES_CHANGED: 13
EXISTING_GOVERNED_FILES_MODIFIED: 0
NEXT_REQUIRED_ACTION: EXTERNAL_RE_REVIEW
```

*Stop for external re-review.*
