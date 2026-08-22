# P0.2 REPRODUCTION-ONLY RUNNER IMPLEMENTATION — SYNTHETIC CPU TESTS ONLY

**Date:** 2026-08-22
**Status:** IMPLEMENTATION + SYNTHETIC CPU TESTS ONLY. No real generator/attacker checkpoint loaded, no medical image opened, no training/evaluation, no GPU/CUDA use, no commit/push. A PASS here means only `READY_FOR_EXTERNAL_SOURCE_REVIEW`.

---

## 1. WORKTREE AND SOURCE IDENTITY

- Worktree `/tmp/p0-canonical-protocol-worktree`, branch `review/p0-canonical-protocol-20260821`, HEAD `a3513e5c3b3b5631838399fc14c9e708909fe923` — verified at start and unchanged.
- Active Direction B process (PID 1875303) present at start and end; untouched.
- Clean-room: quarantined commits `c2ee268…`/`bf43d30…` not read; no prohibited frozen-evaluation artifact accessed; no recursive repository search.
- All Python executed under `CUDA_VISIBLE_DEVICES=""`; `torch.cuda.is_initialized()` asserted `False` in every session.

## 2. UNCHANGED PRE-EXISTING REPORT HASHES

| Report | SHA-256 (before == after) |
|---|---|
| `reproduction/reports/P0_CANONICAL_PROTOCOL_LOCK_2026-08-21.md` | `60f1b972c32f75c26a8a6aa64432efa0156312828981e9113f754822ed020b75` |
| `reproduction/reports/P0_1_PROTOCOL_FEASIBILITY_CLOSEOUT_2026-08-22.md` | `ccc8b87b6ba457835ba2ad0063473327ac34208e93c701bae595e8a043a4ab12` |

Recomputed at the end (§13): identical.

## 3. CREATED-FILE LIST (exactly the 13 allowlisted paths)

```text
reproduction/p0_bridge/__init__.py
reproduction/p0_bridge/protocol_v1.json
reproduction/p0_bridge/seed_contract.py
reproduction/p0_bridge/deterministic_sampler.py
reproduction/p0_bridge/generator_guard.py
reproduction/p0_bridge/manifest_io.py
reproduction/p0_bridge/run_p0_bridge.py
reproduction/p0_bridge/tests/test_seed_contract.py
reproduction/p0_bridge/tests/test_deterministic_sampler.py
reproduction/p0_bridge/tests/test_generator_guard.py
reproduction/p0_bridge/tests/test_manifest_io.py
reproduction/p0_bridge/tests/test_runner_gate.py
reproduction/reports/P0_2_CPU_ONLY_IMPLEMENTATION_REPORT_2026-08-22.md   (this file)
```

No caches, no outputs, no extra files (`__pycache__` removed after runs). No existing file modified.

## 4. BLOCKER → IMPLEMENTATION MAP

| P0.1 blocker | Implementation | Evidence |
|---|---|---|
| H2 FAIL: local generator never passed to DataLoader | `deterministic_sampler.build_permutation`: pure function of (schema, master seed, epoch, N) via fresh local `torch.Generator`; `make_paired_dataloader` builds `shuffle=False` + fixed-order sampler per arm; fail on `shuffle=True`+sampler is structural (shuffle hardcoded False) | tests `test_prior_global_rng_consumption_has_no_effect`, `test_paired_dataloaders_same_order_no_global_shuffle` |
| H2 FAIL: order depends on global RNG | permutation seed = `derive_epoch_order_seed` (SHA-256 domain-separated); no global RNG read | `test_reproducible_and_rng_independent`, sampler prior-consumption test |
| H2 FAIL: realized order not recorded | `order_hash` (P0_ORDERHASH_V1) + `expected_epoch_order_hashes`; per-epoch hashes bound into run manifests and compared across arms in aggregation | manifest paired-mismatch tests |
| H3 FAIL: seed hard-lock 42 | multi-seed contract owned by this package; seeds 42–67 derive cleanly; governed code untouched (`GOVERNED_SEED42_CODE_MODIFIED: NO`) | `test_domains_differ`, `test_arms_receive_identical_seeds` over full range |

## 5. SEED DERIVATION SPECIFICATION AND GOLDEN VECTORS

`P0_SEED_V1`: message = `"P0_SEED_V1|" + str(master_seed) + "|" + domain` (UTF-8, literal `|` delimiters); derived = big-endian int of `sha256(message)[0:8]` mod 2^63. Rejects bools/negatives/floats/ambiguous domains via regex `^[A-Za-z0-9_]{1,64}$`. No Python `hash()`, no RNG-state dependence.

Golden vectors (locked by `tests/test_seed_contract.py::test_golden_values_stable`):
```text
derive_seed(42,"attacker_weight_init") =
  int.from_bytes(sha256(b"P0_SEED_V1|42|attacker_weight_init")[:8],"big") % 2**63
derive_seed(42,"train_order"),
derive_seed(67,"attacker_weight_init")  pinned analogously
```
Epoch order seed additionally binds schema `P0_SAMPLER_V1`, epoch index, dataset length. `seed_everything_for_attacker_construction(weight_seed)` seeds random/numpy/torch-CPU only; CUDA never seeded.

## 6. SAMPLER ALGORITHM AND ORDER-HASH SERIALIZATION

Permutation: fresh CPU `torch.Generator` seeded from the epoch-specific derived seed; `torch.randperm(N, generator=g)`; complete coverage, no duplicates (tested). Order hash `P0_ORDERHASH_V1`: SHA-256 over schema-prefix bytes ‖ `struct.pack("<q", dataset_length)` ‖ `struct.pack("<q", epoch_index)` ‖ each index as `<q`. Master seed deliberately NOT embedded (spec fields only) so hashes are directly comparable across arms; computing the hash consumes no sampler state (tested).

## 7. ATTACKER-INITIALIZATION HASH CONTRACT

`generator_guard.canonical_model_state_hash` — versioned prefix `P0_MODELSTATE_V1`; sorted (name, dtype, shape, contiguous-CPU-bytes) entries; no pickle; insertion-order independent (reverse-order state-dict test passes). Future runner must require byte-equal initial hashes between arms per seed; ImageNet-weight provenance identifier + local file hash must appear in the approved execution manifest; automatic weight download FORBIDDEN at execution time (fail-closed if artifact unavailable).

## 8. GENERATOR STATE-GUARD EVIDENCE

`protected_forward(generator, fn, *args)`: pre-check eval-mode + all params frozen; snapshot training topology + buffer bytes; run under `torch.no_grad()` (never `inference_mode` for attacker-consumed outputs); post-verify flags/buffers unchanged; reject inference tensors; raise `GeneratorStateMutationError` — detection only, never silent restore. Negative control proves train-mode BatchNorm drifts under `no_grad`; protected path proves zero drift; downstream toy-attacker backward succeeds with zero generator grads.

## 9. OUTPUT/MANIFEST DESIGN

One immutable `run_manifest.json` per `<arm>/<seed>/`; atomic temp+fsync+rename; pre-existing final rejected. Identity binds protocol schema+SHA, runner commit, arm/seed, derived seeds, generator role/path/SHA, pair hashes, initial-attacker hash, per-epoch order hashes, best/stop epochs, score direction, predictions hash, environment provenance, status, timestamps. `aggregate_manifest.json` deterministic (sorted seed→arm), validates 52 full / 10 screen identities, rejects missing/duplicate/unexpected identities, stale protocol hashes, paired order-hash or init-hash disagreement; atomic write.

Protocol file `protocol_v1.json` locks everything required by §7 of the tasking; canonical serialization = `json.dumps(sort_keys=True, separators=(",",":"))+"\n"`; **protocol_sha256 = `b63f98af8e37a294b45ea6686282e5f392b4a26ff68179cf9ff86ea4a732e731`** (validate-only output).

## 10. RUNNER AUTHORIZATION GATES

Import has no side effects and imports no scientific module (subprocess-isolated test). `--validate-protocol-only`: schema/hash/seeds/identities/path-syntax checks only; no CUDA, no data, no dirs. `--execute`: fails closed before ANY scientific import unless an external approval manifest supplies all 12 required fields with `authorization_status=APPROVED`, matching protocol SHA, runner commit, stage, seed list, checkpoint roles/hashes, pair hashes, SEOI, non-colliding output root, `active_process_clearance=true`, and no live `run_hardened_verifier` process. No bypass flags exist or are accepted (behaviorally tested). Scientific loop labeled `NOT_EXECUTED_REQUIRES_EXTERNAL_SOURCE_REVIEW`.

## 11. TEST INVENTORY AND RESULTS

Environment note: pytest is unavailable in this offline environment (pip has no index), so the five test files are standalone-runnable pure-Python harnesses (also importable as plain modules). Invoked exactly per tasking intent:

```text
CUDA_VISIBLE_DEVICES="" python reproduction/p0_bridge/tests/<file>
```
plus `python -m py_compile` syntax compilation of ALL 11 new Python files → `COMPILE_OK`.

| File | Tests passed |
|---|---|
| test_seed_contract.py | 7/7 (golden values, domain separation, reproducibility/RNG independence, arm identity, invalid inputs incl. bool/float/negative/bad-domain, epoch-seed binding, seeding w/o CUDA) |
| test_generator_guard.py | 6/6 (BN-drift negative control, protected no-drift, pre-check rejections, mutation injection detected, downstream backward + zero gen grads + non-inference tensor, toy-model init-hash contract incl. insertion-order independence) |
| test_deterministic_sampler.py | 9/9 (cross-sampler identity, prior-RNG immunity, seed/epoch sensitivity, completeness, hash stability/sensitivity/statelessness, expected-hashes helper, worker determinism, paired DataLoader exact order) |
| test_manifest_io.py | 8/8 (atomic write, pre-existing rejection, duplicate/missing/unexpected identity, stale protocol, paired order mismatch, paired init mismatch, deterministic aggregate + collision rejection) |
| test_runner_gate.py | 11/11 (no side-effect import, clean validate mode, behavioral bypass-flag rejection, missing-manifest refusal, NOT_AUTHORIZED refusal, protocol/seed-list/stage/checkpoint mismatch refusals, active-process gate, validate-mode CUDA-free) |
| **Total** | **41/41 PASS** |

## 12. NEGATIVE-CONTROL RESULTS

- Train-mode BatchNorm mutates buffers even under `no_grad` (motivates the guard) — demonstrated.
- Injected buffer mutation inside a protected forward is DETECTED and raised (initially missed because a zero-initialized `running_mean.mul_(1.01)` is a no-op; corrected to `.add_(0.5)` — the detector itself was correct).
- Same master seed + different prior global consumption changes legacy-style order (carried-over P0.1 finding), while the new sampler is immune.
- Every gate mutation (bad hash/stage/seeds/checkpoints/approval status) fails closed.

## 13. FINAL VERIFICATION

- Both pre-existing report SHAs recomputed after all work: identical to §2.
- Filesystem check: exactly the 13 allowlisted files exist under `p0_bridge/` + this report; no `__pycache__`; no existing file modified.
- CUDA never initialized in any session; no real image/pair-file/checkpoint touched.
- Process-presence re-run: Direction B v1 still active (PID 1875303); untouched.
- Nothing committed or pushed.

## 14. UNRESOLVED LIMITATIONS

1. The scientific training/evaluation loop itself is intentionally NOT implemented behind the authorization gate (`NOT_EXECUTED_REQUIRES_EXTERNAL_SOURCE_REVIEW`).
2. pytest unavailability (offline env): harnesses are standalone runners; a future reviewer may re-run them under pytest if desired — structure is compatible.
3. Real-data path (pair-file loading, deformation operator, ResNet-50 Siamese construction) remains to be implemented in the separately reviewed execution task, including torchvision weight-bundle provenance pinning.
4. The approval manifest format is enforced but no approval exists or was created.

---

## 15. MACHINE-READABLE VERDICT

```text
P0_2_IMPLEMENTATION_STATUS: PASS
P0_2_SOURCE_SCOPE: REPRODUCTION_ONLY
P0_2_PROTOCOL_SCHEMA: P0_PROTOCOL_V1

H2_GLOBAL_RNG_DEPENDENCE_REMOVED: YES
EXPLICIT_DOMAIN_SEPARATED_SEEDS: PASS
DETERMINISTIC_EPOCH_SAMPLER: PASS
PAIRED_ORDER_HASH_CONTRACT: PASS
PAIRED_INITIAL_ATTACKER_HASH_CONTRACT: PASS

H3_MULTI_SEED_SUPPORT_IMPLEMENTED: YES
SUPPORTED_SCREEN_SEEDS: 42_43_44_45_46
SUPPORTED_FULL_SEEDS: 42_THROUGH_67
GOVERNED_SEED42_CODE_MODIFIED: NO

GENERATOR_EVAL_GUARD: PASS
BATCHNORM_NO_DRIFT_TEST: PASS
GENERATOR_NO_GRAD_TEST: PASS
DOWNSTREAM_ATTACKER_BACKWARD_TEST: PASS
TORCH_INFERENCE_MODE_USED_FOR_GENERATOR_OUTPUT: NO

IMMUTABLE_PER_RUN_MANIFEST: PASS
DETERMINISTIC_AGGREGATION: PASS
OUTPUT_COLLISION_REJECTION: PASS
EXECUTION_APPROVAL_GATE: PASS
REAL_WEIGHT_AUTO_DOWNLOAD: FORBIDDEN

SYNTHETIC_CPU_TESTS: 41/41
CUDA_INITIALIZED: NO
REAL_IMAGES_OPENED: NO
REAL_PAIR_FILES_OPENED: NO
REAL_CHECKPOINTS_LOADED: NO
MODEL_TRAINING_OR_EVALUATION: NONE

P0_EXECUTION_AUTHORIZATION: NONE
P0_GPU_AUTHORIZATION: NONE
FILES_CREATED: 13
EXISTING_FILES_MODIFIED: 0
COMMIT_OR_PUSH: NONE
ACTIVE_TRAINING_PROCESS_TOUCHED: NO
NEXT_REQUIRED_ACTION: EXTERNAL_SOURCE_REVIEW
```

*Stop for external source review.*
