# P0.1 PROTOCOL FEASIBILITY CLOSEOUT — CPU-ONLY EVIDENCE TASK

**Date:** 2026-08-22
**Status:** AUDIT + PROTOCOL-REMEDIATION SPECIFICATION ONLY. No GPU/CUDA use, no model/checkpoint loading, no image loading, no training/evaluation, no P0-runner implementation, no governed-code modification, no commit/push/merge/PR.

---

## 1. EXECUTIVE VERDICT

The audit **PASSES** and resolves the three execution blockers of the canonical P0 protocol lock with direct evidence:

1. **H1 PASS — TRAIN/VAL pools are patient-disjoint AND image-disjoint** (0 overlapping patients, 0 overlapping image paths; parser rule source-validated). The canonical protocol's blocker (lock-table field 12) is RESOLVED.
2. **H2 FAIL — the current attacker code does NOT guarantee paired data order across arms.** A locally created `torch.Generator` is never passed to the DataLoader; shuffle order depends on global RNG consumption history; realized order is not recorded. Logging alone cannot fix this — a future runner must use explicit independent per-arm generators.
3. **H3 FAIL — current governed code legally runs only attacker seed 42** (`dev_attacker.py:96–97` hard lock). Seeds 42–67 require a reproduction-only runner under `reproduction/p0_bridge/`; governed code must NOT be modified.
4. **H4 — the n=26 / δ=0.03 plan has a complete analytical operating-characteristic table** (below). It is adequate for equivalence at σ≤0.0385 but has limited directional power for true differences near the margin; no change to n or δ is proposed here.

Consequently: the earlier "READY_FOR_HUMAN_RATIFICATION" framing of the canonical lock is superseded insofar as it implied execution readiness. Design status advances to READY_FOR_IMPLEMENTATION_REVIEW; implementation NOT_STARTED; execution BLOCKED until a later reproduction-only implementation task passes CPU tests.

---

## 2. WORKTREE IDENTITY AND SAFETY

- Worktree `/tmp/p0-canonical-protocol-worktree`, branch `review/p0-canonical-protocol-20260821`, HEAD `a3513e5c3b3b5631838399fc14c9e708909fe923` — verified at start. `git status --short` empty (both reports are `.gitignore`d via the `reproduction/` rule; filesystem paths verified directly).
- Canonical report present; SHA-256 recorded before work:
  `60f1b972c32f75c26a8a6aa64432efa0156312828981e9113f754822ed020b75`
  and re-verified unchanged at the end (§13).
- Active Direction B process (PID 1875303) present at start and end; untouched. No `nvidia-smi`, no GPU inspection, no signaling.
- All Python used `CUDA_VISIBLE_DEVICES=""`; `torch.cuda.is_initialized()` asserted `False` in every script run. Clean-room restrictions honored: quarantined commits `c2ee268…`/`bf43d30…` not read; no prohibited frozen-evaluation artifact accessed; no recursive searches; no patient identifier or image path printed or persisted (aggregate counts and set digests only).

---

## 3. HYPOTHESIS H1 — TRAIN/VAL PATIENT DISJOINTNESS: **PASS**

Parser derivation (source-validated, not invented): pair rows are 3-field whitespace rows `image1 image2 label` loaded by `np.loadtxt` (`datasets/SiameseDataset.py:38,40`; `evaluator_common.py:732,781`); images resolve as `image_path + <filename>` (`SiameseDataset.py:70–71`). Patient identity convention used by this project's own code is the NIH filename prefix before the first underscore: `name.split('_')[0]` (`evaluator_common.py:250` and `:335`). PATIENT_IDENTITY_PARSER: VERIFIED.

Aggregate evidence (CPU-only parse of the two authorized files; no image opened; no identifier printed):

| Quantity | TRAIN pool | VAL pool |
|---|---|---|
| Pair rows | 10,000 | 2,000 |
| Endpoint occurrences | 20,000 | 4,000 |
| Unique image paths | 18,089 | 3,484 |
| Unique patients | 9,053 | 1,742 |
| Duplicate pairs | 0 | 0 |
| Self-pairs | 0 | 0 |
| File SHA-256 | `3c535eed…394b268` | `9e33a081…24cba60b…f9fa9f7` |
| Sorted patient-ID-set SHA-256 | `723febfbb270afab9dca5e8050fce6a3376197ca06f237950e732a8ef9092dcf` | `b448ce935c9f8819f8806196e07b9917f3dfa37bd375fac11aa3c5ee0242121e` |

Cross-pool: **TRAIN ∩ VAL patient count = 0**; **TRAIN ∩ VAL image-path count = 0**. Members are not included by design.

Decision: **H1_PASS** (overlap = 0 with source-validated parsing). The canonical lock-table field 12 blocker is resolved. VERIFIED_BY_CPU_AUDIT

---

## 4. HYPOTHESIS H2 — CURRENT PAIRED-ORDER REPRODUCIBILITY: **FAIL**

### 4.1 Static source audit (exact citations)

| Question | Finding | Citation |
|---|---|---|
| When is `seed_all()` called? | Once, in `DevAttacker.__init__`, BEFORE loader construction and model init | `dev_attacker.py:120` |
| When is the Siamese model initialized? | AFTER loaders are constructed (line 131–134) but BEFORE any epoch iteration; ResNet-50 pretrained init consumes global RNG | `dev_attacker.py:139` |
| When is the DataLoader constructed? | Inside `build_dev_loaders`, called from `__init__` | `dev_attacker.py:132`; `evaluator_common.py:617–642` |
| Which RNG does shuffle consume? | The GLOBAL CPU RNG: a local `torch.Generator` is created and seeded (`gen = torch.Generator(); gen.manual_seed(seed)`) but `utils.get_data_loader(...)` is called WITHOUT any generator argument, so the generator is never passed to the DataLoader or sampler | `evaluator_common.py:626–627` (creation) vs `:631–637` (call drops it) |
| Is the locally created generator actually passed? | **NO** | same citations |
| Does model init consume the ordering RNG? | YES — permutations are drawn lazily per epoch from the global RNG, after model-init consumption | PyTorch DataLoader shuffle semantics; consumption chain above |
| Worker seeding | No explicit worker-seed derivation anywhere in the path (`num_workers=0` default masks it, but the contract is uncontrolled) | `evaluator_common.py:617,636` |
| Realized sampler-order recording | **ABSENT** — `DevAttacker` records losses/checkpoints only; no index-order hash | `dev_attacker.py:215–287` |

### 4.2 CPU-only synthetic proof (integer dataset; no medical model imported)

All runs under `CUDA_VISIBLE_DEVICES=""`, `torch.cuda.is_initialized()` False throughout. Order hash = SHA-256 over the realized permutation of indices 0–63.

| Case | Setup | Result |
|---|---|---|
| 1 | Same global seed (7), identical prior RNG consumption, global-RNG shuffle | hashes EQUAL (`29688725b6f6f882` twice) → reproducible only when nothing else consumes RNG |
| 2 | Same global seed (7), DIFFERENT prior consumption (`randn(1000)` vs `randn(5000)` before iterating) | hashes DIFFER (`74d532a06fea874e` vs `95a6ec3f95380fdc`) → **order depends on prior global-RNG consumption** |
| 3 | Explicit independent `torch.Generator`s, same order seed (99), different prior global consumption | hashes EQUAL (`0792fd267fe591c7` twice) → explicit generator removes the coupling |
| 4 | Explicit generators, different order seeds (99 vs 100) | hashes DIFFER (`0792fd267fe591c7` vs `91f94a85150f7e4b`) → order seed controls order |

### 4.3 Decision

**H2_FAIL.** CURRENT_ORDER_GENERATOR_PASSED_TO_DATALOADER: NO. CURRENT_ORDER_DEPENDS_ON_GLOBAL_RNG: YES. CURRENT_ORDER_HASH_LOGGING: ABSENT. CURRENT_PAIRED_ORDER_CONTRACT: FAIL.

Runtime logging alone is INSUFFICIENT: logging can detect divergence after the fact but cannot guarantee the paired-order contract, because the order is an uncontrolled function of global-RNG consumption history. Case 2 proves that two arms with the same seed can diverge if anything between seeding and iteration consumes different amounts of global randomness. VERIFIED_FROM_SOURCE + VERIFIED_BY_CPU_EXPERIMENT

---

## 5. CANONICAL SEED-DOMAIN SEPARATION SPECIFICATION (future design; NOT implemented)

Required sub-seeds per attacker seed `s ∈ {42..67}`, derived by domain-separated hashing:

```text
sub_seed(domain) = int.from_bytes(
    sha256(f"P0|schema_version=1|seed={s}|domain={domain}").digest()[:8], "big"
) mod 2^63
```

| Domain string | Purpose | Range |
|---|---|---|
| `weight` | weight_seed(s): attacker initialization | [0, 2^63) |
| `train_order` | train_order_seed(s): epoch-permutation generation | [0, 2^63) |
| `worker` | worker_seed(s): base for worker_init_fn = worker_seed(s)*k + worker_id | [0, 2^63) |
| `sensitivity` | sensitivity_analysis_seed (single global value, e.g., 20260822) | fixed |

Contract requirements:
1. Identical sub-seeds for BOTH generator arms at every seed s (paired design).
2. Separate, independent `torch.Generator` objects initialized to `train_order_seed(s)` for each arm; NO shared mutable generator object between arms.
3. Weight seeding applied immediately before attacker construction; an initial-attacker-state hash (canonical state-dict hash over sorted keys/dtypes/shapes/bytes) MUST be computed post-init and MUST be byte-identical across arms — proving identical initialization.
4. Workers: `num_workers>0` requires explicit `worker_init_fn` deriving each worker's seed from `worker_seed(s)`; `num_workers=0` also permitted but the rule must be stated either way.
5. Epoch permutation = pure function `(train_order_seed, epoch_index, dataset_length)` — e.g., a deterministic keyed permutation drawn from a fresh `torch.Generator` re-seeded to a domain-separated value `sha256(train_order_seed ‖ epoch_index)` per epoch, so no cross-epoch RNG-state dependence exists.
6. One order hash per (arm, seed, epoch), computed from sampler indices WITHOUT consuming the sampler twice (compute the permutation once, hash it, then serve it).
7. Fail-closed comparison: paired arm order hashes must match per (seed, epoch) before any batch is trained on; mismatch aborts the run.

SEED_DOMAIN_SEPARATION_SPEC: LOCKED (design-level).

---

## 6. HYPOTHESIS H3 — CURRENT MULTI-SEED SUPPORT: **FAIL**

Evidence:
- `dev_attacker.py:96–97`: scientific mode raises unless `attacker_seed == 42`.
- `evaluator_common.py:496–497`: frozen-config verification asserts `attacker_seed == 42` in the canonical attacker config.
- The seed IS a governed configuration field pinned by frozen config SHA (`FROZEN_ATTACKER_CONFIG_SHA`, `evaluator_common.py:79–80`); changing a CLI value alone would violate the frozen contract (documented governance route in S2_CONFIRMATORY_DESIGN_PROPOSAL.md §7).

```text
CURRENT_MULTI_SEED_SUPPORT: FAIL
CURRENT_SEED_HARD_LOCK: 42
GOVERNED_CODE_MODIFICATION_REQUIRED: NO   (governed code stays untouched)
REPRODUCTION_ONLY_RUNNER_REQUIRED: YES
```

Proposed future artifacts (NONE created now), all under `reproduction/p0_bridge/`: reproduction-only runner module; new frozen-style config artifact carrying the predeclared seed list with its own registered SHA; per §5 seed-domain utilities; output/manifest layer per §8. `research_agent/m2_dev/` remains unmodified.

---

## 7. HYPOTHESIS H4 — STATISTICAL LOCK ADEQUACY: COMPLETE TABLE

Analytical calculations only (noncentral-t / normal-theory TOST approximation; no stochastic simulation). n=26, df=25, α=0.05 one-sided ⇒ t_crit = 1.7081. δ=0.03 PROVISIONAL_SEOI_PENDING_HUMAN_RATIFICATION. σ=0.0385 is the existing Stage-A paired-difference SD used ONLY as a planning reference — not the known P0 SD.

### 7.1 Sensitivity table

| Paired SD σ | TOST power at true Δ=0 | Dir. power Δ=−0.04 | Δ=−0.05 | Δ=−0.06 | True Δ needed for ≈80% directional power beyond margin |
|---|---|---|---|---|---|
| 0.0200 | ≈1.000 | 0.798 | 0.9995 | 1.000 | −0.0400 |
| 0.0300 | 0.999 | 0.503 | 0.952 | 0.9995 | −0.0450 |
| 0.0385 | 0.977 | 0.361 | 0.824 | 0.987 | −0.0493 |
| 0.0500 | 0.823 | 0.257 | 0.633 | 0.908 | −0.0551 |

Symmetric interpretation holds exactly for positive deltas (+0.04/+0.05/+0.06 mirror the negative column values).

### 7.2 What n=26 can and cannot reliably distinguish

- CAN: establish practical equivalence with high power (≥97.7%) whenever true |Δ|=0 and σ ≤ 0.0385 (≥82% even at σ=0.05); reliably detect true differences of ≥0.05 beyond chance at σ ≤ 0.0385 (power ≥82%).
- CANNOT: reliably detect small true differences near the margin — at the planning-reference σ=0.0385, a true Δ=−0.04 (only 0.01 beyond the margin) has just 36% directional power; even Δ=−0.04 needs σ≤0.02 for 80% power. Differences inside (−0.03,+0.03) are unidentifiable from noise by design (that is what equivalence testing means here).
- No change to n or δ is made or proposed. If either changes, that requires NEW human ratification before any result exists.

### 7.3 Locked permutation sensitivity procedure

Primary inference remains paired t-based TOST and directional confidence bounds. Sensitivity analysis predeclared:

```text
Monte Carlo sign flips:      1,000,000
Monte Carlo seed:            20260822
Statistic:                   arithmetic mean of boundary-centered paired differences
P-value correction:          (extreme_count + 1) / (B + 1)
```

Four SEPARATE tests, defined independently:
1. Lower equivalence-bound test: statistic uses centered differences `Δ_s + δ` (flipping signs).
2. Upper equivalence-bound test: statistic uses `Δ_s − δ`.
3. U_PUBLISHED-better boundary test (one-sided, negative direction).
4. D_BDEV-better boundary test (one-sided, positive direction).

Validity statement: sign-flip inference requires symmetry/exchangeability of the boundary-centered paired differences under the null; it is a SENSITIVITY ANALYSIS and never a replacement for the primary t-based analysis. PERMUTATION_SENSITIVITY_SPEC: LOCKED. POWER_SENSITIVITY_TABLE: COMPLETE.

---

## 8. OUTPUT-MANIFEST CORRECTION (future design; NOT implemented)

The previously sketched shared append-only `manifest.jsonl` is superseded by:

1. ONE immutable `run_manifest.json` per `<arm>/<seed>/` directory.
2. Atomic persistence: write to unique temp file in the same directory → `fsync` → atomic `rename` to final name; never append concurrently to a shared file.
3. Deterministic `aggregate_manifest.json` generated ONLY after validating ALL expected per-run manifests.

Aggregate-manifest validation rules:
- sort records by (arm, seed);
- validate all expected 52 identities (2 arms × 26 seeds);
- reject duplicate identities;
- reject missing runs;
- reject extra/unexpected runs;
- bind every record to checkpoint SHA, pair-file SHAs, config hash, code commit, environment id, seed-domain sub-seeds, and per-epoch order hashes;
- never silently reuse stale output (freshness marker checked; any mismatch fails closed).

OUTPUT_MANIFEST_SPEC: LOCKED (design-level).

---

## 9. STATUS CORRECTION

This report explicitly supersedes ONLY the earlier premature execution-readiness status of `P0_CANONICAL_PROTOCOL_LOCK_2026-08-21.md` (its `READY_FOR_HUMAN_RATIFICATION` framing, insofar as it implied readiness to execute). That file itself is NOT modified (SHA verified unchanged in §13). All other content of the canonical lock remains in force, now amended by: H1 resolution (field 12 blocker cleared), H2/H3 remediation requirements (§5, §6), H4 operating characteristics (§7), and the manifest correction (§8).

```text
P0_1_AUDIT_STATUS: PASS
P0_PROTOCOL_DESIGN_STATUS: READY_FOR_IMPLEMENTATION_REVIEW
P0_IMPLEMENTATION_STATUS: NOT_STARTED
P0_EXECUTION_STATUS: BLOCKED
```

A PASSING audit here legitimately proves the current runner is UNSUITABLE for P0 as-is; execution remains blocked until a later reproduction-only implementation task passes its own CPU-only tests and receives human authorization.

---

## 10. MACHINE-READABLE FINAL VERDICT

```text
P0_1_AUDIT_STATUS: PASS
PATIENT_IDENTITY_PARSER: VERIFIED
TRAIN_VAL_PATIENT_OVERLAP_COUNT: 0
TRAIN_VAL_PATIENT_DISJOINTNESS: VERIFIED

CURRENT_ORDER_GENERATOR_PASSED_TO_DATALOADER: NO
CURRENT_ORDER_DEPENDS_ON_GLOBAL_RNG: YES
CURRENT_ORDER_HASH_LOGGING: ABSENT
CURRENT_PAIRED_ORDER_CONTRACT: FAIL

CURRENT_MULTI_SEED_SUPPORT: FAIL
CURRENT_SEED_HARD_LOCK: 42
REPRODUCTION_ONLY_RUNNER_REQUIRED: YES

SEED_DOMAIN_SEPARATION_SPEC: LOCKED
PERMUTATION_SENSITIVITY_SPEC: LOCKED
POWER_SENSITIVITY_TABLE: COMPLETE
OUTPUT_MANIFEST_SPEC: LOCKED

P0_CANONICAL_PREVIOUS_READY_STATUS: SUPERSEDED
P0_PROTOCOL_DESIGN_STATUS: READY_FOR_IMPLEMENTATION_REVIEW
P0_IMPLEMENTATION_STATUS: NOT_STARTED
P0_EXECUTION_STATUS: BLOCKED
P0_GPU_AUTHORIZATION: NONE

MODEL_OR_CHECKPOINT_EXECUTION: NONE
IMAGE_LOADING: NONE
GPU_OR_CUDA_USE: NONE
FILES_CREATED: 1
EXISTING_FILES_MODIFIED: 0
COMMIT_OR_PUSH: NONE
ACTIVE_TRAINING_PROCESS_TOUCHED: NO
NEXT_REQUIRED_ACTION: EXTERNAL_HUMAN_REVIEW
```

---

## 13. FINAL VERIFICATION RECORD (completed)

1. Canonical report SHA-256 recomputed after all work: `60f1b972c32f75c26a8a6aa64432efa0156312828981e9113f754822ed020b75` — IDENTICAL to the pre-task value; canonical file unmodified.
2. Exactly ONE new file created by this task: `reproduction/reports/P0_1_PROTOCOL_FEASIBILITY_CLOSEOUT_2026-08-22.md` (verified by direct filesystem listing; the `reproduction/` `.gitignore` rule hides it from `git status`, so filesystem verification was used, not `git status`).
3. Process-presence re-run: Direction B v1 process still active (PID 1875303); untouched throughout.
4. No CUDA context was ever initialized: every Python invocation ran under `CUDA_VISIBLE_DEVICES=""` and asserted `torch.cuda.is_initialized() is False` before and after work.
5. Nothing committed or pushed.

*Stop and wait for external human review.*
