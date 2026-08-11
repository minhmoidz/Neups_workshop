# STEP 2B.3 — FINAL NARROW PATCH AFTER SCIENTIFIC REVIEW

## 1. Reviewed base commit

- Reviewed base (HEAD at start): `009101cc909b177a576f9dd698f6e94f41552ea9` (`009101c`)
- Branch: `main`
- Working tree at start: clean
- The Scientist's review document `02F_FINAL_STEP2_SCIENTIFIC_REVIEW.md` was NOT present in the repository; the three required changes were implemented directly from the review instructions.
- Frozen artifacts preserved verbatim (not modified):
  - `research_agent/01_ADAPTIVE_REID_PROTOCOL.md`
  - `research_agent/01B_PROTOCOL_AMENDMENT.md`
  - `research_agent/topk_frozen_list.csv`

## 2. D-1 generator-content pinning

Defect: `run_signature` included only `generator_checkpoint_path`; same path + different bytes produced the same signature, permitting stale runs trained against `G_old` to be reused after the path was overwritten by `G_new`.

Fix in `run_adaptive_reid_arm.py::run_signature`:

```
run_signature = {
    ...
    generator_checkpoint_path,
    generator_checkpoint_hash,   # NEW: sha256 of the actual checkpoint bytes
    ...
}
```

`generator_checkpoint_hash` is computed by `_generator_hash(args)` -> `diag.sha256_file(checkpoint)`, i.e. from the real bytes only (never path, mtime, filename, or experiment name).

## 3. Generator-hash reuse cross-check

- `reuse_completed_run` now verifies the persisted `training_diagnostics.generator_checkpoint_hash` equals the current arm's `generator_checkpoint_hash` (from the signature). A stale mismatch forces retraining (`None`), never silent reuse.
- `verify_stage_e_generator_hash(run_dir, seed, current_hash)` refuses Stage E test evaluation of any run whose recorded generator digest does not equal the current arm digest (`RuntimeError`, loud, no silent provenance repair). Wired into `main()`'s `evaluate_test`.

## 4. D-1 mutation tests

- `test_D1_signature_binds_generator_content`: same path + same bytes -> identical signature (reuse compatible); same path + overwritten bytes -> digest and signature change (critical mutation test).
- `test_D1_stale_generator_prevents_reuse`: completed run recorded with `H_old`; current arm on `H_new` -> reuse rejected via BOTH the changed signature and the diagnostics cross-check.
- `test_D1_stage_e_refuses_stale_generator`: Stage E raises `RuntimeError` when recorded digest `H_old` != current `H_new`; allowed when matching.

Mutation power (bug present -> test FAILS): removing `generator_checkpoint_hash` from the signature made an overwritten-path signature compare equal (`s1 == s2 -> True`), so the mutation test asserting `s2 != s1` fails under the old bug.

## 5. D-2 tmp_path repair

Defect: `_td_path` used `hasattr(tmp_path, 'name')`, and `pathlib.Path` always has `.name` (the final component), so `/tmp/.../test_case` became `test_case`, creating relative directories inside the repository (order-dependent, non-repeatable).

Fix in `test_adaptive_reid_protocol.py::_td_path` — explicit type handling only:

```python
if isinstance(tmp_path, (str, os.PathLike)):
    return str(tmp_path)
if isinstance(tmp_path, tempfile.TemporaryDirectory):
    return tmp_path.name
raise TypeError(...)
```

Verified: `pathlib.Path('/tmp/example/test_case') -> '/tmp/example/test_case'` (absolute), and `TemporaryDirectory.name` -> absolute temp dir. Tests added:
- `test_D2_td_path_absolute_for_pathlib`
- `test_D2_td_path_absolute_for_tempdir`
- `test_D2_z2_tests_isolated` (re-runs `test_Z2_checkpoint_loadability_reflects_reality` and `test_Z2_partial_artifact_set_not_reused`)

Mutation power (old `.name` helper present -> test FAILS): the old helper returned `'test_case'` for a `Path`; the new test asserting the absolute path fails under the old bug.

## 6. Consecutive clean-suite verification

From a clean working tree (only the 3 patched source files modified), the STEP 2 adaptive protocol suite was run twice consecutively:

| run | result | `git status --porcelain` after run |
|-----|--------|--------------------------------------|
| run 1 | PASS (63 PASS lines) | `M adaptive_reid/topk.py`, `M run_adaptive_reid_arm.py`, `M test_adaptive_reid_protocol.py` |
| run 2 | PASS (63 PASS lines) | identical set |

- PASS lines are byte-identical between the two runs.
- No test-created directories/files appeared (no untracked `??` entries); the suite does not depend on leftovers from a previous invocation.

## 7. A-1 frozen Top-k hard pin

- Added pinned digest constant in `adaptive_reid/topk.py`:
  `FROZEN_TOPK_SHA256 = "4ebb6e15786b7c25eb4220521e5d70cf03ceb8f7ca480581dd89ef3945b24d44"`
- Added `load_frozen_topk_list_canonical(path, expected_sha256)`: requires the exact tracked path, fails hard (`FileNotFoundError`) if missing, computes SHA-256, fails hard (`ValueError`) on digest mismatch. Never regenerates.
- Production path (`run_adaptive_reid_arm.py::main`) now calls ONLY the canonical loader. The try/except regeneration fallback was removed.
- `protocol_and_frozen_hashes` now raises `RuntimeError` if the provenance frozen-Top-k digest is missing or != the pin, so provenance can never record a non-frozen digest.
- `build_frozen_topk_list` remains available ONLY as a development builder (used by dev/regression test `test_M_topk_frozen_list_build`); it is never invoked by the production/canonical path.

## 8. Frozen artifact SHA-256 verification

```
$ sha256sum research_agent/topk_frozen_list.csv
4ebb6e15786b7c25eb4220521e5d70cf03ceb8f7ca480581dd89ef3945b24d44  research_agent/topk_frozen_list.csv
```

Matches the Scientist's independently verified digest exactly. The CSV was NOT modified.

A-1 tests:
- `test_A1_canonical_load_accepts_correct_digest`
- `test_A1_missing_canonical_fails_hard`
- `test_A1_modified_byte_fails_hard` (single-byte tamper)
- `test_A1_canonical_mode_never_regenerates`
- `test_A1_provenance_digest_equals_pinned`

Mutation power (regeneration fallback present -> test FAILS): a simulated "missing artifact regenerates" path returned `'REGENERATED'` instead of raising, so the missing-canonical test asserting `FileNotFoundError` fails under the old behavior.

## 9. Security/protocol invariants preserved

Reconfirmed (all asserted by the existing suite plus the new tests):

1. Near-chance run: `VALID`, `near_chance=True` (not excluded).
2. Replacement: only on `NUMERICALLY_INVALID`.
3. Representative selection: validation-derived only.
4. One fresh seed: exactly one training invocation.
5. Non-stub Stage E: cannot fabricate test metric (`NotImplementedError`).
6. Sample SD: ddof=1.
7. Real training diagnostics: contain no test-derived fields.
8. Idempotent reuse: requires complete + loadable + now content-compatible (D-1) artifacts. D-1 strengthens, never weakens, compatibility.

## 10. Full regression results

- `test_adaptive_reid_protocol.py` (run 1): PASS
- `test_adaptive_reid_protocol.py` (run 2): PASS
- `test_operator_repair.py`: `STEP 1B REVIEW REMEDIATION: PASS`
- `test_grad_accum.py`: `ALL GRADIENT-ACCUMULATION REGRESSION TESTS PASSED`

New D-1/D-2/A-1 regression tests (13): all PASS.

## 11. Working-tree cleanliness

After both suite runs and all regression runs, `git status --porcelain` shows only the 3 intentionally modified source files plus the 2 new tracked report/diff artifacts; no untracked leftovers.

## 12. Files changed

- `adaptive_reid/topk.py` (A-1: pinned digest + canonical loader)
- `run_adaptive_reid_arm.py` (D-1: signature hash + reuse cross-check + Stage E gate; A-1: canonical production path)
- `test_adaptive_reid_protocol.py` (D-1/D-2/A-1 regression tests + `_td_path` fix)
- `research_agent/02G_FINAL_STEP2_PATCH.md` (this report)
- `research_agent/STEP2B3_FINAL_PATCH.diff` (full patch)

## 13. Remaining blockers

None. All three review items (D-1, D-2, A-1) are resolved.

## 14. Exact reproduction commands

```bash
# clean base
git checkout 009101cc909b177a576f9dd698f6e94f41552ea9   # expected base
pwd && git status && git branch --show-current && git rev-parse HEAD

# frozen artifact must match the pin
sha256sum research_agent/topk_frozen_list.csv
# -> 4ebb6e15786b7c25eb4220521e5d70cf03ceb8f7ca480581dd89ef3945b24d44

# D-2 gate: two consecutive clean runs + clean status
.venv/bin/python test_adaptive_reid_protocol.py && git status --porcelain
.venv/bin/python test_adaptive_reid_protocol.py && git status --porcelain

# full regression
.venv/bin/python test_operator_repair.py
.venv/bin/python test_grad_accum.py

# commit + push
git add -A
git commit -m "Fix final STEP 2 reuse and protocol test gates"
git push origin main
git status --porcelain   # must be empty
sha256sum research_agent/topk_frozen_list.csv
```
