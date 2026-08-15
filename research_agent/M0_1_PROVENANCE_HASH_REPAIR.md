# M0.1 — Checkpoint Provenance Hash Repair

**Date:** 2026-08-15 · **Branch:** `research/method-restart`
**Type:** provenance metadata repair only. No training, no TEST access, no scientific
parameter change, no history rewrite.

## Issue discovered after M0 publish

A post-publish audit found incorrect placeholder/full SHA256 values inside the three
development configs committed in M0 (`6d9b5ba`). The generator hash was correct; the
classifier and verifier hashes were wrong placeholders.

## Incorrect config values (as committed in M0)

| Field | Config value (wrong) | Authoritative LFS SHA (correct) |
|---|---|---|
| `classifier_checkpoint_sha256` | `8ad15b38286f1d9cb2584873252c66e28a8f37b1eb17b0d63c12e1a1a1a1a1a1` | `8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663` |
| `verification_model_sha256` | `331efaed0c0412fbbefe2f9d1e19ea75de5732fba7bfd83d76d6d81d5c1b44c5b` | `331efaed0c0433c69941ddc003a14a936c688d94fd4ecfbefd34e53bfa7c051a` |

The generator hash was already correct and was NOT changed:
`4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153`.

## Verification source (self-verified, not prompt-trust)

Hashes were extracted independently from repository metadata:

- materialized file SHA256 (`sha256sum`) of each checkpoint, AND
- Git LFS pointer `oid sha256:` from the git index (`git cat-file` on the staged blob).

All three agreed with the authoritative values in this addendum.

## Files repaired

- `config_files/config_dev_restored_baseline.json`
- `config_files/config_dev_c4.json`
- `config_files/config_dev_c2c4.json`

Only the two `*_sha256` fields changed (6 lines each). No mu, operator, batch size,
epochs, feature loss, C2/C3 settings, seeds, or any scientific parameter changed.

## Test added

- `research_agent/m0_tests/test_m01_provenance_hashes.py`
- registered in `research_agent/m0_tests/run_all.py`

Non-vacuous behavior:
- config SHA must equal the materialized-file SHA256 when real bytes exist, OR
  the `oid sha256:` in the Git LFS pointer otherwise;
- FAILS on mismatch AND FAILS if a configured checkpoint path is absent
  (cannot become vacuously PASS because a large checkpoint is unavailable).

## Run results

- full M0 suite + M0.1 test: all PASS
- negative control (inject wrong hash) correctly FAILS the test.

## Declarations

- no scientific result affected
- no experiment run
- TEST untouched
- history preserved: the original M0 commit is left intact; this repair is a new commit.

## Commit

`git commit -m "M0.1: fix development checkpoint provenance hashes"`