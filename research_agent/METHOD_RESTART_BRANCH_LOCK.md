# METHOD RESTART BRANCH LOCK

**Created:** 2026-08-15
**Status:** locked reference for the method-restart lineage.

| Field | Value |
|---|---|
| base branch | `original-upstream` |
| base commit | `29245d1f71571898d9527417df4ae3f63a8695f6` |
| method branch | `research/method-restart` |
| first method-restart commit | `6d9b5ba76e1312e54ec1c10934d3a9bfd7f42f43` (M0) |

## Restored baseline

- released PriCheXy-Net generator (`networks/generator_lowest_total_loss_mu_0.01.pth`,
  SHA256 `4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153`)
- legacy operator (== upstream, bit-for-bit; diff 0.0 per `operator_audit_results.json`)
- mu = 0.01

## Reproduction context (NOT development decisions)

- historical paper reproduction (R1-FINAL, 10 seeds): ≈ 0.6080 ± 0.0435, n=10
- historical paper result: ≈ 0.577 ± 0.040
- both are context only and MUST NOT be used for method development decisions.

## TEST policy

- official TEST: **CLOSED FOR METHOD DEVELOPMENT**
- development uses TRAIN + VALIDATION only.

## Current research phase

- M0: **PASS** (26/26 M0 tests, TEST untouched, no training performed)
- M1: NOT YET EXECUTED

## Next allowed step

- freeze validation-only C4 development protocol (M1)

## Explicit declarations

- no new anonymizer training was performed during branch creation
- no TEST was accessed
- no dataset / model checkpoint was newly committed (checkpoints are referenced by
  path + SHA256 only)