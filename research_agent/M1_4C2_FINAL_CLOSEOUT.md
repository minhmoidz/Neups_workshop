# M1.4c.2 FINAL CLOSEOUT — Classification Replay Integrity + Audit Hygiene

**Status**: CLOSEOUT RECORD (this file)
**Branch**: `audit/m2-final-certification`
**Audit head before**: `01d2011eb10606ed6fc7f7264ed41fa58449629e`
**Canonical branch**: `research/method-restart` @ `c6431310061c04e54dce82d30ae6e0ce24440562` (unchanged)
**Method**: NO method change. NO canonical write. NO full M2. NO TEST.

---

## 1. Scope

This hotfix closes a real production replay-integrity bug (P0 execution-integrity blocker)
found by independent source audit, plus a set of non-method audit-hygiene gaps:

- A1 classification replay writer/reader schema mismatch
- A2 replay must require exactly 14/14 pathologies
- A3 replay must compare each per-pathology AUC (not macro only)
- A4 replay tests must use production writer schema/path
- A5 `audit_operator_equivalence.py` dead/broken on the audited branch
- A6 historical 10-run SD reported with ddof=0 (must label or use ddof=1)
- A7 report incorrectly describes batch_size=16 as "8 pairs"
- A8 promotion diff contains unrelated historical/reproduction baggage
- A9 missing M1.4c.1 closeout report
- M1.4c.2a follow-up: `M1_4C2_TEST_INVENTORY.json` explicitly classified in promotion fileset, T214 dependency contradiction resolved (self-contained statistical check), and T215 strengthened to verify complete git coverage of all canonical→audit changes.

## 2. Exact audit input commit

`01d2011eb10606ed6fc7f7264ed41fa58449629e` (`audit/m2-final-certification`).

## 3. Independent ChatGPT + Claude Scientist findings

- Production writer `classify_val_dataset()` writes `<Pathology>` and `prob_<Pathology>`.
- `check_run_validity()` replayed with `true_<p>` / `pred_<p>` and only compared macro AUC when
  `len(replayed_aucs) == 14`. When 0 expected columns were found the replay block was silently
  skipped and a scientific run could still return VALID.
- Existing T195/T196 hand-created the wrong fake schema, so they passed without exercising the
  production writer schema.

## 4. Classification replay bug root cause

`research_agent/m2_dev/run_m2_s1.py` `check_run_validity()` used column names
`true_%s` / `pred_%s` which the production writer never emits, and gated the whole comparison
behind `if len(replayed_aucs) == 14`. Zero found columns ⇒ replay skipped ⇒ VALID returned.

## 5. Production writer schema (authoritative, unchanged)

`classify_val_dataset()` (in `research_agent/m2_dev/eval_classifier_val.py`) emits:

- ground-truth column: `<Pathology>`
- probability column: `prob_<Pathology>`
- `Image Index` column
- AUC CSV columns: `label`, `auc` (14 rows)
- `macro_auc = mean(14 per-pathology AUCs)`

## 6. Production replay reader schema after fix

`check_run_validity()` now replays using:

- `t_col = pathology`
- `p_col = 'prob_' + pathology`

for every entry of the single canonical `NIH_PATHOLOGIES` constant (A6 single-source rule).

## 7. Exact 14/14 fail-closed logic

- Every one of the 14 canonical pathologies must have both its ground-truth and its
  `prob_` column present in `classification_val_predictions.csv`; otherwise INVALID.
- Exactly 14 replayed AUCs are required: 0, 13, 15, or any unexpected count ⇒ INVALID.
- AUC CSV must contain exactly 14 rows, one per canonical pathology, no duplicates, no
  unknown pathology, no NaN/Inf ⇒ otherwise INVALID.

## 8. Per-class replay validation

For each pathology, `roc_auc_score(pred_df[pathology], pred_df['prob_' + pathology])` is
recomputed and compared (tolerance ≤ 1e-7) against BOTH:

- `c_res['auc_df']` (in-memory), and
- `classification_val_aucs.csv` (serialized).

Macro AUC = mean of the 14 replayed AUCs is compared against `c_res['macro_auc']` (≤ 1e-7).

## 9. Production end-to-end artifact tests

`research_agent/m0_tests/test_m14c2_closeout.py` (T201–T216):

- T210 `evaluate_classification_arm()` → real production files → `check_run_validity()` ⇒ VALID.
- T211 copies those exact files, removes `prob_Hernia`, updates SHA metadata ⇒ INVALID.
- Closes the writer → serializer → reader → validity loop.

## 10. Privacy replay regression

`privacy_val_predictions.npz` (`y_true`, `y_score`) replay ROC-AUC is re-run unchanged.
Added T212: tamper `y_score` (update SHA accordingly) ⇒ replay mismatch ⇒ INVALID.

## 11. Batch-size wording correction

Report corrected to: DataLoader `batch_size = 16` pair samples; each sample contains
`image1`, `image2`, pathology target, identity target; each batch holds up to 16 pairs =
up to 32 image tensors. Config unchanged.

## 12. SD convention correction

`upstream_10run_reproduction_results.json` now exposes clearly labelled:

- `std_auc_ddof0 = 0.028215762629424015` (population SD, historical field preserved)
- `std_auc_sample_ddof1 = 0.02974202527588046` (sample SD, ddof=1, n=10)

Recommended publication wording: mean = 0.81847, sample SD = 0.02974 (ddof=1, n=10),
or explicitly label ddof=0 as population SD. Raw run values are NOT altered.
T214 is self-contained and computes the sample SD and population SD directly from the historical
individual AUC metrics without forcing canonical promotion to depend on audit-only files.

## 13. audit_operator_equivalence disposition

`research_agent/audit_operator_equivalence.py` imports `build_sampling_grid` / `deform`
from `utils.utils`, which do not exist on the audited branch, and hardcodes the upstream
reproduction path `/home/minhtt/PriCheXy-Net_upstream_reproduction`. It is imported by NO
production code and NO test. Disposition: **AUDIT-ONLY** — retained on the audit branch,
excluded from canonical promotion (A5/A8). Production operator is NOT modified to satisfy it.

## 14. Promotion fileset

`research_agent/M1_4C2_PROMOTION_FILESET.json` classifies every file changed since canonical
head into `include_in_canonical_promotion` (25 files: production hardening, certification tests, current
provenance) and `retain_on_audit_branch_only` (13 files: historical/reproduction baggage, test inventories).
No forbidden historical runtime baggage is listed for promotion. T215 statically and dynamically (via git diff)
asserts 100% complete coverage of all changed files relative to canonical SHA `c6431310061c04e54dce82d30ae6e0ce24440562`.

## 15. Protocol authority reconciliation

- `M2_S1_EXECUTION_LOCK.json`: scientific method / frozen scientific execution choices (supreme).
- `M1_4C_CERTIFICATION_MANIFEST.json`: certification evidence derived from those choices.
- `PROTOCOL_AUTHORITY.md` updated to state this explicitly; the M1.4c manifest must NOT override
  scientific method hyperparameters. No self-referential lock hash was created.

## 16. Test results

### Full Certified Post-Fix Suite Execution (M1.4c.2 / M1.4c.2a)
- Total executed: **216 tests** across 17 test modules.
- Total passed: **216 / 216 PASS (100%)**.
- Total failed: **0 FAIL / 0 ERROR**.
- Total runtime: **5555.89 seconds (~92.6 minutes)**.
- Detailed machine inventory: recorded in `research_agent/M1_4C2_TEST_INVENTORY.json`.
- Closeout module (`test_m14c2_closeout.py` T201–T216): **16/16 PASS** in 91.07s.

### Baseline 01d2011 Retrospective Execution Status
- A retrospective baseline verification suite on commit `01d2011eb10606ed6fc7f7264ed41fa58449629e`
  (200 tests across 16 test modules) was initiated in detached worktree `/tmp/neups_m14c2_baseline` (PID `942426`).
- **Status at closeout**: Still actively executing in background (at `test_m14b_execution_integrity.py`, accumulated >53 CPU hours); completion is NOT fabricated.

## 17. Real preflight

`verify_scientific_dependencies('/home/minhtt/datasets/nih/images/')` PASS — real NIH images,
real checkpoints, real SHAs (see §18), configs, pair splits, metadata, classification VAL
fingerprints. No training, no scientific AUC.

## 18. Git-LFS checkpoint materialization status

Real binary model files (PK/ZIP torch checkpoints), NOT 133-byte LFS pointer text:

| Checkpoint | Size | SHA256 | LFS pointer |
|---|---|---|---|
| `networks/pretrained_generator_prichexy_net.pth` | 31,116,923 | `101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb` | false |
| `networks/pretrained_classifier.pth` | 29,322,467 | `8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663` | false |
| `networks/pretrained_verification_model.pth` | 95,202,467 | `331efaed0c0433c69941ddc003a14a936c688d94fd4ecfbefd34e53bfa7c051a` | false |

Frozen SHAs remain unchanged.

## 19. TEST firewall status

This closeout did not access `image_pairs_testing_5000.txt`; no `fold='test'`, no testing
loader, no `final_test`. `test_touched: false`. Firewall asserted via `TestFirewall(allow=False)`.

## 20. Remaining limitations

- Scientific M2-S1 execution (B_dev/C4 250 epochs, full attacker, S1/S2) has NOT been run.
- The end-to-end replay tests run on synthetic CPU data; scientific VAL artifact replay uses
  the same production writer/reader/serializer path.
- `audit_operator_equivalence.py` remains dead on the audited branch by design (historical only).

## 21. Final recommendation

PASS for closeout. **STOP** — wait for an independent final audit before any canonical
promotion or M2 launch. Do not merge; do not fast-forward canonical; do not push canonical.