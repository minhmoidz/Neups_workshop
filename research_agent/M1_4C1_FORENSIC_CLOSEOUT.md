# M1.4c.1 FORENSIC CLOSEOUT — Retrospective Record

> **Retrospective closeout record created during M1.4c.2 because the originally
> requested M1.4c.1 closeout file was missing.**
>
> This file is a factual record assembled from commits and artifacts that are verifiable
> on the audit branch. No historical timestamps or results are invented here; where a
> value is known only from a commit/artifact it is cited, and where it cannot be verified
> it is stated as unverifiable.

---

## 1. Purpose

The prior task requested `research_agent/M1_4C1_FORENSIC_CLOSEOUT.md` as the M1.4c.1
closeout record. It was not committed. This retrospective record reconstructs only the
facts that can be verified from the repository history and frozen artifacts.

## 2. Commit evidence

- M1.4c.1 forensic closeout commit (audit branch): `01d2011eb10606ed6fc7f7264ed41fa58449629e`
  — "feat(m1.4c1): complete M1.4c.1 forensic closeout certification and full 200/200 test suite".
- Parent M1.4c commit: `b79f494a4480a92f4dd4d950011f35ecdb2faab8` (subject reports 186/186 PASS).
- The M1.4c.1 closeout file itself was not part of that commit (A9).

## 3. Verified artifacts relevant to M1.4c.1 scope

| Artifact | Status |
|---|---|
| `research_agent/M1_4C_CERTIFICATION_MANIFEST.json` (v1.4.3) | present; frozen configs, dataset files, model checkpoints, classification VAL fingerprints, patient-separation contracts, firewalls |
| `research_agent/M1_4C_FINAL_FORENSIC_CERTIFICATION.md` | present; corrected by M1.4c.1 after independent forensic review |
| `research_agent/M1_4C_FINAL_PARITY_CERTIFICATION.json` | present |
| `research_agent/m0_tests/test_m14c1_forensic_closeout.py` (T177–T200) | present; full synthetic orchestration with replay |
| `research_agent/m0_tests/pristine_reference.py` | present; pristine upstream reference (commit `29245d1f...`) |
| `research_agent/PROTOCOL_AUTHORITY.md` | present; protocol authority hierarchy |

## 4. What M1.4c.1 covered (from committed test/report evidence)

- T177–T182: frozen `nih_labels.csv` SHA and Classification VAL 5-part contract
  (10,816 images, 3,854 patients, image-index/patient-sequence/label-matrix fingerprints).
- T183–T184: runtime order telemetry and CUDA determinism seeding policy.
- T185–T190: pristine reference parity (≤ 1e-6) and provenance commit.
- T191: CUDA strict determinism micro-cert (blocked by `upsample_bilinear2d_aa_backward_out_cuda`).
- T192–T193: global scientific mode gate and output freshness wording.
- T194–T197: strict raw output existence and replay integrity.
- T198–T199: scientific verdict gating and in-place report corrections.
- T200: full synthetic orchestration with replay (VALID).

## 5. Correction history (verifiable)

- `M1_4C_FINAL_FORENSIC_CERTIFICATION.md` line 4: "Corrected by M1.4c.1 after independent
  forensic review."
- M1.4c.1 commit reports **200/200** test suite PASS.

## 6. M1.4c.2 gap-fixes over M1.4c.1

M1.4c.2 (see `M1_4C2_FINAL_CLOSEOUT.md`) repaired the production replay schema mismatch
(`true_/pred_` → `<Pathology>/prob_<Pathology>`), made 14/14 replay fail-closed, added
per-pathology AUC comparison, AUC-CSV 14-row contract enforcement, production-schema
tests, and audit-hygiene fixes. Those fixes are NOT part of this retrospective record;
they are documented in the M1.4c.2 closeout.

## 7. Unverifiable / intentionally not fabricated

- Exact wall-clock timestamps of the M1.4c.1 run.
- Whether the full suite was run on real checkpoint files or synthetic stubs during M1.4c.1.
- Any M1.4c.1-specific numeric result beyond those stored in frozen JSON artifacts.

None of the above are asserted here because this is a retrospective factual record only.