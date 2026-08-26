# V2 CONFIRMATION PREREGISTRATION — NESTED-SPLIT ADAPTIVE-ATTACKER CONFIRMATION (LOCKED BEFORE UNBLINDING)

**Date/time locked:** 2026-08-26, evening. Status of outcome variables at lock time:

- V2 attention run1 training is IN PROGRESS at epoch ~21/250
  (`research_runs/V2_QUEUE/train.log`, status.txt == `RUNNING`). Zero V2
  privacy AUC values have been observed. Evidence: `grep -c COMPLETE
  research_runs/V2_QUEUE/status.txt` == 0; train.log contains no evaluation
  summary lines.
- The nested 5-role pair files described below DO NOT EXIST yet.
- Batch-ablation arms (`v2ctrl_b16acc1`, `v2ctrl_b16acc2`) have not started.

This file must not be edited after unblinding; corrections go into a dated
addendum.

---

## 0. PURPOSE

Provide the first **confirmatory** (development-firewall-respecting) test of
the project's method candidate against the canonical anchor, under a split in
which generator selection, attacker selection, and privacy evaluation use
disjoint pair pools — closing the role-reuse hazard documented in
`G0_1B_MINIMAL_ERRATA_2026-08-21.md` and `P0_P1_PREEXPERIMENT_PROTOCOL_REVIEW_2026-08-21.md` §1 item 7.

## 1. ANCHOR AND CANDIDATE (selection rule locked now)

- **Anchor U = U_PUBLISHED**, the released upstream generator used by the P0
  screen (mean screen AUC 0.6336 ± 0.0551). All comparisons are paired,
  within-harness, on identical attacker seeds and identical new splits —
  absolute cross-split band transfer is NOT assumed.
- **Candidate selection rule (declared before any candidate privacy AUC is
  known):** once ALL of {run1 attention, arm acc1, arm acc2} pipelines reach
  COMPLETE, the confirmation candidate is the checkpoint with the LOWEST
  development-stage privacy VAL AUC (`generator_lowest_total_loss.pth`,
  method-neutral selection metric per M2_S1_EXECUTION_LOCK.json), evaluated by
  the governed evaluator on fold=val. If Direction C has also reached COMPLETE
  by decision time, it enters the same argmin. Tie-break: earlier epoch.
  Exactly ONE candidate enters confirmation. No substitution after unblinding.

## 2. NEW SPLIT (nested, patient-disjoint roles)

Generated from `Data_Entry_2017_v2020.csv` patient IDs; hard requirement:
pairwise patient overlap across all pools == 0 (audited, documented like
`CHESTXRAY14_DATA_AUDIT.md`).

```text
P_gen_train   : generator training pool            (new)
P_gen_select  : generator selection pool           (new)
P_att_train   : attacker training pool             (new)
P_att_select  : attacker inner early-stop pool     (new)
P_confirm     : privacy confirmation scoring pool  (new, disjoint from ALL above)
```

The legacy TRAIN/VAL pair files remain for development runs already in
flight (run1, ablation); they are NOT part of this confirmation protocol.

## 3. HARNESS

Reuse P0 bridge components verbatim: `seed_contract.py` (domain-separated
`P0_SEED_V1`), `attacker_loop.py` lifecycle, sealed `run_manifest.json`,
fail-closed gate list of `protocol_v1.json`. Only `.pair_files` change to the
§2 pools. Attacker: Siamese ResNet-50, Adam lr 1e-4, batch 32, max 100 epochs,
patience 5; train geometry anon/anon; scoring geometry anon(x1)/real(x2) on
P_confirm only. Raw ROC AUC, orientation fixed a priori; post-hoc flipping and
"effective AUC" forbidden (protocol §5.2 items 16–18).

## 4. SEEDS AND PARTITION

- Exactly 10 attacker seeds per arm: derived seeds for master seeds 42–51,
  domains as in `seed_contract.py`. No seed may be added, replaced, or rerun
  after unblinding.
- PRIMARY: all 10 pairs. There is no held-back secondary set because every run
  under the NEW split is fresh; disclosure duty: seed numbers 42–46 appeared in
  prior harnesses (screen), acknowledged here.
- Unit-of-analysis caveat declared now: the generator seed is single (42);
  attacker seeds are nested replicates. Any claim is scoped to "this generator
  state"; generator-seed variance remains future work (27–59 GPU-h/seed).

## 5. PRE-REGISTERED DECISION RULES (mutually exclusive)

Let Δ = mean_seed[AUC(candidate)] − mean_seed[AUC(U_PUBLISHED)], paired by
seed; CI = one-sided 95% upper bound of paired differences (bootstrap over
seeds, 10000 resamples, fixed seed 12345).

| Rule | Classification |
|---|---|
| Δ ≤ −0.03 AND CI_upper < 0 AND utility gate PASS | **H-SUPERIOR** — publishable privacy-improvement claim under THIS harness |
| \|Δ\| < 0.03 (or CI spans it) AND utility gate PASS | **H-EQUIVALENT** |
| Δ > +0.03 OR utility gate FAIL | **H-NOT-SUPPORTED** |

δ = 0.03 AUC (SEOI margin reused; frozen since §5.4 of the P0 review).
Utility gate: macro-AUC (14 pathologies, governed evaluator, fold=val)
candidate ≥ U_PUBLISHED reference 0.7730 − 0.00 (zero-degradation, no
tolerance). Utility is measured ONCE, after privacy unblinding, by a fixed
command recorded in the addendum trigger log.

Secondary (exploratory, non-decisive): per-seed scatter; D_BDEV re-measured on
the new split for context; convergence curves.

## 6. CONSEQUENCE MAP

| Classification | Immediate action |
|---|---|
| H-SUPERIOR | Freeze candidate + manifests; manuscript track A (method paper): V2 architecture + adaptive-attacker protocol + confirmation table |
| H-EQUIVALENT | Manuscript track B (audit/methodology paper): P0 screen (Δ=−0.110, 21/21), I_M2 H_SIBLING, Stage-A pilot n=26, batch ablation, nested-split framework as the contribution; candidate reported as null result |
| H-NOT-SUPPORTED | Same track B; candidate analysis restricted to loss-curve forensics appendix |

## 7. KNOWN LIMITATIONS ACKNOWLEDGED NOW

- Development-stage metrics (all existing numbers) come from the shared-VAL
  harness; they are reported in the paper as development results only.
- Patient-level bootstrap CIs remain UNVALIDATED (P0 review §9); seed-level
  intervals only.
- Segmentation utility: BLOCKED, excluded from all claims.
- Manifest `runner_commit` provenance flaw (P0.3 prereg §4) applies here too;
  schema bump scheduled with this experiment's manifests.

## 8. ANTI-ANCHORING COMMITMENTS

No additional seeds; no metric transformation; no threshold edits after seeing
results; classification made exactly once from the sealed summary JSON; this
file immutable after unblinding (addenda only, dated).

*End of preregistration.*
