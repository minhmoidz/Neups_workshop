# P0.3 PREREGISTRATION — I_M2 DIAGNOSTIC HYPOTHESES (LOCKED BEFORE UNBLINDING)

**Date/time locked:** 2026-08-23, while `phase_c_driver.py` (PID recorded in `/tmp/p0_phaseC.log`) was still executing the FIRST of ten I_M2 trajectories. Zero I_M2 AUC values had been observed at locking time. Evidence: `grep -c 'RUN DONE' /tmp/p0_phaseC.log` == 0 for I_M2 entries at commit time of this file.

**Purpose:** eliminate post-hoc interpretation freedom for the I_M2 diagnostic anchor (`networks/pretrained_generator_prichexy_net.pth`, SHA `10122689…`), and fix the confirmatory-data partition for the P0 bridge.

---

## 1. REFERENCE BANDS (from already-unblinded screen, n=5 each)

```text
U_PUBLISHED : mean 0.6336  SD 0.0551   (screen seeds 42–46)
D_BDEV      : mean 0.8275  SD 0.0051   (screen seeds 42–46)
SEOI margin reused as band half-width: 0.03 (PROVISIONAL→treated as locked here)
```

## 2. PRE-REGISTERED CLASSIFICATION RULE (mutually exclusive)

Let `M = mean raw ROC AUC over the 10 I_M2 runs (seeds 42–51)`, computed ONLY
from sealed manifests via `im2_summary.json`.

| Rule | Classification |
|---|---|
| \|M − 0.6336\| ≤ 0.03 | **H-FINETUNE supported** — I_M2 is privacy-equivalent to U_PUBLISHED; the M2 fine-tuning path is the prime suspect for the +0.19 leakage shift |
| \|M − 0.8275\| ≤ 0.03 | **H-SIBLING supported** — the upstream release contains two generator states with drastically different privacy; D_BDEV faithfully inherited its weak init; no fine-tuning blame |
| otherwise | **H-MIXED** — both lineage difference and fine-tuning contribute; decomposition study required |

Tie-breaker if both distances ≤ 0.03 (impossible given 0.19 separation, stated
for completeness): classify H-MIXED.

Secondary (exploratory, non-decisive): per-seed scatter; Welch t-test of
I_M2 vs the five U screen values on overlapping seeds 42–46; SD comparison
(attacker-variance phenomenon).

**Anti-anchoring commitments:** no additional I_M2 seeds may be added after
unblinding to move M across a boundary; no seed replacement; no metric
transformation; classification is made exactly once from `im2_summary.json`.

## 3. CONFIRMATORY PARTITION FOR THE P0 BRIDGE (locked)

```text
PRIMARY confirmatory analysis:  paired attacker seeds 47–67 (21 pairs)
                                — never inspected prior to this document.
SECONDARY (descriptive):        all 26 pairs (42–67), disclosing that
                                seeds 42–46 informed the screen stage.
Decision rules: canonical P0 table (TOST α=0.05 / directional bounds,
δ=0.03) applied to the PRIMARY set; secondary reported with disclosure.
```

## 4. KNOWN PROVENANCE FLAW ACKNOWLEDGED NOW

Manifest field `runner_commit` currently records the artifact-repository HEAD
(`c2ee268…`) while executing code lives in review branch
`review/p0-runner-attacker-loop-20260823` (`e15b168+`). Planned fix after
phase C completes: schema bump adding separate `code_commit` +
`artifact_repo_commit`; existing runs annotated as exceptions. Recorded here
so the flaw cannot be silently forgotten or retro-fixed unnoticed.

## 5. CONSEQUENCE MAP (pre-registered next actions)

| Classification | Immediate action |
|---|---|
| H-SIBLING | Declare two-release hazard finding; adopt C1 (fine-tune-from-U) as primary method candidate; utility check vs published numbers |
| H-FINETUNE | Forensic pass over anonymizer training loop (selection criterion, stale fakes, AC-critic co-adaptation) before any method candidate |
| H-MIXED | Both tracks, mechanism-first |

Method-candidate target (any branch): reduce adaptive-attacker AUC below the
U_PUBLISHED reference band (≤ 0.60) at preserved classification utility,
measured under THIS locked harness.

*End of preregistration. This file must not be edited after unblinding;
corrections go into a dated addendum.*
