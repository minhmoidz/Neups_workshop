# 01B — STEP 2A.1 Protocol Amendment

**Scope:** minimal amendment to `01_ADAPTIVE_REID_PROTOCOL.md`. No redesign, no method implementation, and no experiments. **Version:** STEP 2A.1 plus the later STEP 2B R-9/statistics clarification.

Three issues were raised. All three are accepted; **Issue 2 identifies a genuine conceptual error in my original design**, and I record that plainly below rather than presenting the change as a refinement.

---

# 1. Representative-attacker correction

## What was wrong

Original §11.3 selected the representative attacker as the valid restart whose **test AUC** was closest to the arm's median test AUC. That is test-derived model selection. It does not matter that the representative attacker was only used for secondary analyses — a model chosen by consulting the test set contaminates every quantity computed from it, and the protocol's own §15 rule 1 prohibits exactly this pattern elsewhere. I applied the rule to seed exclusion and then violated it two sections later.

## The replacement rule

> The **representative attacker** is the numerically valid completed restart whose **best validation AUC** is closest to the **median best validation AUC** across that arm's numerically valid completed runs. Ties broken by lower seed index.

**Ordering is binding and mechanically enforced:** the representative attacker is identified and written to the run provenance **before any test AUC for that arm is consulted**. Implemented as a distinct pipeline stage whose inputs cannot include test results (§17, R-8) — not as a convention.

## Downstream references updated

| **ConsumerStatus**                   |                                                                  |
| ------------------------------------ | ---------------------------------------------------------------- |
| Top-k / MRR feature extractor (§9.2) | now uses the validation-selected representative attacker         |
| Bootstrap CI (§12.2)                 | now computed for the validation-selected representative attacker |
| Qualitative figures (§11.3)          | now use the validation-selected representative attacker          |

**Not changed:** the *maximum* Re-ID AUC estimand (§12.1) is still defined on test AUC. That is a **reported quantity**, not a model selection — nothing downstream is chosen by it, so it carries no selection bias. The distinction is between measuring the max and *using* the max to pick a model; only the latter is prohibited.

---

# 2. Run-validity correction

## What was wrong — stated plainly

My original §5 classified a run as `FAILED_OPTIMIZATION` when best validation loss ≥ 0.680 **and** best validation AUC ≤ 0.55, and **excluded such runs from the privacy estimand**.

That was conceptually unsafe, and the objection is correct. A successful anonymizer *should* be able to hold a correctly executed adaptive attacker at chance. My rule would have classified exactly that outcome as an optimizer failure and deleted it from the estimand — systematically discarding the strongest possible evidence for the project's own hypothesis, and biasing every arm's reported Re-ID AUC **upward** (making methods look worse, but for an invalid reason, and asymmetrically across arms with different collapse rates).

My original §5.2 did acknowledge this risk in a "residual risk" paragraph and routed it to §14.3. That was not sufficient. Acknowledging that a rule may delete real results is not a substitute for not deleting them.

## The corrected design: two orthogonal concepts

**Training health** (did the computation execute?) and **attack effectiveness** (how well did the attacker do?) are now separate, and only the former can exclude a run.

### NUMERICALLY\_INVALID — the only excludable state

Objective execution failures only:

- crash, kill, or OOM;
- NaN / Inf in training or validation loss;
- missing or corrupted checkpoint (fails to load);
- optimizer did not execute — **weights bit-identical to initialization**;
- terminated before 2 epochs due to infrastructure failure;
- did not terminate via either legitimate path (early stopping or epoch cap).

Every criterion is a statement about whether the computation ran. **None can be satisfied by a correctly executed attack that happens to be weak.**

### VALID

Any completed, finite run that is not NUMERICALLY\_INVALID. **Enters the privacy estimand regardless of performance.**

### VALID\_NEAR\_CHANCE

A **subset of VALID** — a flag, not an exclusion class. Applied when best validation loss ≥ 0.6800 **and** best validation AUC ≤ 0.55. The same two conditions as the old gate, with the consequence inverted: they now **label** a run instead of removing it. Flagged runs are counted, reported, and **included in every estimand**.

## Interpretation — the ambiguity is preserved, not resolved

A high near-chance fraction means **one of**: (1) the anonymizer genuinely removes identity, or (2) the anonymized inputs create a pathological optimization landscape. The protocol does not silently pick one.

Measured context that constrains but does not settle it: on **unprotected** images all 10 archived attacker seeds trained successfully (test AUC 0.742–0.836, min 0.742), so near-chance behavior is induced by the anonymized inputs rather than intrinsic to the architecture. That makes the phenomenon real and arm-dependent — it does not discriminate (1) from (2).

**Predeclared escalation (§14.3)** when ≥ 5 of 10 numerically valid runs are near-chance in an arm entering a paper claim:

- **Attacker B is upgraded from RECOMMENDED to REQUIRED for that arm.** If a different backbone also sits at chance, (1) gains support; if B trains where A did not, (2) is favored and the A result cannot stand alone.
- **Extended-budget attack**: attacker A re-run at 3× epoch cap and patience 15, on 3 restarts. If the extended budget recovers above-chance performance, the original result was budget-limited, not private.

Declared now, so triggering is mechanical rather than a reaction to an inconvenient result.

---

# 3. Revised restart policy

**Target: 10 numerically valid *****completed***** runs** under the fixed attacker recipe — not 10 "successful" attacks. The budget is fixed by the threat model, not by outcomes.

| **ConfirmatoryScreening**  |                                |                               |
| -------------------------- | ------------------------------ | ----------------------------- |
| Target                     | 10 numerically valid completed | 3 numerically valid completed |
| Initial seeds              | 0–9                            | 0–2                           |
| Replacement trigger        | **NUMERICALLY\_INVALID only**  | **NUMERICALLY\_INVALID only** |
| Max infrastructure retries | **5** (≤15 attempts total)     | 2 (≤5 attempts)               |
| Seed selection             | strict ascending, predeclared  | strict ascending, predeclared |

**A completed near-chance run is never replaced.** Retraining a low-AUC run and keeping the retry would be selection on attack outcome — the same error as Issue 2, arriving through the restart policy instead of the validity gate.

The retry budget now exists **solely for infrastructure failures**. Consequently, failing to reach 10 numerically valid runs in 15 attempts is an infrastructure diagnosis (§14.4), completely distinct from an arm where attackers complete but stay at chance (§14.3). Conflating these two is prohibited.

---

# 4. Revised privacy estimands

Computed over the **fixed attack budget** — all 10 numerically valid completed restarts, near-chance runs included — never conditioned on attack success. Reported for every arm:

1. **Mean Re-ID AUC ± SD** (n stated) — typical attack outcome. Headline.
2. **Median Re-ID AUC** — robust to the skew near-chance runs introduce.
3. **Maximum Re-ID AUC under the fixed restart budget** — strongest attacker found. Privacy is a worst-case property.
4. **Near-chance fraction = ****`VALID_NEAR_CHANCE / attempted restarts`** — attack-trainability diagnostic and possible strong-privacy signal.

Reporting only the mean is prohibited. Reporting 1–3 without 4 is prohibited.

**Binding interpretation rule:** a high near-chance fraction is read as neither guaranteed privacy nor guaranteed optimizer collapse; it triggers §14.3.

---

# 5. AUC/BCE correction

**Removed:** the table mapping validation AUC values to "implied validation loss" (0.55→0.6893, 0.60→0.6777, …). The objection is correct — AUC measures ranking only, BCE depends on ranking **and** calibration, so no general deterministic mapping exists. Two models with identical AUC can have very different BCE.

**Retained, because it is exact:** for a perfectly balanced binary dataset, a constant probability-0.5 predictor has **BCE = ln 2 ≈ 0.6931**. The pair sets are verified exactly 50/50 balanced in all three splits, so this identity holds here.

**The protocol does not depend on any AUC↔BCE relationship.** `ln 2` now serves only as a descriptive reference point for the near-chance *flag*, and the flag requires **both** a loss condition and an independently measured validation-AUC condition — it never infers one from the other. Since the flag no longer excludes anything, nothing load-bearing rests on it at all.

---

# 6. Statistics clarification

The original amendment briefly proposed a patient-clustered bootstrap. Subsequent STEP 2B scientific review showed that the verification dataset is **dyadic**:

- every positive pair is same-patient;
- every negative pair spans two distinct patient identities.

A negative pair therefore has no unique membership in a one-way patient cluster. A naive patient-resampling bootstrap would duplicate or ambiguously assign negative-pair observations and must not be described as patient-level uncertainty.

## Final ruling

The patient-clustered bootstrap proposal is **withdrawn**.

Primary uncertainty for privacy claims is the distribution over independently trained attacker restarts:

- mean Re-ID AUC;
- **sample SD (`ddof=1`)**;
- unpaired Welch comparison between arms.

The existing pair-level bootstrap may remain only as a secondary diagnostic for the validation-selected representative attacker. It must be labeled:

> **PAIR-SAMPLING DIAGNOSTIC — NOT PATIENT-LEVEL UNCERTAINTY**

The STEP 2B scientific review bounded plausible pair-clustering inflation and found it remains smaller than restart-to-restart uncertainty under the current test-pair structure. Therefore the workshop protocol does not need a new dyadic/multiway bootstrap estimator.

No hierarchical uncertainty model is required.

# 7. Final frozen protocol summary

## Preserved unchanged

The following remain frozen:

- threat model and oracle separation;
- canonical ResNet-50 adaptive attacker A;
- DenseNet-121 attacker B as recommended, and required under the near-chance escalation trigger;
- frozen train/validation/test pair files;
- lowest-validation-loss attacker checkpoint selection;
- no paired-seed statistical inference;
- validation-only method screening;
- 3 numerically valid attacker restarts for screening;
- 10 numerically valid attacker restarts for confirmation;
- Top-1/Top-5/MRR as secondary metrics;
- no proxy/frozen-attacker privacy claims;
- no mixing legacy and corrected operator results;
- primary comparison against the corrected reproduced baseline;
- ~0.577 as an external aspirational target, not the primary controlled comparison.

## Changes introduced by STEP 2A.1

| Area | Final rule |
|---|---|
| Run health | `NUMERICALLY_INVALID` is the only excludable state |
| Near-chance attacks | `VALID_NEAR_CHANCE` is a flag; the run remains in every estimand |
| Replacement seeds | only for objective execution failures |
| Representative attacker | selected from validation AUC, before test evaluation |
| Privacy estimands | mean ± sample SD, median, maximum, near-chance fraction |
| AUC/BCE | no general mapping; only constant-0.5 BCE = ln 2 is exact |
| Escalation | ≥5/10 near-chance triggers attacker B + extended-budget attacker A |

## Later STEP 2B scientific clarification

The attempted patient-cluster bootstrap is withdrawn because the verification data are dyadic and negative pairs span two identities. Final uncertainty policy:

- **primary:** attacker-restart distribution;
- restart SD uses **`ddof=1`**;
- **secondary only:** pair-level bootstrap labeled `PAIR-SAMPLING DIAGNOSTIC — NOT PATIENT-LEVEL UNCERTAINTY`.

## Freeze conditions

1. No test-derived model selection.
2. No completed near-chance attacker excluded because of performance.
3. No replacement of a completed near-chance run.
4. No fabricated/stub metric may enter a scientific arm summary.
5. Every arm records the exact protocol-document hashes in provenance.
6. Every method comparison uses the corrected operator and the same frozen adaptive-attacker protocol.

**STEP 2A FINAL PROTOCOL: PASS**
