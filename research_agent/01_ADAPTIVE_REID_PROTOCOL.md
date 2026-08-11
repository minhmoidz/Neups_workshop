# 01 — Canonical Adaptive Re-Identification Protocol (STEP 2A)

**Status:** protocol design only. No code, no training, no baseline runs. **Version:** STEP 2A + STEP 2A.1 amendment + post-STEP-2B R-9/statistics clarification. **Applies to:** the corrected PriCheXy-Net baseline, all future proposed methods, and every final privacy claim in the paper. **Basis:** pipeline audit at `db2d1fe` plus measurement of 100 archived attacker runs across 10 legacy-operator arms. Every threshold below is traced to a measurement or to a mathematical constant; none is chosen to make existing results look better.

> **Operator scope.** All archived numbers quoted in this document were produced under the **legacy** operator. They are used here *only* to characterize attacker training dynamics and variance — properties of the optimizer and data pipeline, not of the anonymization. They are never used as privacy baselines. The corrected baseline has not been measured (STEP 2B).

---

# 1. Threat model

## 1.1 Adversary goal

Given anonymized chest radiographs, decide whether two images belong to the same patient (verification), and secondarily, match an anonymized probe against a gallery of identified scans (identification, §9).

## 1.2 Attacker knowledge — canonical (used for all headline claims)

| **Attacker hasY/NRationale**                                |         |                                                                                                                                                |
| ----------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Knowledge that images are anonymized                        | **YES** | Kerckhoffs. Security through obscurity is not a privacy claim.                                                                                 |
| The anonymization algorithm (published method)              | **YES** | Same.                                                                                                                                          |
| Generator architecture                                      | **YES** | Follows from the published algorithm.                                                                                                          |
| **Generator checkpoint (weights)**                          | **NO**  | The trained checkpoint is the deployed secret; a hospital does not publish it.                                                                 |
| Training *distribution* (NIH ChestX-ray14 is public)        | **YES** | Public dataset; assuming otherwise is indefensible.                                                                                            |
| **Anonymized training images from the deployed anonymizer** | **YES** | This is what makes the attacker *adaptive* — see §1.4.                                                                                         |
| **Patient ID labels for the attacker's own training pairs** | **YES** | Worst-case but realistic: an insider or a linked historical release. Assuming the attacker has no labels would make the attack trivially weak. |
| **Deformation grids / displacement fields**                 | **NO**  | Not available in deployment. Oracle only (§1.5).                                                                                               |
| **Aligned (original, anonymized) pairs of the same image**  | **NO**  | Not available in deployment. Oracle only (§1.5).                                                                                               |

The canonical attacker therefore sees **anonymized images plus identity labels for its own training/validation split, and nothing else**. It never sees the generator weights, the grids, or any original/anonymized correspondence.

## 1.3 What the attacker never gets, in any configuration

- Any image or patient from the **test** split during training or model selection (§3).
- Test-set labels, test-set AUC, or any test-derived quantity used to select a checkpoint, a seed, or a hyperparameter (§7, §15).

## 1.4 "Adaptive" — exact definition

An attacker is **adaptive** iff, for the specific anonymizer under evaluation, it is:

1. **freshly initialized** (no weights carried from another arm, and no weights carried from a raw-image attacker), and
2. **trained end-to-end on images produced by that exact anonymizer** — same checkpoint, same μ, same `transform_mode`, same stochastic settings — for its full training and validation phases.

A frozen attacker, an attacker trained on a different arm, a raw-image attacker evaluated on anonymized inputs, or any frozen-encoder proxy similarity **may never back a final privacy claim**, in either direction. These are diagnostics only. (This restates the standing project rule and is the reason the STEP 1C smoke proxy AUC was declared non-evidence.)

## 1.5 Oracle diagnostics — explicitly separated

The following are scientifically interesting and may appear in the paper, but **only** in an analysis/mechanism section, never as the privacy estimand:

- Known-grid inversion (attacker given the displacement field). My `00B` §5 result shows a regularized truncated-SVD inversion recovers the image to visually perfect quality at the operating budget. This bounds what the *transformation* protects; it is not the deployment threat.
- Attackers given original/anonymized pairs.

Any figure or table containing an oracle number must be labeled `ORACLE — not a deployment threat model` in the caption.

---

# 2. Canonical attacker

## 2.1 Attacker A — canonical (REQUIRED)

The existing Siamese verification network, unchanged from the repository:

| **ComponentValueSource** |                                                |                                                        |
| ------------------------ | ---------------------------------------------- | ------------------------------------------------------ |
| Backbone                 | ResNet-50, ImageNet-pretrained, shared weights | `networks/`, current repo                              |
| Head                     | Siamese pair head, scalar logit                | current repo                                           |
| Loss                     | `nn.BCEWithLogitsLoss` on balanced pairs       | `agents/AgentSiameseNetwork.py:97`                     |
| Optimizer                | Adam, lr = 1e-4                                | `config_retrainSNN.json`                               |
| Batch size               | 16                                             | `config_retrainSNN.json`                               |
| Max epochs               | 100                                            | `config_retrainSNN.json`                               |
| Early stopping           | patience 5 on validation loss                  | `config_retrainSNN.json`, `AgentSiameseNetwork.py:151` |
| Image size               | 256                                            | `config_retrainSNN.json`                               |

**Attacker A is frozen for the entire project.** Its architecture, loss, optimizer, learning rate, batch size, epoch budget, and early-stopping patience are **identical for every anonymizer arm including the baseline**. Any change to A invalidates every previously measured arm and requires re-running all of them (§15).

**Why not tune the attacker per method?** Because a stronger attacker on one arm and a weaker one on another is precisely the cherry-pick the protocol exists to prevent. The attacker budget is a fixed property of the threat model, not a free parameter.

## 2.2 Attacker B — unseen backbone

**Status: RECOMMENDED, not required.**

- **Definition:** the same Siamese training recipe with a **DenseNet-121** backbone (ImageNet-pretrained), everything else identical to A.
- **Why DenseNet-121:** it is the CheXNet backbone already vendored in this repository (`chexnet/`), so it adds no new dependency, and it is architecturally distinct from ResNet-50 (dense connectivity vs residual), which is what makes it a meaningful generalization test.
- **Purpose:** to test whether a proposed method overfits to the specific attacker it was trained against. A method that lowers Re-ID against A but not against B has learned to defeat A, not to remove identity.
- **When run:** only for the **final** proposed method and the corrected baseline, after all development is complete. Not during screening.
- **Declared now, before any results exist** — this is the point. Selecting a backbone after seeing which one flatters the method is prohibited (§15).
- **Reporting rule:** if B is run, its result is reported **whatever it shows**. Running B and omitting it is a cherry-pick.

Why RECOMMENDED rather than REQUIRED: for a workshop paper, one canonical adaptive attacker is a defensible privacy claim; B strengthens it materially but its absence is not a methodological error, and the GPU budget (§17) may not permit 10 restarts × 2 backbones × all arms.

**No attack zoo.** A and B are the complete set.

---

# 3. Data and pair protocol

## 3.1 Verified properties (re-checked at `db2d1fe`)

| **Splitpairspatientsimagespositive fraction** |        |       |        |       |
| --------------------------------------------- | ------ | ----- | ------ | ----- |
| train                                         | 10 000 | 9 053 | 18 089 | 0.500 |
| validation                                    | 2 000  | 1 742 | 3 484  | 0.500 |
| test                                          | 5 000  | 1 307 | 8 062  | 0.500 |

- **Patient-disjoint:** train ∩ val = 0 patients, train ∩ test = 0, val ∩ test = 0.
- **Image-disjoint:** 0 shared images across all three pairs of splits.
- **Balance:** exactly 0.500 positive in every split — this is load-bearing for the validity threshold in §5.
- **Official NIH fold adherence:** all train and validation images come from `train_val_list.txt` (18 089 and 3 484, zero from `test_list.txt`); all test images come from `test_list.txt` (8 062, zero from `train_val_list.txt`).

**These three pair files are FROZEN**: `image_pairs_training_10000.txt`, `image_pairs_validation_2000.txt`, `image_pairs_testing_5000.txt`. Identical pairs, identical identity labels, identical order, for **every method and every restart**. Regenerating them for any reason invalidates all prior arms.

## 3.2 Generator training data

The anonymization generator is trained on the train split only. Because the generator's training images and the attacker's test images are patient-disjoint by the above, there is no generator→test adaptation leakage.

## 3.3 Anonymized image generation: dynamic, not cached

Anonymized images are generated **on the fly** inside the data loader, as the current pipeline does. Rationale: the generator is deterministic given (checkpoint, μ, `transform_mode`, `stochastic_lambda=0`), so caching buys only speed, and a stale cache after an operator or checkpoint change is a serious and silent correctness hazard — exactly the class of bug STEP 1 was about.

**Mandatory determinism check** (cheap, run once per arm): generate the anonymized test set twice and assert bit-identical output. If it is not bit-identical, the arm is stochastic and §3.4 applies.

## 3.4 Stochastic methods — principles fixed now

Not solved in full (not needed yet), but the metric definition is fixed now so it cannot be changed later to suit a result:

1. **The privacy estimand is defined over the deployed distribution**, not over one lucky draw. If a method is stochastic at inference, each test image is anonymized with **one** draw, and that draw is fixed by a declared evaluation seed recorded in the run provenance.
2. **The attacker sees the same stochasticity in training** that it faces at test — otherwise the attacker is artificially handicapped and the privacy number is inflated.
3. **Transformation randomness is a third variance source** and must be reported separately from attacker-restart variance (§12), by re-drawing the transformation over ≥5 evaluation seeds for the final method only.
4. Reporting the best transformation draw is prohibited (§15).

---

# 4. Attacker training

Fixed recipe (§2.1) applied identically to every arm.

1. Fresh random initialization from ImageNet weights; new attacker seed per restart. The seed controls initialization of the head, data-loader shuffling, and any augmentation.
2. Train on **anonymized** training pairs from the arm's anonymizer.
3. Validate each epoch on **anonymized** validation pairs from the same anonymizer.
4. Early stopping: patience 5 on validation loss.
5. Checkpoint selection: §7.
6. Evaluate once on the frozen test pairs. **One test evaluation per run, at the end.** Test performance is never consulted during training, checkpointing, or seed handling.

## 4.1 Required logging — currently missing, must be added

The pipeline audit found that the training loop records **only** `loss_dict['training']` and `loss_dict['validation']` (`AgentSiameseNetwork.py:134–135`), and that `validate_snn` returns `validation_loss` alone (`utils/utils.py:820`) — it never collects `y_true`/`y_pred`, so **no validation AUC or validation accuracy exists anywhere in the current pipeline**. Archived runs retain only final test summaries; no loss curves were saved.

The validity criterion in §5 therefore requires new logging. This is the one implementation dependency of this protocol (§17). Per run, the following must be persisted to a machine-readable file:

- per-epoch training loss and validation loss (already computed, just not saved per-run),
- per-epoch **validation AUC** and validation accuracy (requires collecting `y_true`/`y_pred` in `validate_snn`),
- best validation loss and the epoch at which it occurred,
- epochs completed, and whether early stopping or the epoch cap terminated the run,
- a NaN/Inf flag for training and validation loss,
- the resolved `transform_mode`, generator checkpoint hash, μ, and attacker seed.

---

# 5. Attacker-validity criterion

**This is the core of STEP 2A.** The criterion is predeclared, computed only from training and validation statistics, and never from test AUC.

## 5.0 Governing principle (amended, STEP 2A.1)

**Training health and attack effectiveness are separate concepts and are never conflated.**

A genuinely successful anonymizer should be capable of holding a correctly executed adaptive attacker at chance. Therefore chance-level attacker performance is **a possible privacy result**, not evidence of optimizer failure. Only **objective execution failures** may be excluded from the privacy estimand. Everything else — including a completed run that never rose above chance — is retained, reported, and counted.

This supersedes the original `FAILED_OPTIMIZATION` state, which excluded near-chance runs and could therefore have discarded exactly the outcome the project is trying to detect.

## 5.1 The anchor: chance-level loss is a mathematical constant

The attacker loss is `BCEWithLogitsLoss` on a pair set verified to be **exactly 50/50 balanced** (§3.1). A predictor that outputs constant probability 0.5 (equivalently, constant logit 0) achieves:

**BCE = ln 2 ≈ 0.6931.**

This is the only exact claim the protocol makes about the loss scale.

**No general AUC↔BCE mapping is claimed or used.** AUC measures ranking only; BCE depends on ranking **and** probability/logit calibration, so two models with identical AUC can have very different BCE. `ln 2` is used solely as a descriptive reference for the near-chance flag in §5.3 — never as an exclusion criterion.

## 5.2 Run states (amended)

A run is assigned exactly one execution-health state. Only the first is excluded from privacy estimands.

### NUMERICALLY_INVALID — the only excludable state

An **objective execution failure**. Any of:

- the process crashed, was killed, or ran out of memory;
- NaN or Inf appeared in training or validation loss at any epoch;
- the checkpoint file is missing or fails to load;
- the optimizer did not execute — model weights remain bit-identical to initialization after training;
- the run terminated before completing at least 2 epochs because of infrastructure failure;
- the run did not terminate through either legitimate path: early stopping or reaching the epoch cap.

Every condition above is a statement about **whether the computation executed**, not how well the attacker performed. None can be satisfied merely because a correctly executed attacker happens to be weak.

### VALID

A completed, finite run that is not `NUMERICALLY_INVALID`.

**Every VALID run enters the privacy estimand regardless of performance.**

### VALID_NEAR_CHANCE

A **diagnostic flag on a VALID run**, not an exclusion class. A VALID run is additionally flagged `VALID_NEAR_CHANCE` when both:

- best validation loss ≥ **0.6800**, and
- best validation AUC ≤ **0.55**.

The flag is computed on validation data only and never touches the final test quantity.

**Flagged runs remain VALID and are included in every privacy estimand.**

## 5.3 How the near-chance flag is interpreted

The flag is deliberately ambiguous. A high near-chance fraction can mean:

1. the anonymizer genuinely removes identity information, or
2. the anonymized input distribution creates a pathological optimization landscape for the current attacker.

The protocol does not silently choose either explanation.

Measured historical context: on unprotected/raw images, all 10 archived attacker seeds trained successfully and produced well-above-chance test AUCs. Near-chance behavior therefore appears arm-dependent rather than intrinsic to the architecture, but this observation does **not** distinguish genuine privacy from attack-trainability failure.

**Reporting rule:** `near-chance count / attempted restarts` is reported for every arm as a first-class diagnostic alongside the privacy estimands (§12.1). If most attackers in a paper arm remain near chance, the predeclared escalation in §14.3 is triggered.

### Signals and their roles

| Signal | Role |
|---|---|
| NaN / Inf / crash / OOM | **EXCLUSION** — objective execution failure |
| Weights bit-identical to initialization | **EXCLUSION** — optimizer did not execute |
| Checkpoint loadability | **EXCLUSION** — corrupted/missing checkpoint |
| Termination path | **EXCLUSION** only for infrastructure-invalid termination |
| Validation loss | **FLAG only** — near-chance diagnostic |
| Validation ROC-AUC | **FLAG only** — near-chance diagnostic |
| Training loss | Logged; diagnostic only |
| Verification accuracy | Logged; diagnostic only |
| Embedding separation | Optional diagnostic; never a gate |
| Best epoch / early stopping | Logged; diagnostic only |

**No performance-based signal excludes a completed run.**

# 6. Restart policy

**Unit of replication: one independently initialized attacker training run.** Seed IDs are reproducibility identifiers, not matched blocks across methods (§8).

## 6.1 Confirmatory arms (all paper claims) — amended

The attack budget is fixed in advance.

- **Target: 10 numerically valid completed restarts.**
- Initial seeds: **0–9**.
- Replacement seeds are used **only** for `NUMERICALLY_INVALID` execution failures.
- A completed `VALID_NEAR_CHANCE` run is **never replaced**.
- Replacement seeds proceed in strict ascending order: 10, 11, 12, ...
- Maximum infrastructure retry budget: **5 replacement attempts**, i.e. at most 15 attempted runs per arm.
- Every attempted run is retained in provenance with its execution state and near-chance flag.
- Report per arm: attempted, numerically invalid, valid, and near-chance counts.

If 10 numerically valid completed runs cannot be obtained within 15 attempts, the arm is treated as an infrastructure/numerical problem (§14.4), not as a privacy result.

## 6.2 Screening arms (development only)

- **Target: 3 numerically valid completed restarts.**
- Initial seeds: **0–2**.
- Maximum 2 infrastructure replacements, i.e. at most 5 attempts.
- Replacement again occurs **only** for `NUMERICALLY_INVALID`.
- Screening is exploratory and may not support final paper claims (§10).

## 6.3 Power justification

Minimum detectable ΔAUC at 80% power, α = 0.05, two-sided Welch, based on the observed archived per-arm SD spread:

| Numerically valid restarts / arm | MDD at median SD | MDD at worst-case SD |
|---|---:|---:|
| 3 | 0.118 | 0.199 |
| 5 | 0.091 | 0.154 |
| **10** | **0.064** | **0.109** |
| 15 | 0.053 | 0.089 |
| 20 | 0.046 | 0.077 |

The intended effect size is on the order of ~0.08 AUC. Ten restarts are therefore reasonable for confirmatory workshop-level evaluation at typical historical variance, while three restarts are suitable only for screening.

All reported restart SDs use the **sample standard deviation (`ddof=1`)**, because the observed restarts are a sample from the attacker-training randomness distribution.

# 7. Checkpoint selection

**Audit result:** the current rule is **already correct**. `AgentSiameseNetwork.py:141–143` keeps `best_net` at the **lowest validation loss**; early stopping steps on the same quantity (`:151`); the test set is touched exactly once, after training, at `:158`. **No test-derived quantity enters checkpoint selection.**

**Canonical rule (unchanged from the audited code):** select the epoch with the **lowest validation verification loss**; ties broken by the earlier epoch.

This identical rule applies to every anonymizer. **No method-specific checkpoint tuning.**

Not adopted: selecting on validation *AUC* instead of loss. Validation AUC will now be logged (§4.1) and could arguably be the better selection signal, but changing the rule would break comparability with the audited pipeline for no demonstrated gain. Selection stays on validation loss; validation AUC is used for the validity gate and diagnostics only.

---

# 8. Attacker seed handling

**Finding: seed IDs carry no shared signal across methods. Paired-seed inference is not justified.**

Across all 45 method pairs among the 10 archived arms, correlating per-seed test AUC on matched seed IDs:

- mean r = **+0.072**, median r = −0.008, range [−0.483, +0.857]
- one-sample t-test on the 45 correlations against 0: **p = 0.146** (not distinguishable from zero)
- variance ratio var(paired difference)/var(unpaired difference): median **1.01** — pairing buys essentially nothing

**Consequences, binding:**

1. **Unpaired (Welch) two-sample tests only.** Paired t-tests, matched-seed differences, and any language implying "the same attacker" across arms are prohibited.
2. **The unit of replication is one independently initialized attacker training run**, not a seed index. Seeds are bookkeeping for reproducibility, not a blocking factor.
3. Seed numbers are still recorded and reused for reproducibility (a given seed reproduces a given run within an arm), but no statistical structure is claimed from them.

---

# 9. Top-k identification protocol

## 9.1 Status

**SECONDARY evidence. Re-ID ROC-AUC from the adaptive attacker is the PRIMARY privacy estimand.**

Top-k depends on gallery size and gallery construction, so it is not directly comparable across papers. The historical `eval_top1.py` frozen-feature proxy may remain as a diagnostic, but it cannot back a final privacy claim.

## 9.2 Protocol

- **Gallery:** one clean/reference image per patient.
- **Probe/query:** one anonymized image from the same patient.
- **N = 500 patients**, frozen with selection seed **42**.
- The frozen gallery/probe metadata file is reused identically for every arm.
- One gallery entry per patient ⇒ chance Top-1 = `1/500 = 0.002`.
- Gallery and probe images must be distinct.
- Prefer **different follow-up numbers** where available to avoid near-duplicate same-study pairs.
- Canonical deployment-realistic setting: **clean gallery, anonymized probe**.
- Symmetric anonymized-gallery/anonymized-probe may be reported only as a secondary configuration.
- Metrics: **Top-1, Top-5, MRR**.
- Feature extractor: the **adaptive attacker A representative checkpoint**, selected using validation statistics only (§11.3).
- Frozen ImageNet or non-adaptive feature extractors may be reported only as clearly labeled proxies.

# 10. Screening protocol

Used during method development. **Never a source of a final paper privacy claim.**

- 3 numerically valid completed attacker restarts.
- Initial seeds 0–2; ≤2 infrastructure replacements.
- Replacement only for `NUMERICALLY_INVALID`.
- **Development and method selection use the VALIDATION split only.**
- Method variants are compared using validation Re-ID evidence and utility metrics.
- The official test split is not repeatedly evaluated during screening.
- Screening results are internally labeled `SCREENING (n=3, exploratory)`.

The purpose is to reject obviously poor variants cheaply while preserving the final test split for confirmatory evaluation.

# 11. Final confirmation protocol

- Triggered **once per frozen arm** after development for that arm is complete.
- 10 numerically valid completed attacker restarts under §6.1; near-chance runs remain included.
- Canonical attacker recipe (§2.1/§4), checkpoint rule (§7), and frozen pair files (§3.1).
- Test evaluation occurs only after training/validation stages and representative-attacker selection are frozen.
- Each completed attacker evaluates the frozen test pairs exactly once.
- If the anonymizer changes, it is a new arm with a new identifier; the previous confirmatory result is retained rather than overwritten.

## 11.3 Representative attacker (validation-only selection)

Where a single attacker is required for secondary analyses (Top-k, pair-sampling bootstrap diagnostic, qualitative figures), select:

> the numerically valid completed restart whose **best validation AUC** is closest to the **median best validation AUC** across that arm's numerically valid completed runs.

Ties are broken by lower attacker seed.

**No test-derived quantity enters this selection.**

The representative attacker identity must be written to provenance **before any test AUC for the arm is consulted**. The implementation must enforce this as a separate pipeline stage.

The strongest attacker is defined separately as the maximum test AUC under the fixed attack budget (§12.1). That maximum is a reported estimand, not a model-selection signal.

# 12. Statistical reporting

## 12.1 Primary privacy estimands — fixed attack budget

All privacy estimands are computed over the **10 numerically valid completed attacker restarts**, including near-chance runs.

Report for every arm:

1. **Mean Re-ID AUC ± sample SD (`ddof=1`)** — typical adaptive attacker outcome; headline value.
2. **Median Re-ID AUC** — robust summary under skew.
3. **Maximum Re-ID AUC under the fixed restart budget** — strongest successful recovery observed.
4. **Near-chance fraction** — `VALID_NEAR_CHANCE / attempted restarts`.

Reporting only the mean is prohibited.

**Interpretation:** a high near-chance fraction is neither automatically proof of privacy nor automatically optimizer collapse. It triggers the escalation procedure in §14.3 when it is large enough to support a paper claim.

Higher Re-ID AUC = worse privacy.

## 12.2 Uncertainty sources

Two uncertainty axes are kept conceptually separate:

1. **Attacker-restart variability** — the primary uncertainty for between-method privacy comparison.
2. **Finite test-pair sampling variability** — secondary diagnostic uncertainty for a fixed representative attacker.

### Primary uncertainty

For every arm report:

- mean ± **sample SD (`ddof=1`)** across the 10 numerically valid attacker restarts;
- effect size and 95% CI for the between-arm difference using the restart distributions.

This is the uncertainty axis on which the paper's primary inference rests.

### Pair-level bootstrap — secondary diagnostic only

The verification test set is dyadic:

- positive pairs link two images from the same patient;
- negative pairs link two distinct patient identities.

A negative pair therefore does not have a unique one-way patient-cluster membership. The earlier proposal to perform a naive patient-cluster bootstrap is **withdrawn**.

Final policy:

- retain the existing **pair-level bootstrap** only as a secondary diagnostic for the validation-selected representative attacker;
- label it explicitly:

  `PAIR-SAMPLING DIAGNOSTIC — NOT PATIENT-LEVEL UNCERTAINTY`

- do **not** describe it as a patient-clustered CI;
- do not use it for the primary between-method significance claim.

The STEP 2B scientific review measured that plausible clustering corrections remain smaller than the restart-to-restart standard error of the mean, so the unresolved dyadic clustering does not dominate the primary inference.

No hierarchical model is required for the workshop scope.

## 12.3 Between-arm tests

- Primary comparison: **unpaired Welch two-sample t-test**, two-sided.
- Unit of replication: one independently trained attacker restart.
- Report difference in means with a 95% CI and p-value.
- Use **sample SD (`ddof=1`)**.
- Primary planned comparison: proposed method vs corrected baseline.
- Secondary arm comparisons use Holm–Bonferroni adjustment.
- If a future arm has a clearly problematic distribution, report a Mann–Whitney U sensitivity analysis in addition to, not instead of, the declared Welch analysis.

# 13. Optional unseen attacker

See §2.2. **Attacker B (DenseNet-121): RECOMMENDED.** Declared now; run only on the final method and the corrected baseline; reported whatever it shows.

---

# 14. Failure handling

## 14.1 Individual execution failures

`NUMERICALLY_INVALID` runs are excluded from the privacy estimands, but **counted and reported**. They are never silently dropped.

Near-chance runs are **not failures** and are never excluded.

## 14.2 Infrastructure instability

Replacement seeds are issued only for objective execution failures.

A high count of replacements indicates a numerical/infrastructure issue and must be reported and diagnosed. It is not privacy evidence.

## 14.3 Arm with many near-chance attackers

If **≥5 of 10 numerically valid completed attacker A restarts** are flagged near chance for an arm entering a paper claim, trigger the predeclared escalation:

1. **Attacker B (DenseNet-121) becomes REQUIRED for that arm.**
2. Run an **extended-budget attacker A diagnostic** on 3 restarts:
   - 3× epoch cap,
   - patience 15.

Interpretation:

- if attacker B and extended-budget A also remain near chance, genuine strong privacy gains support;
- if a stronger/different attacker recovers identity, the original near-chance behavior was attack-budget/backbone dependent.

Both escalation results must be reported.

## 14.4 Failure to obtain 10 numerically valid runs

If 15 attempts do not yield 10 numerically valid completed runs:

- mark the arm `INCOMPLETE (k/10 numerically valid in 15 attempts)`;
- report the available diagnostics but do not treat the arm as directly comparable to complete confirmatory arms;
- diagnose the infrastructure/numerical failure before drawing a privacy conclusion.

# 15. Anti-cherry-picking rules

Binding prohibitions. Each maps to a mechanism above.

1. **No excluding seeds because test AUC is low.** Run exclusion is decided only from objective execution-health signals; near-chance flagging uses validation statistics only (§5), before any test evaluation.
2. **No selecting the generator checkpoint using test Re-ID.** Generator selection uses generator-side training/validation criteria only.
3. **No changing attacker hyperparameters per method.** Attacker A is frozen (§2.1).
4. **No changing test pairs per method.** The three pair files are frozen (§3.1).
5. **No reporting only favorable attacker seeds.** All numerically valid completed restarts enter the estimand, including near-chance runs; all attempted runs are counted (§14.1).
6. **No increasing the retry count for selected methods.** The maximum infrastructure budget is 15 attempts for every confirmatory arm (§6.1), fixed in advance; replacements occur only for NUMERICALLY_INVALID runs.
7. **No selecting the attacker backbone after seeing which hurts the proposed method least.** A and B are declared here, before any result (§2.1, §2.2), and B is reported whatever it shows.
8. **No repeated confirmatory evaluation.** One confirmatory run per arm; a changed method is a new arm (§11).
9. **No mixing legacy- and corrected-operator results** in any comparison (carried over from the STEP 1 review).
10. **No reporting the best transformation draw** for stochastic methods (§3.4).
11. **No frozen-attacker or proxy-similarity number presented as a privacy result**, in either direction (§1.4).

Every arm's provenance record must carry: `transform_mode`, generator checkpoint hash, μ, attacker seed, pair-file hashes, and the per-run validity state. This makes each rule auditable after the fact rather than merely promised.

---

# 16. Paper success criteria

**No corrected-baseline numbers are invented here. They have not been measured.** Only the comparison logic is defined.

Let `B` = corrected PriCheXy-Net baseline under this protocol; `P` = proposed method under this protocol. Both measured with 10 numerically valid completed restarts.

## 16.1 Privacy win (primary claim)

**Required:** mean Re-ID AUC(P) < mean Re-ID AUC(B), with a two-sided Welch test at α = 0.05 and a 95% CI for the difference that **excludes zero**.

**Supporting requirement:** max Re-ID AUC(P) < max Re-ID AUC(B). A method that lowers the mean but not the maximum is reported as such and does not qualify as a clean privacy win.

**Strong external target:** mean Re-ID AUC(P) < \~0.577 (the original paper's figure). Treated as an **external aspiration only**. The original number was produced under a different operator (legacy, with the border defect quantified in STEP 1) and a protocol whose validity criterion and restart policy differ from this one. **The scientifically valid primary comparison is against the corrected reproduced baseline ****`B`**, measured under this protocol. Crossing 0.577 is reported if achieved; failing to cross it while beating `B` is still a valid result.

**Minimum meaningful effect: TO BE LOCKED AFTER BASELINE ESTIMATION.** Given the archived SD spread, a 0.08 difference is detectable at n = 10 at median variance and marginal at worst-case variance (§6.3). The exact margin must be set from the corrected baseline's own measured SD, not from legacy-operator variance.

## 16.2 Classification preservation (non-inferiority)

Utility is measured as mean AUC over the 14 NIH abnormality labels using the existing CheXNet evaluation path.

**Logic:** non-inferiority, not superiority. Claim requires the **lower bound of the 95% CI for [AUC(P) − AUC(B)] to lie above −δ**.

**δ: TO BE LOCKED AFTER BASELINE ESTIMATION.** Justification for deferring: δ must be smaller than the classifier's own reproducibility spread to be meaningful, and that spread has not been measured under the corrected operator. Setting δ now would be inventing a number.

## 16.3 Segmentation improvement

Relative to `B`: higher Dice **and** higher IoU **and** lower HD95, each with a 95% CI excluding zero difference, over the same frozen evaluation set.

**Margins: TO BE LOCKED AFTER BASELINE ESTIMATION.**

## 16.4 Reporting rule

All four axes (privacy mean, privacy max, utility, segmentation) are reported for every arm regardless of outcome. A method that wins on privacy and loses on utility is reported as such. **A rigorous negative result is an acceptable and publishable outcome** — this remains the project's standing position.

---

# 17. Implementation requirements for Coding Agent

These requirements implement the frozen protocol; they do not authorize experiments.

**R-1 — Validation metrics.** Extend `validate_snn` to provide validation loss, validation ROC-AUC, and validation accuracy while preserving backward compatibility. AUC uses continuous logits/scores. Accuracy must use the correct 0.5 probability boundary (equivalently logit threshold 0.0).

**R-2 — Machine-readable diagnostics.** Persist per-run training/validation diagnostics, including loss curves, validation AUC/accuracy, best epochs, termination reason, NaN/Inf status, checkpoint state, weight-update state, transform mode, μ, generator checkpoint/hash, pair-file hashes, seed, and timestamps. Keep training diagnostics separate from test metrics.

**R-3 — Run-health classifier.** Implement a standalone classifier returning `NUMERICALLY_INVALID` or `VALID` plus an independent `near_chance` boolean. Exclusion may consult only objective execution-health fields. The API must take no test-derived input.

**R-4 — Restart driver.** Implement the fixed ascending-seed policy:
- screening target 3 valid completed runs, ≤2 infrastructure replacements;
- confirmatory target 10 valid completed runs, ≤5 infrastructure replacements;
- replacement only for `NUMERICALLY_INVALID`;
- completed near-chance runs are never replaced;
- each seed attempt trains exactly once and completed runs are idempotently reusable.

**R-5 — Frozen Top-k protocol.** Track the N=500, seed=42 gallery/probe CSV; canonical configuration is clean gallery / anonymized probe. Support Top-1, Top-5, and MRR with the adaptive representative attacker's embedding.

**R-6 — Determinism check.** Provide a reusable deterministic-arm check that generates identical outputs twice under identical configuration. Stochastic methods must be explicitly marked and handled under §3.4.

**R-7 — Provenance.** Every arm record must contain at least: arm ID, git commit, transform mode, generator checkpoint path/hash, μ, stochastic settings, attacker architecture/hyperparameters, attempted seeds, pair-file hashes, protocol document hashes, frozen Top-k list hash, validity/near-chance states, representative attacker, and timestamps.

**R-8 — Test-leakage firewall.** Enforce pipeline staging:
A. train/validate,
B. classify execution health,
C. choose representative on validation only,
D. persist/freeze representative selection,
E. only then evaluate test.
The representative selector and health classifier must structurally accept no test-derived inputs.

**R-9 — Pair-bootstrap policy (final).** Do **not** implement a naive patient-cluster bootstrap for the dyadic verification pairs. Retain pair-level bootstrap only as a clearly labeled secondary diagnostic: `PAIR-SAMPLING DIAGNOSTIC — NOT PATIENT-LEVEL UNCERTAINTY`.

**R-10 — Summary statistics.** Across attacker restarts use **sample SD (`ddof=1`)**, consistent with the declared power/statistical analysis.

**R-11 — Stub isolation.** Synthetic/stub test metrics must be impossible to confuse with scientific results. Real execution must never silently emit fabricated AUCs. Stub outputs must be explicitly marked invalid for scientific reporting.

**R-12 — Protocol provenance.** The authoritative tracked protocol documents are:
- `research_agent/01_ADAPTIVE_REID_PROTOCOL.md`
- `research_agent/01B_PROTOCOL_AMENDMENT.md`

Their SHA-256 hashes must be written to every future arm provenance record.

# 18. STEP 2A verdict

The protocol is frozen with the STEP 2A.1 amendments and the later R-9 statistical clarification.

It contains:

- no test-derived model selection;
- no performance-based exclusion of completed near-chance attackers;
- a fixed attacker/restart budget;
- validation-only checkpoint and representative-attacker selection;
- primary inference over independent attacker restarts;
- sample SD (`ddof=1`) for restart variability;
- pair-level bootstrap only as a secondary diagnostic, not patient-level uncertainty;
- predeclared escalation for arms with many near-chance attacks;
- explicit anti-cherry-picking and provenance requirements.

The corrected PriCheXy-Net baseline has not yet been measured under this protocol.

**STEP 2A FINAL PROTOCOL: PASS**
