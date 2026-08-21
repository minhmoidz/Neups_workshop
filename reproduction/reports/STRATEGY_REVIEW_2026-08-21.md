# Research Strategy Review — Medical Image Anonymization Privacy Improvement

**Date:** 2026-08-21  
**Repository:** `minhmoidz/Neups_workshop`  
**Branch:** `research/method-restart`  
**Scope:** read-only research-strategy audit; TRAIN/VALIDATION only; no training or evaluation executed

## Executive verdict

**Direction B, as currently implemented, is not a clean test of the proposed mechanism and should not be treated as the most promising next method.** The running `k=3` experiment may still be useful as an exploratory candidate, and it should be allowed to finish without interference, but its result cannot be attributed solely to a stronger verifier.

The main reason is a concrete implementation confound. During each extra verifier step, the script calls the U-Net generator under `torch.no_grad()` while the generator remains in `train()` mode. The U-Net contains BatchNorm layers. Therefore the extra forwards still update BatchNorm running statistics and `num_batches_tracked`; the generator state is changed two additional times per batch even though no generator gradients are computed. This contradicts the script's claim that the generator is “not touched” and that only the verifier update ratio changes.

There are also two scientific gaps behind Direction B:

1. The repository has not measured that the live training verifier is actually weaker than a freshly trained best-response attacker at the same generator snapshot. “One update per generator update” establishes co-adaptation, but does not by itself establish inadequate verifier strength.
2. Increasing the number of updates on the same batch does not reproduce the evaluation attacker. The live critic is optimized on `anon(x1), real(x2)`, whereas the adaptive evaluation attacker is trained and early-stopped on `anon(x1), anon(x2)` and is only then evaluated on `anon(x1), real(x2)`.

### Ranked decision

1. **Top recommendation: Direction A+ — periodic evaluation-matched best-response refresh, with a small replay pool of frozen critics.** This directly attacks the observed train/evaluation asymmetry, breaks persistent co-adaptation, and protects against forgetting or overfitting to one critic trajectory.
2. **Required low-cost baseline: corrected TTUR / fresh-batch multi-step critic training.** This is a useful ablation and possibly a practical solution, but by itself is a training heuristic with limited novelty.
3. **Conditional stabilizers: spectral normalization or a zero-centered/R1-style gradient penalty.** Add only if stronger critic pressure produces measured instability; do not transplant WGAN-GP mechanically into the current BCE Siamese objective.
4. **Independent critic ensemble without periodic best-response refresh.** Potentially robust but expensive and less directly aligned with the evaluation procedure.
5. **Unrolled optimization.** Scientifically relevant but not cost-effective for the current ResNet-50 Siamese critic and 16 GB GPU.

No existing result supports a publishable privacy-improvement claim. C4 is privacy-non-inferior conditional on two fixed generator checkpoints, but it is not privacy-superior. A single favorable AUC from the current k=3 run would also be insufficient.

---

## 1. Evidence reviewed

The review began with the repository's required index and followed its links before forming a recommendation.

| Artifact | Blob SHA reviewed | Role |
|---|---|---|
| [`RESEARCH_DIRECTION_SUMMARY.md`](./RESEARCH_DIRECTION_SUMMARY.md) | `3a964f45e86bc3d294d792dcf5802e752b80bb18` | Current project state and evidence index |
| [`S2_CONFIRMATORY_DESIGN_PROPOSAL.md`](./S2_CONFIRMATORY_DESIGN_PROPOSAL.md) | `91d6fdf45bc25a78a829f06fa16237679f6a6dbc` | Seed-design rationale and final Stage A analysis |
| [`PAPER_REPRO_PROTOCOL.md`](./PAPER_REPRO_PROTOCOL.md) | `390bbf517f44aa96d6e300ffb9e74afb6bb252dc` | Historical protocol and documented predeclaration limitation |
| [`pilot_summary_final.json`](../s2_pilot/results/pilot_summary_final.json) | `aa94070f63b24c7db006628924f9531100491ce7` | Raw 50-run pilot records plus certified seed 42 |
| [`run_stage_a_pilot.py`](../s2_pilot/run_stage_a_pilot.py) | `5f5d7f6a792c4d4be7916e582fba13d228e61107` | Pilot attacker implementation |
| [`anonymizer_runner.py`](../../research_agent/m2_dev/anonymizer_runner.py) | `7f3de01ab101872b69210219e3ef775e814a993b` | Certified B_dev/C4 anonymizer training path |
| [`run_hardened_verifier.py`](../method_dev/run_hardened_verifier.py) | `5982bdc0b51fab90c2cd39cf731c72a5339843ce` | Direction B implementation |
| [`dev_attacker.py`](../../research_agent/m2_dev/dev_attacker.py) | `d0e4ec9177d18d16fd6787dcd2cdd4f72e5f07a9` | Adaptive attacker training and selection geometry |
| [`eval_reid_val.py`](../../research_agent/m2_dev/eval_reid_val.py) | `22786cfc93170d14c345e13888c6ed07add267c4` | Scientific adaptive Re-ID evaluation geometry |
| [`M2_S1_C4_RESULT.md`](../../research_agent/M2_S1_C4_RESULT.md) | `c546363b13583ea6b5f638bbce968402b0e2865c` | Certified single-seed S1 result |
| [`M2_S1_EXECUTION_LOCK.json`](../../research_agent/M2_S1_EXECUTION_LOCK.json) | `c16a064a190736d031ef443d6129d550ec247830` | Frozen S1 scientific choices |
| [`PROTOCOL_AUTHORITY.md`](../../research_agent/PROTOCOL_AUTHORITY.md) | `e9dd2633aa3bc8d49d4db24d050aa791c89788d4` | Authority hierarchy and method semantics |

The live Direction B logs and checkpoint manifest named in the index were not present in the pushed branch, so run progress and completion could not be independently verified from GitHub. No process was stopped, restarted, signaled, or otherwise touched, and no GPU workload was launched.

---

## 2. Current state: what the evidence does and does not establish

### 2.1 Certified S1

The certified seed-42 comparison reports:

| Metric | B_dev | C4 | C4 − B_dev |
|---|---:|---:|---:|
| Adaptive Re-ID VAL AUC | 0.8132 | 0.8023 | −0.0109 |
| Classification macro VAL AUC | 0.7841 | 0.8056 | +0.0216 |

This passes the frozen non-inferiority-style privacy gate (`Δpriv ≤ +0.03`) and the utility gate. It does not prove that C4 reduces privacy leakage because it is one anonymizer seed and one attacker seed.

### 2.2 Independent recomputation from the raw Stage A JSON

The raw JSON contains 25 pilot attacker seeds per arm (`43–67`) plus certified seed `42`, giving 26 AUCs per fixed generator. All 50 pilot runs:

- used the same 2,000-pair validation file and identical pair-order hash;
- used the expected, arm-specific generator SHA;
- terminated by early stopping;
- selected attacker epochs 2–7 and completed 8–13 epochs;
- consumed 16.80 GPU-hours in total, or about 20.2 minutes per attacker run.

Reviewer recomputation from the per-run AUC values matches the report:

| Quantity | Recomputed value |
|---|---:|
| B_dev mean / SD | 0.823686 / 0.030935 |
| C4 mean / SD | 0.832630 / 0.028766 |
| Paired Δ mean / SD | +0.008943 / 0.038472 |
| Pearson correlation | 0.171019 |
| Seeds with Δ > +0.03 | 7 / 26 |
| Seeds with C4 AUC lower than B_dev | 11 / 26 |
| Δ range | [−0.077171, +0.076127] |

Excluding certified seed 42 barely changes the conclusion: the 25-pilot-seed mean Δ is `+0.009738`. The single certified seed is therefore not driving the multi-seed result.

The paired t-test in the design report is arithmetically consistent with these values: non-inferiority against `+0.03` is supported, while superiority against `0` is not.

### 2.3 Correct interpretation

The Stage A result supports only this conditional statement:

> For the two fixed seed-42 generator checkpoints, under the frozen Siamese architecture, training recipe, validation pairs, and attacker-seed distribution sampled by seeds 42–67, C4 is non-inferior to B_dev by the project's +0.03 privacy margin.

It does not estimate anonymizer-seed variance, data-sampling variance, attacker-architecture variance, or privacy against a broader attacker family. Consequently, 26 attacker seeds must not be described as 26 independent method replications.

---

## 3. Critical audit of the “weak live critic” hypothesis

### 3.1 Plausible, but not yet measured

The certified anonymizer runner alternates one generator update and one verifier update per batch. The evaluation attacker is trained against a frozen generator and early-stopped after no improvement for five epochs. This is a real asymmetry and a plausible failure mechanism.

However, the stronger claim that the live critic “never converges” is not established by the available evidence:

- It begins from a released pretrained verifier checkpoint, not a random initialization.
- It receives one update on every batch for 250 epochs. With 10,000 training pairs and batch size 16, that is approximately 625 verifier updates per epoch. Even the B_dev checkpoint selected at epoch 13 was preceded by roughly 8,750 online verifier updates.
- No repository artifact compares the live verifier at a frozen generator snapshot against independently reinitialized attackers trained to the same stopping rule.
- No live-verifier TRAIN/VAL AUC, generalization gap, gradient norm, calibration, or best-response gap is logged.

Therefore the gap must be treated as a hypothesis to test, not as an established diagnosis.

### 3.2 The geometry mismatch is at least as important as the update ratio

The current live verifier is trained on:

`anon(x1), real(x2)`

The adaptive evaluation attacker is trained and selected on:

`anon(x1), anon(x2)`

and only evaluated on:

`anon(x1), real(x2)`

Direction B increases optimization pressure on the first objective. It does not make the training adversary follow the evaluation attacker's learning problem. A method intended to close the train/evaluation gap should reproduce both the attacker's initialization/training geometry and its frozen-generator advantage.

### 3.3 Direction B's BatchNorm confound

The script performs each extra verifier step as follows:

1. The generator remains in `train()` mode for the whole epoch.
2. A fresh generator forward is called inside `torch.no_grad()`.
3. The resulting image is detached and used for an additional verifier update.

`torch.no_grad()` disables autograd recording; it does not disable stateful module updates. [`UNet_PriCheXyNet.py`](../../networks/UNet_PriCheXyNet.py) contains BatchNorm after every convolution block. Thus every extra forward changes BatchNorm buffers. For default `k_extra_verifier_steps=2`, the generator's BatchNorm statistics are updated three times per training batch rather than once.

Implications:

- The experiment is not a pure `3:1` critic/generator update-ratio intervention.
- A favorable result could be caused partly by altered generator normalization dynamics.
- An unfavorable result cannot falsify multi-step critic training because the intervention is confounded.
- The generated images used by later batches are affected by the additional BatchNorm updates.

The current run should be labeled something like **`k3 + generator-BN-drift exploratory pilot`**, not a clean Direction B result.

### 3.4 Repeating the same batch is not a best response

The two extra steps reuse identical `inputs1`, `inputs2`, and labels. Because the generator is deterministic, the extra fakes are also nearly identical aside from the unintended BatchNorm state evolution. Three optimizer steps on one batch can improve batch fitting without producing a critic that generalizes across the pair distribution.

A corrected multi-step experiment should use a separate deterministic critic iterator drawing fresh TRAIN batches, while leaving the generator's own sampler traversal and update count unchanged.

### 3.5 Direction B drops several certified guards

The reproduction script preserves core models and config hashes, but it is not literally “line-for-line plus two steps.” Relative to the certified runner it removes or weakens:

- expected epoch-order hash comparison, retaining only logging of the observed hash;
- gradient and post-step parameter finiteness checks;
- C4 gradient diagnostics;
- scientific max-epoch and seed restrictions;
- the strict manifest fields used by the certified path.

These differences may not change the intended objective, but they weaken the evidence chain and should be restored in any confirmatory implementation under `reproduction/`.

### 3.6 Moving-critic checkpoint selection is not comparable across methods

Both runners select the generator checkpoint by `val_ac_bce + val_privacy_term`, where `val_privacy_term` is produced by the method's own evolving verifier. Once verifier dynamics change, the scale and difficulty of the selection criterion also change. “Method-neutral” with respect to C4's feature loss is not the same as method-independent across adversary-training schemes.

For a fair method comparison, generator checkpoint selection must use a fixed, predeclared selector that does not change with the candidate method, or it must use an inner adaptive-attacker procedure applied identically to all arms. The final privacy-evaluation pairs must not also drive candidate selection.

---

## 4. Ranked strategy recommendations

### Rank 1 — Direction A+: periodic evaluation-matched best-response refresh with critic replay

#### Method

At predeclared refresh points:

1. Freeze the current generator and put it in `eval()` mode.
2. Initialize a fresh Siamese attacker using the same initialization family as the evaluation attacker.
3. Train and early-stop it on `anon/anon` TRAIN-derived data, with a disjoint inner selection split.
4. Freeze that attacker and use its differentiable `anon/real` score as a privacy loss for subsequent generator updates.
5. Retain a small pool—recommended size 2 initially—containing the newest best response and the hardest historical critic. Optimize the generator against the maximum or a predeclared smooth maximum of their privacy losses.

This is Direction A plus limited historical replay. The fresh restart breaks co-adaptation; matching the evaluation attacker's training geometry addresses the actual threat-model gap; replay reduces cycling and “forgetting” attacks that an earlier critic discovered.

This recommendation is consistent with the game-theoretic motivation of historical-opponent training in [Fictitious GAN](https://arxiv.org/abs/1803.08647) and with multiple-discriminator training in [GMAN](https://openreview.net/forum?id=Byk-VI9eg). These papers do not prove effectiveness for CXR privacy; they motivate the design, which still requires the repository's adaptive-attacker evaluation.

#### Why it ranks above naive hard restart

A single restarted critic can become obsolete after the generator moves. A small replay pool preserves attack diversity without the cost of a large ensemble. Pool size 2 is a starting design constraint, not a tuned result; if two critics exceed memory or latency limits, use one current best response first and add replay only if measured forgetting occurs.

#### Proposed refresh schedule

Because the certified B_dev checkpoint was selected early, a schedule that starts only at epoch 25 would miss the most relevant period. A reasonable predeclared pilot schedule is:

`epochs {0, 5, 10, 20, 40, 80, 120, 160, 200, 240}`

The schedule must be frozen before the run. It is a proposal for human approval, not an optimized schedule.

#### Estimated cost per anonymizer seed

- Certified B_dev reference: 31.57 GPU-hours.
- One fresh adaptive attacker: observed mean about 0.336 GPU-hours.
- Ten refreshes: about 3.36 GPU-hours in attacker training.
- One active best-response critic: approximately **35–40 GPU-hours** total, allowing for integration overhead.
- Two-critic replay pool: approximately **40–50 GPU-hours** total, because generator backpropagation passes through two full ResNet-50 critics.

These are planning ranges. A short TRAIN-only timing smoke run is required to replace them with measured values before authorization.

### Rank 2 — corrected TTUR / fresh-batch multi-step critic baseline

The [two-time-scale update rule](https://papers.nips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium) motivates separate learning time scales for generator and discriminator. A clean repository baseline should compare:

- `1:1` with a higher verifier learning rate (TTUR-style);
- `3:1` using fresh critic batches;
- optionally `3:1` plus a lower verifier learning rate so that update count and total effective step size are not conflated.

Required corrections:

- generator in `eval()` for critic-only fake generation, with parameters and BatchNorm buffers asserted unchanged;
- separate, deterministic critic-only TRAIN iterator;
- exact generator update-count and generator TRAIN-order parity;
- restored NaN/Inf, gradient, parameter, order-hash, and manifest guards;
- direct best-response-gap telemetry.

Estimated cost: **32–36 GPU-hours per seed**, plus adaptive-attacker evaluation. It is likely the best practical baseline, but a fixed `k=3` value alone is weak as a Q2 method contribution.

### Rank 3 — conditional critic stabilization

[Spectral normalization](https://openreview.net/forum?id=B1QRgziT-) is computationally light and was proposed to stabilize discriminator training. [Mescheder et al.](https://proceedings.mlr.press/v80/mescheder18a.html) show why finite discriminator updates do not automatically guarantee convergence and analyze zero-centered gradient penalties for stabilizing adversarial games.

For this project:

- spectral normalization is the lower-cost first stabilizer;
- an R1/zero-centered input-gradient penalty is more compatible with the current logistic/BCE critic than directly importing WGAN-GP;
- [WGAN-GP](https://arxiv.org/abs/1704.00028) should not be copied verbatim without changing and justifying the current objective.

Use a stabilizer only if measured critic gradients, losses, or best-response gaps show instability under stronger pressure. Otherwise it adds complexity without addressing the core adaptive-attacker mismatch.

Estimated cost: **34–45 GPU-hours per seed**, depending on the penalty and its frequency.

### Rank 4 — multiple independently initialized live critics

An ensemble of two or three critics can reduce dependence on one initialization and one failure mode. However, all critics can remain weak together if trained co-adaptively, so ensembling alone is less direct than fresh best-response refresh.

Estimated cost: **45–65 GPU-hours per seed** for two to three ResNet-50 critics, with substantial memory risk. If used, aggregate with `max` or smooth-max rather than an unweighted mean that lets weak critics dilute the strongest attack.

### Rank 5 — unrolled adversary optimization

[Unrolled GANs](https://openreview.net/forum?id=BydrOIcle) explicitly differentiates through future discriminator optimization and is conceptually close to the problem here. For this repository, unrolling a ResNet-50 Siamese optimizer through several steps would greatly increase memory and compute and may exceed the available 16 GB GPU.

Estimated cost: **at least 70–120 GPU-hours per seed**, with high implementation and memory risk. Do not prioritize it before simpler best-response refresh and TTUR baselines fail.

---

## 5. Rigorous evaluation plan for Direction A+

### 5.1 Phase 0 — implementation and mechanism gates

No full run should be authorized until the new script under `reproduction/` passes all of the following:

1. TRAIN/VALIDATION firewall and governed SHA checks remain fail-closed.
2. Generator optimizer-step count and generator sampler-order hashes match the approved control schedule.
3. A critic-only step leaves every generator parameter and buffer byte-identical, including all BatchNorm buffers.
4. Attacker refresh uses a fresh initialization and the declared `anon/anon` train/selection geometry.
5. The frozen refreshed critic supplies generator gradients only through the declared `anon/real` privacy objective.
6. Critic-pool membership, checkpoint SHAs, refresh epochs, and aggregation rule are logged.
7. Every load-bearing loss, gradient, and parameter has NaN/Inf fail-closed checks.
8. Per-pair predictions are retained with pair IDs and patient IDs; do not strip `y_true` and `y_score` from the only saved artifact.

### 5.2 Mechanism measurement: test the premise before the outcome

At predeclared generator snapshots, train fresh best-response attackers without updating the generator. For each snapshot record:

- live/replay-critic AUC under `anon/real`;
- mean AUC of fresh attackers under the same scientific geometry;
- TRAIN and inner-VAL BCE for each attacker;
- calibration and gradient norms with respect to anonymized inputs;
- best-response gap:

`gap_t = mean(AUC_fresh_best_response,t) − max(AUC_training_pool,t)`.

The weak-critic hypothesis is supported only if the original 1:1 run has a positive reproducible gap and Direction A+ reduces it. A lower train-time privacy loss without a lower fresh-attacker AUC does not validate the mechanism.

### 5.3 Stage 1 — one-generator screening, explicitly non-confirmatory

Use one anonymizer seed and **8 attacker seeds** for the candidate. Pair each seed with the existing raw B_dev AUC for the same attacker seed, provided all code/checkpoint/data SHAs match; otherwise rerun the paired B_dev attackers.

Cost:

- candidate generator: 40–50 GPU-hours for the two-critic design;
- eight adaptive attackers: about 2.7 GPU-hours;
- total: **43–53 GPU-hours**, excluding any baseline reruns.

Predeclare the screening gate:

- paired mean Re-ID delta `candidate − B_dev ≤ −0.03`;
- at least 6 of 8 paired deltas are negative;
- utility passes the frozen zero-degradation gate, or a separately justified utility non-inferiority margin frozen before execution;
- no numerical/provenance failure.

This is a futility/advancement rule, not a significance claim. Failure stops the route. Passing authorizes generator-seed variance estimation, not publication language.

### 5.4 Stage 2 — generator-seed variance pilot

The true independent method unit is the anonymizer training seed. Before this stage, freeze the redesigned patient-disjoint nested protocol described in §6.3 and regenerate the paired B_dev reference under that protocol. Run **three paired generator seeds total** for B_dev and Direction A+, with **five paired attacker seeds per generator checkpoint**.

For each generator seed `g`, compute:

`Delta_g = mean_a[AUC_method(g,a) − AUC_Bdev(g,a)]`.

Use the three `Delta_g` values only to estimate between-generator variance and update the confirmatory power calculation. Do not declare efficacy from three seeds.

Upper-bound planning cost, including rerunning both arms and all attackers:

| Component | Estimated GPU-hours |
|---|---:|
| 3 × B_dev generators | 94.7 |
| 3 × Direction A+ generators | 120–150 |
| 6 checkpoints × 5 attackers | 10.1 |
| **Total** | **225–255** |

Existing seed-42 artifacts can reduce this total, but the conservative budget should not assume reuse until exact comparability is verified.

### 5.5 Stage 3 — powered confirmatory comparison

After Stage 2, choose the paired generator-seed count from the observed SD of `Delta_g`. For a one-sided superiority test with alpha 0.05, target effect `−0.03`, and paired generator-level SD `sigma_g`, the planning approximation is:

`n_g = sigma_g^2 * (z_(1-alpha) + z_(1-beta))^2 / 0.03^2`.

Sensitivity table:

| Generator-level SD of Δ | 80% power | 90% power |
|---:|---:|---:|
| 0.02 | 3 | 4 |
| 0.03 | 7 | 9 |
| 0.04 | 11 | 16 |
| 0.05 | 18 | 24 |

Use `n_g = max(5, powered estimate)`. Five is a floor, not a claim that five is adequately powered.

Primary privacy analysis:

1. Average paired attacker-seed deltas within each generator seed.
2. Test `H0: E[Delta_g] >= 0` versus `H1: E[Delta_g] < 0` at generator-seed level.
3. Report the one-sided 95% upper confidence bound and a two-sided 95% CI.
4. Report an exact sign-flip/permutation sensitivity analysis and a hierarchical bootstrap that resamples generator seeds, attacker seeds within generator, and patient clusters in the evaluation pairs.
5. Keep one predeclared primary method. If several methods reach confirmation, control multiplicity or use an independent confirmation design.

Primary success definition:

- upper one-sided 95% confidence bound on mean privacy delta is `< 0`;
- point estimate is at most `−0.03` absolute AUC, using the project's existing 0.03 threshold as the predeclared practical-effect target;
- utility gate passes across generator seeds;
- result is not driven by one generator seed, one attacker seed, or one pathology;
- all provenance and numerical gates pass.

The reported B_dev mean `0.8237` remains the conditional seed-42 reference. The confirmatory test must use raw paired B_dev results for the same generator/attacker seed design; it must not use `0.8237` as if it were a known population constant.

### 5.6 What is and is not publishable

#### Would count as a genuine improvement

- statistically significant lower adaptive Re-ID AUC at generator-seed level;
- an absolute reduction of at least 0.03 AUC or another predeclared and justified smallest effect of interest;
- preserved classification utility under the same paired generator seeds;
- evidence that the fresh-best-response gap is reduced, supporting the claimed mechanism;
- robustness across multiple independently trained attackers and, ideally, more than one attacker architecture as a secondary threat-model check;
- complete prediction-level and provenance artifacts.

#### Would not count

- one favorable attacker seed;
- one favorable generator seed, even with 26 attacker seeds;
- non-inferiority alone;
- a lower train-time `privacy_term` without lower fresh adaptive-attacker AUC;
- significance obtained by treating attacker seeds as independent generator replications;
- comparison of a single candidate AUC to the baseline mean without raw paired differences;
- a result selected after trying several ratios, refresh schedules, or penalties on the same evaluation pairs without multiplicity control;
- privacy gain accompanied by an unapproved utility loss;
- mechanism claims from the current BatchNorm-confounded k=3 run.

---

## 6. Statistical and protocol gaps to fix

### 6.1 Attacker seeds are nested repeats, not method replicates

The n=26 t-test is valid only for attacker-initialization variability conditional on fixed generators and fixed pairs. The confirmatory experimental unit must be the generator seed. Otherwise the design has pseudoreplication.

### 6.2 Repeated use of the same 2,000 pairs

All attacker seeds are evaluated on exactly the same validation pairs. Seed-level inference does not include patient/data uncertainty. Future runs need prediction-level artifacts and a patient-clustered or multiway patient-endpoint bootstrap because the same patient can contribute to multiple pairs.

### 6.3 Evaluation data also influence selection

The current attacker uses the validation pair set for early stopping under `anon/anon`, then reports AUC on the same pairs under `anon/real`. The generator also uses validation loss for checkpoint selection. Although the geometries differ, this is not a fully independent confirmation set. Moreover, the same validation AUC has now informed C4 and Direction B development, so it cannot become an untouched confirmation set retroactively.

Use the existing validation result and the `0.8237` B_dev mean for continuity screening only. A publishable confirmation needs a redesigned patient-disjoint nested protocol derived from TRAIN data (inner training, method/checkpoint selection, and a locked confirmation fold) or nested cross-validation. That redesign requires a newly paired B_dev baseline; `0.8237` would remain the historical fixed-generator reference rather than the null constant for the new protocol.

### 6.4 “Trained to convergence” is too strong

The 50 pilot attackers selected epochs 2–7 and stopped after 8–13 epochs. Early stopping is a defined and reproducible attack budget, but it is not proof of global or even local convergence. The report should say **“trained under the frozen adaptive-attacker budget with patience-5 early stopping”**, not “fully converged,” unless convergence diagnostics are added.

### 6.5 One attacker architecture is an incomplete privacy threat model

Twenty-six seeds of one ResNet-50 Siamese network test optimization stochasticity, not attack-family robustness. A Q2 claim should retain the frozen attacker as primary for comparability and add at least one predeclared secondary attacker family or embedding loss. This must be secondary so the method is not selected against an ever-expanding attack suite.

### 6.6 No current segmentation conclusion

Segmentation remains blocked by evaluator provenance. Do not claim preservation or improvement. Classification utility is the only currently certified utility endpoint.

### 6.7 Candidate multiplicity and validation overuse

Testing k ratios, learning-rate ratios, hard-restart schedules, pool sizes, spectral normalization, and penalties on the same validation metric creates a hidden multiple-comparisons problem. Use a strict funnel:

1. mechanism diagnostics and inner-TRAIN selection;
2. one predeclared primary candidate;
3. one powered validation analysis.

---

## 7. Immediate action order

1. **Do not interfere with the current k=3 process.** When it finishes, preserve its artifacts and label it as a BatchNorm-confounded exploratory run.
2. **Before spending attacker-evaluation cost, inspect the completed run for numerical validity, exact order logs, selected epoch, BatchNorm buffer divergence, and whether the checkpoint exists.** If the run is invalid or incomplete, do not rescue it with ad hoc continuation.
3. **If scientifically useful, run only an 8-attacker-seed conditional screen on that checkpoint.** Treat any result as hypothesis-generating.
4. **Implement a clean mechanism probe under `reproduction/`:** frozen-snapshot live critic versus fresh best-response attackers. This directly tests the assumed cause.
5. **Implement Direction A+ only after the mechanism probe supports the weak-critic gap.** Start with one active periodic best-response critic; add the size-2 replay pool only if forgetting/cycling is measured.
6. **Implement corrected TTUR/fresh-batch k:1 as the mandatory baseline.** It is cheaper and separates “more optimization” from “fresh best response plus replay.”
7. **Use spectral normalization or R1-style regularization only when telemetry demonstrates instability.**
8. **After a one-seed screen, spend the large budget on paired generator seeds, not dozens more attacker seeds on one generator.**

---

## 8. Final recommendation

The project should not pivot from C4 directly to “k=3 is the method.” The evidence currently supports a narrower statement: adaptive-attacker variance is large, C4 is not privacy-superior, and a co-adaptive live critic may be part of the problem. The current Direction B code does not isolate that mechanism.

The strongest defensible next route is:

> **Evaluation-matched periodic best-response attacker refresh, optionally with a two-critic historical replay pool, evaluated by paired generator seeds and nested adaptive-attacker seeds.**

Corrected TTUR/fresh-batch multi-step training should be the principal baseline. A publishable result requires both generator-level privacy superiority and utility preservation; non-inferiority, live-loss improvement, or a single-seed AUC is not enough.

This strategy is aligned with adversarial privacy as a constrained minimax problem, as formalized by [Generative Adversarial Privacy](https://arxiv.org/abs/1807.05306) and [Privacy-Preserving Adversarial Networks](https://arxiv.org/abs/1712.07008), while retaining the repository's stronger empirical requirement: privacy must be measured against freshly retrained adaptive attackers, not inferred from the training critic.
