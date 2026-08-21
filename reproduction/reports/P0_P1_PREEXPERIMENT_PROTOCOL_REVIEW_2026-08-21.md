# P0–P1 PRE-EXPERIMENT PROTOCOL REVIEW
## Protocol-Equivalence Bridge (P0) and Frozen-Generator Adversary Mechanism Probe (P1)

**Date:** 2026-08-21
**Status:** DESIGN + SOURCE AUDIT ONLY. No experiment executed, no GPU work launched, no governed file modified.
**Canonical source lock:** `minhmoidz/Neups_workshop`, branch `research/method-restart`, commit `34bca1f74662275ced0418ea183a3d9d5ef81f88` — identity verified before reading source (`git rev-parse --abbrev-ref HEAD` = `research/method-restart`; `git rev-parse HEAD` = `34bca1f…`). VERIFIED_FROM_SOURCE.
**Review-evidence lock:** branch `review/g0-2a3-reaudit-closeout-20260821`, commit `a38ab368f3ce84cdea7724ee3416d0b3a9386760`, inspected only via read-only `git show <commit>:<path>`. The worktree was never switched.
**Upstream lock:** PriCheXy-Net commit `29245d1f71571898d9527417df4ae3f63a8695f6`. The working-tree copies of `agents/Agent.py`, `utils/utils.py`, `utils/ACLoss.py`, `datasets/Dataset.py`, `datasets/SiameseDataset.py`, `networks/SiameseNetwork.py` are byte-identical to this anchor (`git log -1 -- <path>` = `29245d1` for each). VERIFIED_FROM_SOURCE.

---

## 1. EXECUTIVE VERDICT

1. **Direction B v1 (`reproduction/method_dev/run_hardened_verifier.py`) is NOT a clean causal test of the weak-verifier hypothesis.** It is confounded by (a) generator BatchNorm drift during extra verifier-only forwards because the generator remains in `.train()` mode under `torch.no_grad()`, (b) same-batch repetition for extra steps, (c) a checkpoint-selection criterion that uses the arm's own evolving verifier, and (d) removal of several certified fail-closed guards. Its running output may be retained as an **exploratory reference only**. VERIFIED_FROM_SOURCE.
2. **The weak-live-critic hypothesis remains UNTESTED.** No repository artifact compares a live training verifier at a frozen generator snapshot against fresh best-response critics under matched budgets. PROPOSED_HYPOTHESIS.
3. **A Direction B v2 runner exists on the review branch and is intentionally design-only** (`run()` stub raises; execution requires a human-approved manifest that does not exist). It is not canonical method source. VERIFIED_FROM_SOURCE.
4. **The ≈0.60 vs 0.8237 discrepancy is very likely protocol/cohort-dependent, not generator regression** — but this is currently supported only by one approximate historical datum (a released-code generator measured at ≈0.818 on VAL pairs in "STEP-8R", vs 0.6080 mean TEST AUC for the same generator family) with incomplete provenance. P0 must convert this into a locked, paired result. SUPPORTED_BY_EXISTING_DATA / UNRESOLVED provenance details.
5. **P0 protocol: SPECIFICATION PASS (design-ready), execution BLOCKED pending external human approval.** One blocking provenance item must be resolved first: two byte-distinct candidate "released generator" artifacts exist (SHA `10122689…` vs `4d82dcdd…`); Anchor U identity must be fixed before any P0 run.
6. **P1 protocol: SPECIFICATION PASS with one blocked sub-cell** — no paired live-critic checkpoint exists for either generator anchor (the certified runner never saves verifier state), so H1's live-reference cells are UNAVAILABLE unless a small live-snapshot capture is separately authorized.
7. Current VAL is **not** a locked confirmation resource; the patient-graph bootstrap is **unvalidated**; segmentation experimentation is **BLOCKED** by evaluator provenance.
8. **No execution authorization is granted by this document.** GPU_AUTHORIZATION: NONE. NEXT_REQUIRED_ACTION: EXTERNAL_HUMAN_REVIEW.

---

## 2. EVIDENCE AND SOURCE FILES INSPECTED

All claims below cite files as read at the locked commits. Labels per protocol §12.

### 2.1 Project evidence (canonical worktree @ `34bca1f`)
| Artifact | Role |
|---|---|
| `reproduction/reports/RESEARCH_DIRECTION_SUMMARY.md` | Entry-point index; certified S1, Stage A, Direction A/B state |
| `reproduction/reports/S2_CONFIRMATORY_DESIGN_PROPOSAL.md` | Stage A design, seed-variance framework, n=26 results |
| `reproduction/s2_pilot/results/pilot_summary_final.json` | Raw 50-run Stage A pilot records (+certified seed 42) |
| `reproduction/reports/M2_S1_C4_RESULT.md` (via `research_agent/M2_S1_C4_RESULT.md`) | Certified single-seed S1 result, checkpoint SHAs, epoch selection |
| `reproduction/reports/PRICHEXY_PAPER_REPRODUCTION.md` | R1-A reproduction verdict (FAIL, 60.80±4.35 vs paper 57.7±4.0); STEP-8R VAL note |
| `reproduction/reports/FINAL_10SEED_PRICHEXY_REPRODUCTION.md` | R1-FINAL: n=10 upstream TEST-pair reproduction, full protocol |
| `reproduction/reports/PAPER_REPRO_PROTOCOL.md` | Historical predeclaration erratum context |
| `research_agent/M1_1_SEGMENTATION_PROVENANCE.md` | `SEGMENTATION_STILL_BLOCKED` determination |

### 2.2 Strategy report provenance note (UNRESOLVED item, disclosed)
`STRATEGY_REVIEW_2026-08-21.md` is **not present at the locked HEAD** and not present at review commit `a38ab368`. It exists at commit `a3513e5` ("docs: add anonymization strategy review"), which is contained in `remotes/origin/research/method-restart` but is NOT an ancestor of local HEAD `34bca1f`. It was read via read-only `git show a3513e5:reproduction/reports/STRATEGY_REVIEW_2026-08-21.md`. This report audits its arguments rather than accepting them; its central BatchNorm-confound finding was independently re-derived from source (see §4). UNRESOLVED: why the strategy review commit and the locked method tip have diverged histories — human curation decision required.

### 2.3 Current implementation (canonical @ `34bca1f`)
`research_agent/m2_dev/anonymizer_runner.py`; `research_agent/m2_dev/dev_attacker.py`; `research_agent/m2_dev/eval_reid_val.py`; `research_agent/m2_dev/evaluator_common.py`; `reproduction/method_dev/run_hardened_verifier.py`; `networks/UNet_PriCheXyNet.py`; `networks/SiameseNetwork.py`; `utils/VerificationLoss.py`; `research_agent/m0_port/ACLoss.py` (diffed against `utils/ACLoss.py`); `datasets/Dataset.py`; `datasets/SiameseDataset.py`.

### 2.4 Review branch (@ `a38ab368`, read-only)
`reproduction/method_dev/run_hardened_verifier_v2.py`; `reproduction/protocol_v2/state_invariants.py`; `reproduction/protocol_v2/role_manifest.py`.
Note: `state_invariants.py` and `role_manifest.py` do **not** exist under `reproduction/method_dev/` at that commit; they live under `reproduction/protocol_v2/`.

### 2.5 Upstream implementation (@ `29245d1`)
`agents/Agent.py` (structure inspected); `utils/utils.py` (`train`, `validate`, `train_snn`, `validate_snn`, `test_snn` read line-by-line); `utils/ACLoss.py`; `utils/VerificationLoss.py`.

### 2.6 Independent recomputation from raw pilot JSON (VERIFIED_FROM_EXISTING_DATA)
Recomputed from `pilot_summary_final.json` (25 pilot seeds/arm; seed 42 stored separately):
- B_dev pilot-only: mean AUC 0.824105, SD 0.031498; mean trajectory 1204.1 s (SD 121.5; range 913–1474).
- C4 pilot-only: mean AUC 0.833843, SD 0.028672; mean trajectory 1214.9 s.
- Paired deltas (n=25): mean +0.009738, SD 0.039047.
- Including certified seed 42 reproduces the reported n=26 values (B_dev 0.8237/0.031; C4 0.8326/0.0288; Δ +0.0089/0.0385).
- Total measured compute of all 50 pilot runs: 16.80 GPU-hours ⇒ **mean 1209.5 s ≈ 0.336 GPU-h per attacker trajectory**.
These match §3.1's stated facts. The 26 seeds are attacker-optimization replicates conditional on two fixed generators and one cohort — not generator replications. VERIFIED_FROM_EXISTING_DATA.

### 2.7 Process check (start)
`ps aux | grep '[r]un_hardened_verifier'` → PID 1875303 active: `run_hardened_verifier.py --arm B_dev --k_extra_verifier_steps 2 --seed 42 --max_epochs 250 --tag k3`. Untouched throughout. The final check is repeated in §16.

---

## 3. UPSTREAM-VERSUS-CURRENT IMPLEMENTATION MAP

| Behavior | Upstream `29245d1` | Current M2 dev pipeline `34bca1f` | Inherited / changed |
|---|---|---|---|
| Anonymizer training loop | `utils/utils.py::train` (249–397) | `anonymizer_runner.py::train_epoch` (296–438) | Inherited semantics + local guards |
| Verifier:update ratio | 1 verifier step per batch (utils.py:371–380) | 1:1 (anonymizer_runner.py:389–404) | INHERITED |
| Fake computed before generator update, reused after | fakes at utils.py:313–323; gen step 352–355; verifier consumes `fakes_1.detach()` (361) | fakes at anonymizer_runner.py:315; gen step 386; verifier consumes detached fake (391) | INHERITED ("stale fake") |
| Generator mode during combined step | `generator.train()` (299); BN buffers update on every forward | same (anonymizer_runner.py:298); UNet has BN after every conv (UNet_PriCheXyNet.py:107,119) | INHERITED |
| Generator privacy term | `−log(1−σ(z))` via VerificationLoss sigmoid + log (utils.py:338–344) | `softplus(z)` (anonymizer_runner.py:338–339) | MATHEMATICALLY EQUIVALENT (softplus z ≡ −log(1−σ(z))); float64 logging differs. VERIFIED_FROM_SOURCE |
| Verifier training geometry | anon(x1) vs real(x2) (utils.py:361–362) | same (anonymizer_runner.py:391–392) | INHERITED |
| AC critic update | BCE on resized fakes (utils.py:383–390) | same (406–419) | INHERITED |
| ACLoss loss-model refresh | deepcopy every forward (`utils/ACLoss.py`) | `refresh()` called at forward start (`m0_port/ACLoss.py` docstring + diff); C4 extensions with detached source features | REPAIRED PORT, baseline-equivalent; SHA-gated (evaluator_common.py:66–67) |
| Attacker training | `train_snn`: anon/anon, `perturbation_net.eval()` (utils.py:534–556) | `DevAttacker.train_epoch`: anon/anon, frozen generator eval-mode (dev_attacker.py:153–187; loader fn sets eval, 59–64) | INHERITED geometry |
| Attacker selection | min val BCE (released code) | min val BCE on anon/anon (dev_attacker.py:189–213) | INHERITED |
| Privacy evaluation geometry | `test_snn`: anon(x1)/real(x2), sigmoid scores (utils.py:681–760) | `evaluate_reid_val_mixed`: anon(x1)/real(x2) (eval_reid_val.py:54–59) | INHERITED; fold moved TEST→VAL |
| Score orientation | sigmoid, higher=more likely same patient (docstring utils.py:706) | same; sklearn `roc_auc_score` on raw probabilities (eval_reid_val.py:59,76) | INHERITED; no post-hoc flip anywhere |
| SNN preprocessing | expand→ImageNet normalize, no resize (utils.py:563–571) | identical `snn_preprocess` (evaluator_common.py:590–594) | INHERITED |
| Operator | legacy: grid=identity−μ·grid, blur whole grid, kernel 9 σ 2, border, align_corners=True | identical constants (evaluator_common.py:38–43, 568–587) | INHERITED (locked) |
| Optimizers | Adam lr per config | Adam lr=1e-4 both G and verifier (anonymizer_runner.py:222–223) | INHERITED equality (no TTUR) |
| Verifier initialization | pretrained verification model | frozen SHA-gated `pretrained_verification_model.pth` (evaluator_common.py:60–61; anonymizer_runner.py:202–207) | INHERITED |
| Data order determinism | DataLoader shuffle (unpinned) | FingerprintedRandomSampler + per-epoch order-hash contract (evaluator_common.py:685–750; anonymizer_runner.py:240–245, 574–579) | LOCAL ADDITION |
| Checkpoint selection | best val loss (generator) | method-neutral min `val_selection_total` = ac_bce + own-verifier privacy_term (anonymizer_runner.py:617–622) | LOCAL ADDITION (confound risk, see §4.3) |
| Fail-closed governance | none | SHA gates on all frozen artifacts, TEST firewall, NaN/Inf fail-fast, scientific locks (evaluator_common.py throughout) | LOCAL ADDITION |

**Conclusion:** the current pipeline preserves the upstream *game structure* (1:1 co-adaptation, stale fakes, anon/real critic objective, anon/anon attacker objective) while adding governance. Any mechanism probe must therefore attribute differences to the intended intervention, not to these inherited semantics. VERIFIED_FROM_SOURCE.

---

## 4. DIRECTION B V1 CAUSAL-VALIDITY VERDICT

**Verdict: INVALID as causal evidence; REFERENCE_ONLY as exploratory signal.**

### 4.1 Confound 1 — generator BatchNorm drift (VERIFIED_FROM_SOURCE)
`run_hardened_verifier.py:194` sets `self.generator.train()` for the whole epoch and never switches to eval inside the extra-step block. Lines 263–275 perform `k_extra` extra verifier updates, each preceded by `with torch.no_grad(): fakes_1_extra = self.anonymize_tensor(inputs1)` (line 264–265). `torch.no_grad()` does not stop BatchNorm running-statistic updates; the UNet contains `nn.BatchNorm2d` after every conv (UNet_PriCheXyNet.py:107,119). With `k_extra_verifier_steps=2` the generator's BN buffers receive **three** forward passes per batch instead of one. The docstring claim "generator is not touched by these extra steps" (lines 27–29) is false for buffer state.

### 4.2 Confound 2 — same-batch repetition (VERIFIED_FROM_SOURCE)
Extra steps reuse `inputs1`, `inputs2_snn`, `labels_id_cast` from the current batch (run_hardened_verifier.py:269–270). Three optimizer steps on one batch improves batch fitting, not distribution-level best response.

### 4.3 Confound 3 — self-referential checkpoint selection (VERIFIED_FROM_SOURCE)
Both certified and v1 runners select the generator by minimum `val_selection_total = ac_bce + privacy_term(own evolving verifier)` (anonymizer_runner.py:617–622; run_hardened_verifier.py:376–380). Once verifier dynamics change (k=3), the scale/difficulty of the selector changes with the method; rankings across methods are not comparable. This also means B_dev's selected epoch 13 is partly an artifact of its own verifier's trajectory.

### 4.4 Confound 4 — dropped guards (VERIFIED_FROM_SOURCE by comparison)
v1 drops: expected-vs-observed order-hash enforcement (compare anonymizer_runner.py:575–579 with run_hardened_verifier.py:355), gradient/post-step finiteness assertions, C4 gradient diagnostics, strict manifest fields. Acceptable for a pilot; unacceptable for confirmation.

### 4.5 Consequences
- A favorable k3 outcome cannot be attributed to stronger verifier pressure alone (could be BN-dynamics change).
- An unfavorable outcome cannot falsify multi-step critic training (intervention confounded).
- The running PID 1875303 process is untouched; when it finishes, label outputs `k3 + generator-BN-drift exploratory pilot`, preserve artifacts, and treat any AUC as hypothesis-generating only.

### 4.6 Review-branch v2 status (VERIFIED_FROM_SOURCE at `a38ab368`)
`run_hardened_verifier_v2.py` fixes Confounds 1–2 structurally: `preserved_eval_forward` forces per-submodule `.eval()` under `no_grad()` and restores exact mode topology (protocol_v2/state_invariants.py, `preserved_eval_forward`); `GeneratorStateGuard.__exit__` verifies mode/buffer/parameter-version/full-canonical-hash invariants on normal and exceptional exit; `batch_policy ∈ {same_batch, fresh_batch}` is explicit. However:
- `HardenedVerifierRunnerV2.run()` raises "design-only stub" (v2_runner.py:380–381);
- `_next_fresh_critic_batch()` raises NotImplementedError (374–378);
- `main()` refuses to start without a human-approved `--execution-manifest` that does not exist (127–141, 392–400).
**The v2 runner is intentionally NOT executable today.** It is supporting audit evidence, not canonical method source.

---

## 5. P0 — PROTOCOL-EQUIVALENCE BRIDGE: HYPOTHESES AND EXACT PROTOCOL

### 5.0 What P0 must decide
Whether the user-reported upstream-side ≈0.60 effective AUC and the current B_dev mean 0.8237 diverge because of (i) protocol/cohort/evaluator differences, (ii) generator/checkpoint regression, or (iii) mixed causes. Until P0 reports, the two numbers must not be compared. VERIFIED_FROM_EXISTING_DATA (both numbers exist in authorized reports; comparability unestablished).

### 5.0.1 Existing evidence bearing on P0 (audit input, not a substitute)
- R1-FINAL (FINAL_10SEED_PRICHEXY_REPRODUCTION.md): upstream code path, released generator SHA `4d82dcdd…`, TEST pairs (SHA `87e52830…`, n=5000), fresh ImageNet ResNet-50 Siamese attackers trained anon/anon (batch 32, Adam 1e-4, ≤100 epochs, patience 5), evaluated once on TEST anon/real → mean 0.6080, SD 0.0435, n=10. VERIFIED_FROM_EXISTING_DATA.
- PRICHEXY_PAPER_REPRODUCTION.md:33: "the earlier STEP-8R validation-pair audit measured a retrained attacker at ~0.818 AUC with this released generator". If accurate, the SAME generator family yields ≈0.61 on TEST pairs and ≈0.82 on VAL pairs — i.e., most of the 0.60↔0.8237 gap would be fold/pair-file/protocol dependence, not generator quality. SUPPORTED_BY_EXISTING_DATA but the STEP-8R artifact's provenance (exact harness, orientation rule, seeds) is UNRESOLVED; P0 exists precisely to settle this under lock.
- Checkpoint identity hazard (verified locally): `networks/pretrained_generator_prichexy_net.pth` (the M2 initial generator, SHA `10122689…`, evaluator_common.py:56–57) and `reproduction/workspace_prichexy/networks/generator_lowest_total_loss_mu_0.01.pth` (the R1-FINAL evaluation target, SHA `4d82dcdd…`) are **byte-distinct files**. Both are described in project documents as the released/canonical upstream generator. Which one defines Anchor U — or whether both must be screened — must be resolved before P0 runs. UNRESOLVED (P0 blocker B-1).

### 5.1 Anchors
- **Anchor U:** the released/canonical upstream PriCheXy generator at privacy setting μ=0.01, identity resolved per B-1 (if unresolved after human review, evaluate BOTH candidate SHAs as U₁/U₂ — adds one attacker-seed set ≈8.7 GPU-h for the bridge).
- **Anchor B:** `research_runs/M2_S1/B_dev/seed_42/generator_best_method_neutral.pth`, selected epoch 13 of the certified 250-epoch B_dev run, SHA256 `18381d92c64bb3d646b62d5fb9d0ed8c208cf2cb3154f8aa1dac4b1baff610cd` (M2_S1_C4_RESULT.md §1, §4). VERIFIED_FROM_EXISTING_DATA.

Both anchors evaluated under ONE locked harness (below). Anchor B historical values (0.8237 etc.) may be reused **only if** semantic+execution equivalence of the prior harness to the new locked harness is proven field-by-field; absent that proof, both anchors are re-evaluated with paired seeds. Default assumption: re-evaluate both (cost already budgeted).

### 5.2 Protocol lock (all fields mandatory, frozen in a written manifest BEFORE any run)
1. Generator checkpoint SHA-256 (Anchor U resolved; Anchor B above).
2. Flow/deformation operator: legacy grid = identity − μ·grid, μ=0.01, GaussianSmoothing kernel 9 σ 2 applied to whole grid, `grid_sample(border, align_corners=True)` (evaluator_common.py:38–43,577–587).
3. Input resizing: Resize(256,256) bilinear at load; ToTensor grayscale.
4. Channel handling: generator operates 1-ch; SNN input = expand to 3-ch then ImageNet normalize, no resize (snn_preprocess, evaluator_common.py:590–594).
5. Pair files: TRAIN `image_pairs_training_10000.txt` (SHA `3c535eed…`), VAL `image_pairs_validation_2000.txt` (SHA `9e33a081…`, order SHA `2b34b491…`). For Anchor-U equivalence testing the primary metric uses the **same VAL pair file** as Anchor B; a secondary TEST-fold measurement is permitted ONLY as a labeled descriptive cross-check, never merged into the primary delta.
6. Pair orientation: row format `image1 image2 label`; anonymize image1 only; label semantics y_true=1 ⇔ same patient (Dataset.py:103; utils.py:703).
7. Attacker architecture & init: `SiameseNetwork` — ResNet-50 `pretrained=True` ImageNet, fc→128, abs-difference head (SiameseNetwork.py:17–23); fresh random head init per seed; NO reuse of the pretrained verification model weights (matches R1-FINAL §8 and DevAttacker default).
8. Pretrained-weight provenance: record torchvision ResNet-50 weight bundle identifier/hash in the manifest.
9. Attacker optimizer: Adam lr 1e-4, batch size 32 (config contract, dev_attacker.py:98–99; R1-FINAL §9 identical).
10. Data-order seed: `utils.seed_all(attacker_seed)` before net init/shuffle/optimizer (dev_attacker.py:119–120).
11. Weight-init seed: same seed list, paired across anchors.
12. Early stopping: patience 5 on selection BCE; max 100 epochs.
13. Checkpoint selector: min anon/anon VAL BCE (dev_attacker.py:189–213).
14. Train geometry: anon(x1)/anon(x2) with frozen generator in eval mode.
15. Evaluation geometry: anon(x1)/real(x2), generator eval, no_grad.
16. Score direction: sigmoid probability, higher = more-likely-match; **orientation fixed a priori; AUC < 0.5 is reported raw and flagged, NEVER flipped post hoc.**
17. AUC implementation: `sklearn.metrics.roc_auc_score` on pooled 2000 pairs (eval_reid_val.py:76).
18. Raw AUC is the primary quantity. Any "effective AUC" transformation (e.g., symmetrization max(AUC,1−AUC)) is forbidden in the primary analysis; if a descriptive effective value is reported it MUST be accompanied by the raw value and use the pre-fixed orientation.
19. Output schema: retain per-pair `y_true`,`y_score` with pair IDs; store generator/attacker/config/pair-file SHAs per run (as pilot_summary_final.json already does).
20. Harness: reuse `evaluate_reid_val()` and the DevAttacker-equivalent trainer unchanged (as `run_stage_a_pilot.py` did); any deviation voids equivalence.

### 5.3 Paired attacker-seed design
Stages:
- **P0 screen:** 5 paired attacker seeds (propose 43–47), diagnostic only, both anchors.
- **P0 full bridge:** 26 paired attacker seeds (42–67, matching Stage A numbering), conditional fixed-generator comparison.

Quantity:
```
Δ_bridge,s = AUC_U,s − AUC_B,s        (per paired attacker seed s)
```
Lower Δ_bridge favors the upstream generator on privacy (more leakage removed).

Analysis (conditional on fixed generators — explicitly NOT generator-level method replication):
- Report per-arm mean/SD, paired mean/SD/range of Δ_bridge, Pearson ρ between arms (Stage A precedent: ρ≈0.17 — CRN cannot be assumed; S2_CONFIRMATORY_DESIGN_PROPOSAL.md §8).
- Primary inference: paired t-based one-sided 95% bound against the predeclared margin; sensitivity: sign-flip permutation over seeds; bootstrap over seeds (NOT patient bootstrap — see §11).
- All statements conditioned as: "for these two fixed generator checkpoints under the locked protocol".

### 5.4 Predeclared practical-equivalence threshold
**δ_equiv = 0.03 AUC**, frozen before execution. Justification: it is the project's existing predeclared privacy margin (S1 gate Δpriv ≤ +0.03, M2_S1_C4_RESULT.md §5) and approximates the observed attacker-seed SD scale (0.031); using an existing predeclared constant avoids post hoc threshold choice. Do not adjust after seeing results.

### 5.5 Interpretation gate (decision table)

| Outcome | Operational definition (predeclared) | Interpretation | Consequences |
|---|---|---|---|
| **P0-A** | \|Δ_bridge\| point estimate and 95% CI within ±δ_equiv (equivalence), OR upstream anchor AUC statistically compatible with the B_dev range (CI overlapping 0.79–0.85 band) | The ≈0.60 result was protocol/cohort/evaluator dependent; project resets its comparable baseline; "below 0.58" ceases to be an unqualified target; all future baselines quote locked-harness numbers | Re-baseline; proceed to P1 with corrected expectations |
| **P0-B** | Δ_bridge > +δ_equiv (upstream anchor meaningfully LOWER AUC than B_dev under identical protocol) | Current method/checkpoint likely regressed in privacy relative to its own starting point | Investigate, before ANY new direction work: batch-size/operator/preprocessing deltas vs upstream, checkpoint-selection epoch 13, training semantics (stale fakes, BN), initial-generator identity (10122689 vs 4d82dcdd lineage) |
| **P0-C** | CI spans both regions, or any lock field fails / harness equivalence unprovable | Unstable or unprovable equivalence | NO baseline-improvement claim authorized anywhere; provenance repair first; no P1 |

Additional predeclared sub-rule: if Anchor U resolves to two candidates (B-1) and they disagree beyond δ_equiv, P0 returns outcome P0-C until checkpoint lineage is documented.

### 5.6 P0 cost audit (planning estimates, anchored to measurement)
Measured basis: 50 completed attacker trajectories averaged 1209.5 s = 0.336 GPU-h (§2.6). The audited estimates are therefore **measurement-anchored planning figures, not runtime guarantees**:
- One attacker trajectory ≈ 0.336 GPU-h ✓ matches measurement.
- 5-seed screen: 5×0.336 = 1.68 h (one anchor) / 3.36 h (both anchors) ✓ ≈1.7/3.4.
- 26-seed bridge: 26×0.336 = 8.74 h (one anchor) / 17.5 h (both anchors) ✓.
Caveats: VRAM contention with the active k3 run forbids concurrent execution; sequential-only execution is the precedent (S2 proposal §7 resolution). Runtime scales mildly with early-stopping behavior; ±20% slack recommended.

---

## 6. P1 — FROZEN-GENERATOR ADVERSARY MECHANISM PROBE: FACTORIAL DESIGN

P1 is contingent on: (i) P0 returning an interpretable outcome (A or B), and (ii) separate explicit human approval. Not executable now.

### 6.1 Generator anchors (correct names mandatory)
- **Anchor 1:** the available initial M2 generator checkpoint, before M2 adversarial fine-tuning — `networks/pretrained_generator_prichexy_net.pth`, SHA `10122689…` (evaluator_common.py:56–57). It is a pretrained PriCheXy-Net flow-field model; calling it "random"/"untrained"/"epoch-0" is prohibited.
- **Anchor 2:** the selected B_dev generator checkpoint, selected at epoch 13 of the 250-epoch B_dev seed-42 run, SHA `18381d92…`.
These are two available development anchors, NOT a temporal trajectory; no smooth-evolution claim is inferable. VERIFIED_FROM_EXISTING_DATA.

### 6.2 Fresh-critic trajectories
Design: **2 anchors × 2 train geometries × 3 fresh critic seeds = 12 trajectories.**
- G_AA: critic trained on anon(x1)/anon(x2) — evaluation-attacker geometry.
- G_AR: critic trained on anon(x1)/real(x2) — live-verifier geometry (upstream inherited).
Per cell: train ONE critic trajectory; save a predeclared short-budget checkpoint; continue the SAME trajectory to the convergence/early-stop checkpoint. Short-budget and converged evaluations share the trajectory — no doubling of runs.

Convergence rule (predeclared, matching the scientific attacker contract): Adam 1e-4, batch 32, ≤100 epochs, patience 5, checkpoint = min inner-VAL BCE **on the critic's own training geometry**; every saved critic is then evaluated on BOTH geometries (train-geometry × eval-geometry matrix, 4 cells per anchor).

Short budget (predeclared NOW, derived from existing authorized data, not post hoc): **5 attacker-loader epochs** (= ceil(10000/32)=313 updates/epoch × 5 ≈ 1565 updates). Basis: median best_epoch ≈ 5 across the 25 Stage-A B_dev pilot attacker manifests (best epochs 2–7 recorded in pilot_summary_final.json). Rationale: mimics the optimization exposure at which the scientific attacker typically peaks.

Live-exposure reference number (context, not a budget): the certified live verifier received 625 updates/epoch (10000 pairs ÷ batch 16) ⇒ ≈8125 updates by B_dev's selection epoch 13. The gap between 1565 and 8125 is itself part of the mechanism question and must be reported.

### 6.3 Live critics — availability audit
**No paired live-critic checkpoint exists for either anchor.** The certified runner saves only `generator.state_dict()` at the best epoch (anonymizer_runner.py:622) and writes resumable checkpoints solely in unit-test mode (625–626); `research_runs/M2_S1/B_dev/seed_42/` contains no verifier artifact (directory listing verified). Therefore:
- Per protocol §6.3, live-critic cells are reported **UNAVAILABLE**, not reconstructed.
- The only legitimately available live-related reference is the frozen critic-initialization state `pretrained_verification_model.pth` (SHA `331efaed…`), evaluated descriptively as "critic-at-init", clearly separated from replicated fresh-critic inference.
- If humans want a true H1 live reference, a separately authorized micro-run must capture verifier snapshots at predeclared epochs of a fresh 1:1 training run; this is out of scope today.

### 6.4 Primary P1 hypotheses (hypothesis-first format)

**H1 — Best-response gap.**
- Causal hypothesis: the live training critic at a frozen generator is materially weaker than a converged fresh best-response critic against the same frozen generator; generator training under the weaker critic leaves exploitable identity leakage.
- Measurable prediction: G_BR = AUC(fresh, converged; eval geometry matching the scientific attacker, i.e., critic evaluated anon/real after anon/anon training) − AUC(live reference) > δ_H1 consistently across anchors and seeds.
- Null/falsifier: G_BR ≈ 0 or inconsistent sign across seeds ⇒ weak-live-critic narrative loses dominant-mechanism status (H5 activates).
- Minimum evidence: ≥2 of 3 seeds positive beyond δ_H1 on BOTH anchors; δ_H1 predeclared = 0.03 (same rationale as §5.4).
- Statistical unit: fresh-critic seed within anchor (descriptive replication, not generator-level inference).
- Decision enabled: supports ranking restart/historical-critic interventions (§7 row 1).
- Cost: within §6.6 budget plus any authorized live-snapshot capture.
- Remaining limitation even if positive: shows the gap exists at two snapshots only; does not prove closing it lowers final-generator attack AUC; no trajectory coverage.
- **Status caveat:** with live cells UNAVAILABLE (§6.3), H1 as written is partially BLOCKED; the executable surrogate is G_BR^init = AUC(fresh,converged) − AUC(critic-at-init), which bounds the effect of critic optimization alone but is NOT the live-critic gap. Human approval decides whether to authorize live snapshots or accept the surrogate.

**H2 — Geometry mismatch.**
- Hypothesis: critic strength and ranking depend on train geometry (AA vs AR); part/all of any gap stems from the live critic optimizing the wrong learning problem.
- Prediction: the 2×2 train×eval matrix is non-additive — specifically AA-trained critics evaluated on AA exceed AR-trained critics evaluated on AA by > δ_H1, and geometry main effect exceeds seed variance.
- Falsifier: near-zero interaction/main effect ⇒ increasing k under either geometry is equivalent w.r.t. geometry.
- Evidence/statistics: 2×2 ANOVA-style decomposition across 3 seeds per cell (screening-grade); unit = critic seed nested in anchor.
- Decisions: geometry mismatch dominates ⇒ geometry-aligned objective ranks first (§7 row 2).
- Limitations: two geometries only; pair-file overlap between train and eval persists.

**H3 — Budget effect.**
- Hypothesis: converged checkpoints beat short-budget checkpoints from the SAME trajectory (isolating optimization budget from initialization noise).
- Prediction: mean AUC(converged) − AUC(short) > 0 across ≥8 of 12 trajectories.
- Falsifier: short≈converged ⇒ critics saturate fast; the live critic's weakness (if any) is not budget-limited.
- Unit: trajectory (paired within-trajectory difference).
- Decisions: supports corrected-k/TTUR framing ("more useful critic steps"); saturation supports rejection of pure-budget explanations.

**H4 — Selector confounding.**
- Hypothesis: generator checkpoint ranking under the arm's own live verifier (val_selection_total) differs from ranking under a fixed external selector.
- Executable diagnosis WITHOUT a neutral selector artifact: at each anchor, compare (a) the arm's own selection statistic, (b) fresh-critic AUCs from this factorial, and (c) — if separately authorized — one shared fixed external critic applied identically to both anchors. Under today's authorization, only (a)+(b) are runnable; the formal H4 test requires the external selector and is marked BLOCKED pending that design's approval. Design requirement: the external selector must be trained on TRAIN-derived pairs disjoint from the privacy-evaluation pairs; it must never be a candidate method's own critic.
- Decision enabled: whether future ablations need a predeclared external selector before any causal arm comparison (they do, regardless — §8).

**H5 — Alternative mechanism.**
If H1-surrogate and H2/H3 jointly fail to support weak-critic dominance, the ranked alternatives are: geometry mismatch (covered by H2), method-dependent selection (H4), moving C4 feature teacher (co-evolving DenseNet critic — inspectable from existing C4 gradient diagnostics), flow-field capacity limits, preprocessing differences, attacker-family limitation, protocol mismatch (P0 outcome). The weak-critic narrative must be dropped on falsifying evidence, not preserved.

### 6.5 P1 measurements and epistemic role
| Measurement | Definition | Role |
|---|---|---|
| Raw ROC AUC | sklearn, pre-fixed orientation, per eval geometry | PRIMARY endpoint |
| Log loss/BCE | inner-VAL BCE on training geometry | Selection + diagnostic (not causal) |
| Calibration (ECE/Brier) | reliability of match probabilities | Diagnostic |
| Score distributions | per-cell histograms/KS stats | Diagnostic (leakage character) |
| Convergence curve + early-stop epoch | per trajectory | Diagnostic (budget attribution) |
| Live→best-response gap (G_BR, G_BR^init) | §6.4 | Causal-supporting for H1 |
| Input gradient norm ′∂score/∂anon-image′ | L2 at VAL batches | Diagnostic (attack surface strength) |
| Critic prediction entropy | on score distribution | Diagnostic |
| Critic seed variance | SD across 3 seeds per cell | Uncertainty quantification |
| Geometry transfer matrix | 2×2 per anchor | Causal-supporting for H2 |
| BN/generator-state invariants | buffer-hash + mode-topology snapshot around every frozen-generator forward (pattern of review-branch `state_invariants.preserved_eval_forward`/`GeneratorStateGuard`) | Validity gate (must PASS for any cell to count) |

Only comparisons isolating one factor under fixed everything else (within-trajectory budget contrasts; geometry contrasts at matched budget/seeds) carry causal weight; AUC levels, calibration, gradients are diagnostics.

### 6.6 P1 cost audit (planning estimate, tied to measured evidence)
Observed attacker-trajectory runtime distribution: 913–1474 s (n=50, mean 1209.5 s) under the same architecture/loader family. 12 fresh-critic trajectories at the observed budget distribution ⇒ expected ≈ 12 × 0.336 ≈ **4.0 h**; plausible upper ≈ 12 × 1474 s ≈ 4.9 h if all run long; the quoted 4.2–10.5 h range implies up to ~31 min/trajectory average, reachable only if early stopping behaves materially differently (e.g., G_AR slower convergence) or schedules lengthen. The estimate correctly does NOT double-count short/converged checkpoints (same trajectory). Label: planning estimate; require a 1-trajectory timing smoke before final authorization. Evaluation passes (each saved checkpoint scored on 2000 VAL pairs ×2 geometries) add negligible GPU time (<0.05 h total, forward-only).

### 6.7 Post-P1 decision table (predeclared; do not rank methods before applying it)

| P1 result | Next method considered |
|---|---|
| Large reproducible fresh-converged critic gap (H1-surrogate + H3) | Periodic best-response restart or historical critic bank |
| Geometry mismatch dominates (H2) | Geometry-aligned or mixed-geometry adversarial objective |
| Additional clean critic steps close gap without instability (H3 + stable curves) | Corrected k:1 or TTUR |
| Higher pressure causes instability (loss divergence, grad-norm blowup) | Spectral normalization or R1 as supporting stabilizers |
| Different fresh critics expose complementary leakage (low inter-seed score correlation, union of errors) | Small independent critic ensemble |
| No meaningful best-response gap | Reject weak-critic hypothesis; investigate selector, teacher, flow capacity, protocol |

---

## 7. CLEAN FUTURE CAUSAL ABLATION (describe-only)

Precondition: a fixed external generator selector (H4 design) approved BEFORE any arm runs. Each arm changes exactly ONE thing vs B0:

| Arm | Single causal change vs B0 |
|---|---|
| **B0** | Certified 1:1 stale-fake control (exact certified semantics, incl. BN behavior of combined step) |
| **B1** | 1:1 with post-generator-update fresh fake only (fake regenerated after optimizer_g.step(), generator protected eval/no_grad — v2 pattern) |
| **B2** | 1:1 with TTUR: verifier LR changed, nothing else (optionally a second sub-arm with compensated verifier LR to separate step-count from effective step size) |
| **B3** | k=3 with fresh independent critic batches AND complete generator-state protection (v2 `fresh_batch` + `GeneratorStateGuard`) |
| **A** | Periodic fresh verifier restart at predeclared epochs |
| **A+** | A plus small historical critic bank (pool size predeclared) |

Forbidden inside any single "causal" comparison: combining update ratio with fake freshness, batch reuse, critic init change, regularization, or selection-rule change. Screening funnel: mechanism diagnostics → one predeclared primary candidate → powered analysis (§9).

---

## 8. STATISTICAL HIERARCHY AND POWER STRATEGY

Three uncertainty levels, explicitly separated:
1. **Attacker-optimization seed** — addressed by Stage A (n=26, conditional on 2 fixed generators + fixed cohort). SD≈0.03–0.04.
2. **Generator-training seed** — NEVER measured in this project (S2 proposal §8 "still completely unmeasured").
3. **Patient/cohort sampling** — never addressed; single fixed pair split since project start.

- **Screening (fixed generator):** can rank mechanisms conditionally; cannot establish population-level superiority.
- **Generator-level pilot (future):** 3–4 paired generator seeds × 3–5 attacker seeds per checkpoint; purpose = variance estimation of Δ_g only; explicitly non-confirmatory.
- **Confirmatory:** experimental unit = generator training seed. Required n from paired generator-level SD σ_g: n_g = σ_g²(z_{1−α}+z_{1−β})²/Δ². Sensitivity (α=0.05 one-sided, target −0.03): σ_g=0.02→3–4 seeds; 0.03→7–9; 0.04→11–16; 0.05→18–24. Five seeds is a floor, not adequate power under substantial variance; do not assume sufficiency.
- **Primary endpoint:** Δ = AUC_candidate − AUC_B_dev (negative = privacy improvement), generator-level. Publishable superiority requires: one-sided 95% CI upper bound < 0 AND point estimate ≤ predeclared −0.03 AND utility gate preserved AND robustness checks. Non-inferiority alone is insufficient.

---

## 9. DATA-ROLE AND PATIENT-BOOTSTRAP LIMITATIONS

- **VAL role saturation (VERIFIED_FROM_SOURCE/EXISTING_DATA):** the same 2000 VAL pairs (SHA `9e33a081…`) have been used for attacker early stopping (dev_attacker.validate_selection), generator checkpoint selection (val_selection_total includes own-verifier privacy term), the S1 privacy gate, the 26-seed pilot, and strategy development. CURRENT_VAL_CONFIRMATORY_STATUS: NOT_LOCKED. No analysis on current VAL may be labeled confirmatory.
- Path to a publishable confirmation: prospective patient-disjoint role design — five logical roles (generator_train, generator_select, attacker_train, attacker_select, locked_confirm) with hard-forbidden overlaps exactly as encoded in review-branch `role_manifest.py` ROLES/_HARD_FORBIDDEN — derived from TRAIN data or a newly approved resource, with a freshly generated paired B_dev baseline under the same redesign; 0.8237 remains a historical conditional reference, not the null constant.
- **Patient bootstrap: UNVALIDATED.** The current patient-graph bootstrap has no validated data-generating-process specification; its confidence intervals must not be reported as valid. A valid future approach requires (a) prospectively specified patient-level pair-sampling DGP, (b) pair regeneration inside patient resamples, or (c) an explicitly conditional inference target that disclaims cohort-general sampling uncertainty. Prior simulation conclusions are not reusable.

---

## 10. SEGMENTATION READINESS GATE

SEGMENTATION_EXECUTION_STATUS: **BLOCKED** — `research_agent/M1_1_SEGMENTATION_PROVENANCE.md:4` determines `SEGMENTATION_STILL_BLOCKED` (no certified segmentation evaluator; split/checkpoint/log provenance recovered but incomplete certification chain). Conceptual requirements to record now (design, no runs):
1. **Anonymized-image segmentation utility:** the flow field moves anatomy; the reference mask MUST be transformed by the exact same sampling grid (`F.grid_sample` on the mask with the identical grids), nearest-neighbor interpolation for discrete labels. Valid pair: anonymized image vs same-grid warped mask. Comparing anonymized images to ORIGINAL masks is invalid.
2. **Frozen raw-model robustness:** a segmenter trained on originals may be evaluated on anonymized images, but predictions must still be compared in the anonymized coordinate system (same warped-mask rule).
3. **Spatial fidelity:** warped-vs-original mask agreement measures deformation fidelity, not segmentation accuracy; label it as such.
4. **Shape leakage (PROPOSED_HYPOTHESIS, clearly labeled):** preserving lung/heart/rib/thoracic shape may itself preserve identity; a future mask/shape-leakage probe (e.g., mask-conditioned re-ID attacker) should test whether shape preservation transmits identity cues. Preservation of Dice does NOT automatically improve the privacy–utility trade-off.

---

## 11. LITERATURE COLLISION AND DEFENSIBLE NOVELTY

Positioning (primary sources; SUPPORTED_BY_LITERATURE):
- **PriCheXy-Net** (arxiv.org/html/2209.11531v2): flow-field CXR anonymization vs Siamese re-ID; paper reports ≈0.577±0.040 TEST AUC at μ=0.01 — the origin of the ≈0.60 question; our R1-FINAL reproduced 0.6080±0.0435 (FINAL_10SEED report).
- Chest X-ray re-identification threat literature (nature.com/articles/s41598-022-19045-3) and recent medical-privacy surveys/approaches (nature.com/articles/s41746-026-02440-9; proceedings.mlr.press/v250/heinrich24a.html; arxiv.org/html/2503.08173v1; arxiv.org/html/2507.21703v1; proceedings.mlr.press/v317/akhter26a.html): Privacy-Net-style face/CXR de-identification, mixup-based privacy, implicit neural obfuscation, All-in-One medical Re-ID benchmarks, DCM-DeID, PrivDiff-Net — collectively establish that adaptive/multi-family attacker evaluation and generator-seed statistics are increasingly expected; none removes the obligation to measure them.
- Adversarial-training mechanics: TTUR (arxiv.org/abs/1706.08500), zero-centered/R1 gradient penalty (arxiv.org/abs/1801.04406), spectral normalization (arxiv.org/abs/1802.05957), f-divergence perspective (arxiv.org/abs/1611.01673), unrolled GAN (arxiv.org/abs/1611.02163), fictitious-play/history-based opponents (arxiv.org/abs/1803.08647). These motivate — but do not validate for CXR privacy — restart/TTUR/SN/ensemble choices; novelty cannot be claimed from mechanism names alone.

Not sufficient for a Q2-level contribution (explicit list, endorsed): one seed with lower AUC; non-inferiority only; live-critic loss reduction; comparison only against historical 0.8237; a single ResNet50-Siamese attacker family; reporting sub-0.5 AUC without a pre-fixed orientation rule; Dice against coordinate-inconsistent masks; repeated tuning on the same VAL pairs; novelty claimed from a new loss name.

Required future held-out threat models: ≥1 independently initialized CNN attacker (same arch, fresh init — partially covered by attacker seeds), ≥1 architecturally different attacker (e.g., Vision-Transformer/embedding-loss Siamese), simple pixel/similarity baselines (SSIM/orb-feature matching, raw-pixel kNN), a representation/foundation-model retrieval attacker where feasible, and near-duplicate/acquisition-cue/interval sensitivity analysis of the pair construction.

Defensible core contribution IF (and only if) the full ladder completes: generator-seed-level privacy superiority under a locked protocol with adaptive + transferred + baseline attacker families, preserved classification utility, and spatially correct segmentation utility reporting — a combination not demonstrated by the cited works on NIH CXR, though each ingredient exists elsewhere; positioning must claim rigor and evaluation validity, not a new mechanism name.

---

## 12. RANKED RISKS AND INVALID ASSUMPTIONS

1. **R1 — Treating 0.60 vs 0.8237 as comparable** before P0. Invalid assumption; blocks all baseline language.
2. **R2 — Reading the running k3 run causally** despite BN drift, batch reuse, self-referential selection (§4). Reference-only.
3. **R3 — Anchor U ambiguity** (two byte-distinct "released" generators). Must resolve before P0.
4. **R4 — Pseudoreplication**: quoting n=26 as power for method claims; generator-seed variance unmeasured.
5. **R5 — VAL saturation**: any further tuning on current VAL erodes even screening validity.
6. **R6 — Orientation laundering**: any post-hoc score flipping or "effective AUC" without pre-fixed rules invalidates privacy claims.
7. **R7 — Live-critic fabrication temptation**: no live checkpoint exists; reconstructing one ad hoc would break the audit chain (§6.3).
8. **R8 — Segmentation claims from coordinate-inconsistent masks**; currently moot (BLOCKED).
9. **R9 — Bootstrap CIs presented as valid** (unvalidated DGP).
10. **R10 — Branch-state divergence** (strategy-review commit not an ancestor of method tip): provenance hygiene issue requiring human curation.

---

## 13. HUMAN APPROVAL CHECKLIST

- [ ] Approve P0 protocol lock (§5.2 fields 1–20) and δ_equiv = 0.03 (§5.4).
- [ ] Resolve B-1: designate canonical Anchor U (or approve dual-anchor screen, +≈8.7 GPU-h).
- [ ] Decide reuse-vs-rerun of B_dev historical 26-seed values (default: rerun both anchors).
- [ ] Authorize P0 screen (5 seeds) then bridge (26 seeds) — sequential, after k3 run completes.
- [ ] Decide H1 live-reference route: authorize live-verifier snapshot capture, or accept critic-at-init surrogate.
- [ ] Approve P1 factorial, short budget = 5 attacker-loader epochs, δ_H1 = 0.03.
- [ ] Approve external-selector design before any future causal ablation arm.
- [ ] Curate branch divergence (a3513e5 vs 34bca1f lineage).
- [ ] Confirm segmentation remains BLOCKED until certified evaluator exists.
- [ ] Explicitly forbid post-hoc score flipping and post-hoc threshold edits.

---

## 14. MACHINE-READABLE FINAL VERDICT BLOCK

```text
P0_PROTOCOL_SPECIFICATION: PASS
P0_EXECUTION_AUTHORIZATION: NONE
P1_PROTOCOL_SPECIFICATION: PASS
P1_EXECUTION_AUTHORIZATION: NONE
WEAK_CRITIC_HYPOTHESIS_STATUS: UNTESTED
DIRECTION_B_V1_CAUSAL_EVIDENCE: INVALID
CURRENT_VAL_CONFIRMATORY_STATUS: NOT_LOCKED
PATIENT_GRAPH_BOOTSTRAP_STATUS: UNVALIDATED
SEGMENTATION_EXECUTION_STATUS: BLOCKED
GENERATOR_TRAINING_AUTHORIZATION: NONE
GPU_AUTHORIZATION: NONE
FILES_CREATED: 1
EXISTING_FILES_MODIFIED: 0
ACTIVE_TRAINING_PROCESS_TOUCHED: NO
COMMIT_OR_PUSH: NONE
NEXT_REQUIRED_ACTION: EXTERNAL_HUMAN_REVIEW
```

*End of report. Execution of P0/P1 awaits explicit external human approval.*
