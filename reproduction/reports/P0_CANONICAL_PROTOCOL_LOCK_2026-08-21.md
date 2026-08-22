# P0 CANONICAL PROTOCOL LOCK — CLEAN-ROOM DESIGN

**Date:** 2026-08-21
**Status:** PROTOCOL DESIGN ONLY. This document authorizes NO implementation, NO checkpoint loading, NO model execution, NO testing, simulation, training, evaluation, GPU/CUDA use, NO commit, push, merge, or pull request.
**Independence:** this report is self-contained; it does not depend on any previous audit or erratum report.

---

## 1. EXECUTIVE VERDICT

This report locks a single canonical P0 protocol-equivalence bridge comparing the **released trained PriCheXy-Net endpoint at μ=0.01 (U_PUBLISHED)** against the **current selected B_dev generator (D_BDEV)** under one identical, authorized attacker-evaluation protocol reconstructed from source at base commit `a3513e5c3b3b5631838399fc14c9e708909fe923`. The protocol is READY FOR HUMAN RATIFICATION with ONE unresolved blocker (TRAIN/VAL patient-disjointness status) that must be resolved before any future execution. All statistical inference is conditional on two fixed generator checkpoints and one attacker architecture; P0 is not evidence of a new method. `P0_IMPLEMENTATION_AUTHORIZATION: NONE`.

---

## 2. CLEAN-ROOM PROVENANCE STATEMENT

This protocol was reconstructed exclusively from:
1. base commit `a3513e5c3b3b5631838399fc14c9e708909fe923` of `origin/research/method-restart` (verified via `git ls-remote` immediately before worktree creation);
2. the explicitly allowed current source files listed in §4;
3. public upstream files at locked upstream commit `29245d1f71571898d9527417df4ae3f63a8695f6`;
4. the locked checkpoint-role premises supplied by the human tasking.

The following are quarantined audit-history inputs and were NOT read, merged, cited, or used: branch `research/method-restart-p0p1-review`, commits `c2ee268d…` and `bf43d30c…`. VERIFIED_FROM_SOURCE (worktree isolation).

The user-reported value of approximately 0.60 is treated only as unverified historical motivation for asking the P0 question at all. It is NOT admissible evidence and does not appear in any decision rule below.

No prohibited frozen-evaluation artifact was opened, listed, searched, named, hashed, quoted, or referenced. No recursive repository listing or unrestricted search was performed.

---

## 3. SOURCE AND REPOSITORY IDENTITY

| Item | Value | Verification |
|---|---|---|
| Repository | `https://github.com/minhmoidz/Neups_workshop.git` | remote origin |
| Authorized source branch | `origin/research/method-restart` | `git ls-remote` = required base commit ✓ |
| Base commit | `a3513e5c3b3b5631838399fc14c9e708909fe923` | exact match ✓ |
| Local branch created | `review/p0-canonical-protocol-20260821` | did not pre-exist ✓ |
| Isolated worktree | `/tmp/p0-canonical-protocol-worktree` | did not pre-exist ✓ |
| Upstream lock | PriCheXy-Net `29245d1f71571898d9527417df4ae3f63a8695f6` | read-only `git show <commit>:<path>` |
| Active training process | Direction B v1 runner active (PID 1875303) throughout this task | recorded at start and end; untouched |

The existing worktree hosting the active training process was not switched or modified. VERIFIED_FROM_SOURCE

---

## 4. EVIDENCE ACTUALLY USED (allowed set only)

Read-only inspection of: `reproduction/reports/RESEARCH_DIRECTION_SUMMARY.md`; `reproduction/reports/S2_CONFIRMATORY_DESIGN_PROPOSAL.md`; `reproduction/s2_pilot/results/pilot_summary_final.json`; `research_agent/M2_S1_C4_RESULT.md` (the certified M2-S1 result file linked directly by RESEARCH_DIRECTION_SUMMARY.md §6); code files `research_agent/m2_dev/dev_attacker.py`, `research_agent/m2_dev/eval_reid_val.py`, `research_agent/m2_dev/evaluator_common.py`, `datasets/SiameseDataset.py`, `networks/SiameseNetwork.py`; upstream paths `README.md`, `agents/Agent.py`, `config_files/config_anonymization.json`, `config_files/config_retrainSNN.json`, `config_files/config_eval_classifier.json` at `29245d1`.

Freshly verified at the isolated worktree: SHA-256 of both authorized pair files (`sha256sum` of the two named files only; contents not opened):
- TRAIN: `image_pairs/image_pairs_training_10000.txt` = `3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268`
- VAL (selection/scoring): `image_pairs/image_pairs_validation_2000.txt` = `9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7`

These match the SHA-gated constants in `evaluator_common.py:121–125`. VERIFIED_FROM_SOURCE / VERIFIED_BY_HASH

---

## 5. THREE-ROLE CHECKPOINT TABLE (fixed premises; none loaded or executed)

| Role | Artifact | SHA-256 | Role basis (public upstream source) |
|---|---|---|---|
| **I_M2** — initial M2 generator | `networks/pretrained_generator_prichexy_net.pth` | `101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb` | Loaded BEFORE adversarial training begins (`agents/Agent.py:65`); an initialization checkpoint, NOT the released endpoint |
| **U_PUBLISHED** — released trained PriCheXy endpoint at μ=0.01 | `networks/generator_lowest_total_loss_mu_0.01.pth` (local materialized copy: `reproduction/workspace_prichexy/networks/generator_lowest_total_loss_mu_0.01.pth`) | `4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153` | Designated perturbation model by upstream `README.md` ("perturbation_model_file", μ=0.01), `config_files/config_retrainSNN.json`, `config_files/config_eval_classifier.json` |
| **D_BDEV** — selected current B_dev checkpoint | `research_runs/M2_S1/B_dev/seed_42/generator_best_method_neutral.pth` | `18381d92c64bb3d646b62d5fb9d0ed8c208cf2cb3154f8aa1dac4b1baff610cd` | Selected epoch 13 of the certified 250-epoch B_dev run (`research_agent/M2_S1_C4_RESULT.md` §1, §4) |

---

## 6. CANONICAL P0 COMPARISON

Primary comparison exactly:

```text
Anchor U = U_PUBLISHED   (SHA 4d82dcdd…)
Anchor D = D_BDEV        (SHA 18381d92…)
```

Paired per-seed difference:

```text
Δ_s = AUC_U_PUBLISHED,s − AUC_D_BDEV,s
Δ_s < 0 → U_PUBLISHED has lower re-identification AUC (better privacy)
Δ_s > 0 → D_BDEV has lower re-identification AUC (better privacy)
```

Substitution rules: I_M2 must NOT be substituted for U_PUBLISHED. An I_M2-vs-D_BDEV comparison belongs to a later internal P1 diagnostic and is outside P0.

P0 is a protocol-equivalence bridge and baseline-selection experiment only. It is not itself evidence of a new publishable method.

---

## 7. HYPOTHESIS AND FALSIFIER TABLE

Provisional smallest effect of interest: δ = 0.03 AUC, labeled everywhere **PROVISIONAL_SEOI_PENDING_HUMAN_RATIFICATION**.

| Outcome | Observable quantity | Exact decision rule | Falsifier | Interpretation | What it CANNOT establish |
|---|---|---|---|---|---|
| **P0_EQ** practical equivalence | mean paired Δ over n=26 seeds | TOST α=0.05: complete 90% CI of mean Δ inside (−0.03, +0.03); two-sided 95% CI reported descriptively | Either directional bound of the 90% CI touches/exits (−0.03, +0.03) | The two fixed checkpoints behave similarly under this exact current attacker protocol; historical ≈0.60 context is not transportable into this protocol without further work | Population-level protection; robustness to other attackers/generators; non-significance alone is NEVER equivalence evidence |
| **P0_U** released endpoint materially better | same | one-sided 95% UPPER confidence bound for mean Δ < −0.03 | Upper bound ≥ −0.03 | D_BDEV shows a conditional privacy regression relative to the released trained endpoint under identical protocol | That regression is caused by any specific mechanism (requires later diagnostic studies); generalization beyond these checkpoints |
| **P0_D** B_dev materially better | same | one-sided 95% LOWER confidence bound for mean Δ > +0.03 | Lower bound ≤ +0.03 | Current pipeline meaningfully improves privacy over its published starting point, conditionally | Publishable superiority; method-level claims (single fixed generator pair) |
| **P0_INC** inconclusive | same | none of the above satisfied; or protocol/provenance failure | — | No baseline-selection decision is authorized from P0 | Everything above |

Mutually exclusive by construction; evaluated in order EQ → U → D → INC. Do not describe non-significance as equivalence.

---

## 8. FULL PROTOCOL-LOCK TABLE

Reconstructed from allowed code; uncertain fields are marked. All line references verified at base commit `a3513e5`.

| # | Field | Locked value | Source |
|---|---|---|---|
| 1 | P0 purpose | Protocol-equivalence bridge between released trained endpoint and current selected generator under one authorized attacker protocol | this report |
| 2 | Generator role | Anchor U = U_PUBLISHED; Anchor D = D_BDEV (roles per premises; no substitution) | §5–§6 |
| 3 | Generator path | `networks/generator_lowest_total_loss_mu_0.01.pth` (materialized copy path for local execution) vs `research_runs/M2_S1/B_dev/seed_42/generator_best_method_neutral.pth` | premises; M2_S1_C4_RESULT.md |
| 4 | Generator SHA-256 | `4d82dcdd…` / `18381d92…` — re-hash fail-closed before every trajectory | premises; evaluator_common pattern |
| 5 | Flow-field operator | grid = identity − μ·grid, μ=0.01; GaussianSmoothing kernel 9 σ=2 applied to whole grid; `F.grid_sample(border, align_corners=True)`; frozen generator `.eval()` | evaluator_common.py:38–43,568–587; dev_attacker.py:57–64 |
| 6 | Attacker-training pair file | `image_pairs/image_pairs_training_10000.txt` | evaluator_common.py:124; SiameseDataset.py:38 |
| 7 | Training pair-file SHA-256 | `3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268` | freshly hashed ✓; evaluator_common.py:125 |
| 8 | Training pair-order SHA-256 | Seed-dependent contract: deterministic permutation from `torch.Generator(attacker_seed)` via DataLoader shuffle; NO precomputed fixed hash exists for attacker loaders — actual per-epoch order hashes MUST be logged at runtime (field 31). Precomputed order hashes exist only for the seed-42 batch-16 anonymizer sampler and do not apply here | evaluator_common.py:617–642; honest gap, not silently filled |
| 9 | Attacker-selection pair file | `image_pairs/image_pairs_validation_2000.txt` | evaluator_common.py:121; SiameseDataset.py:40 |
| 10 | Selection pair-file SHA-256 | `9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7` | freshly hashed ✓; evaluator_common.py:122 |
| 11 | Selection pair-order SHA-256 | Sequential (no shuffle); semantic row-order SHA `2b34b491c01695d3a058b21791faf8e1bccb0e27a3cdbcba552b9102c6f34f4f` recorded uniformly across all authorized pilot records | pilot_summary_final.json (`validation_pair_order_sha256`) |
| 12 | TRAIN-vs-VAL patient disjointness | **UNRESOLVED — BLOCKER.** Not established anywhere in the allowed evidence. Must be verified (patient-ID overlap check between pair files) and documented before execution | open item |
| 13 | Image resize | Resize to 256×256 bilinear at load | evaluator_common.py:43; Dataset/SiameseDataset resize |
| 14 | Channel handling | Grayscale 1-ch load; SNN input expanded 1→3 channels | evaluator_common.py:590–594 |
| 15 | Intensity normalization | ImageNet mean/std `[0.485,0.456,0.406]/[0.229,0.224,0.225]`, applied after channel expand, no resize for SNN input | evaluator_common.py:590–594 |
| 16 | Siamese architecture | ResNet-50 backbone → fc 2048→128 → abs-difference → Linear 128→1 logit | SiameseNetwork.py:16–23 |
| 17 | Backbone initialization provenance | Fresh ImageNet `pretrained=True` weights for EVERY attacker seed; never the released verification-model weights; record torchvision weight-bundle identifier per run | SiameseNetwork.py:17 |
| 18 | Optimizer | Adam | dev_attacker.py:143 |
| 19 | Learning rate | 1e-4 | dev_attacker.py:98 (frozen config contract) |
| 20 | Batch size | 32 | dev_attacker.py:98; evaluator_common.py:635 |
| 21 | Maximum epochs | 100 | dev_attacker.py:100 |
| 22 | Early-stopping patience | 5 (no-improvement epochs on selection loss) | dev_attacker.py:100,241–247 |
| 23 | Attacker-training geometry | anon(x1)/anon(x2), frozen generator eval-mode, no_grad deformation | dev_attacker.py:153–161,102 |
| 24 | Attacker-selection geometry | anon(x1)/anon(x2) BCE on VAL pairs (early stopping/checkpoint selection ONLY — not the privacy metric) | dev_attacker.py:189–213,102 |
| 25 | Final P0 scoring geometry | anon(x1)/real(x2): image1 anonymized, image2 untouched real | eval_reid_val.py:55–56 |
| 26 | Selection loss & direction | BCEWithLogitsLoss on selection geometry; MINIMIZE; best epoch = minimum | dev_attacker.py:142,189–213,233–240 |
| 27 | Primary score definition | sigmoid(logit) ∈ [0,1]; higher = more likely SAME patient; direction fixed a priori | eval_reid_val.py:59; upstream test_snn semantics |
| 28 | AUC implementation | `sklearn.metrics.roc_auc_score(y_true, y_score)` raw; NO effective-AUC transform, NO max(AUC,1−AUC), NO label flip, NO score inversion, NO direction change after viewing results | eval_reid_val.py:76 |
| 29 | Seed→weight-init contract | `utils.seed_all(attacker_seed)` called BEFORE network init, loader creation, and optimizer creation | dev_attacker.py:119–120 |
| 30 | Seed→data-order contract | Same `attacker_seed` drives TRAIN shuffle order; identical seed mapping used for BOTH arms (paired design) | dev_attacker.py:10,120 |
| 31 | Per-epoch order-hash logging | REQUIRED (new): each future runner must compute and persist a semantic order hash per epoch per arm/seed; current DevAttacker does not log it — implementation requirement, absence = execution blocker | this report; evaluator_common.py hash pattern |
| 32 | Output freshness & collision protection | REQUIRED: refuse pre-existing outputs; stale-output rejection; atomic write-then-rename; deterministic filenames (§10) | this report |
| 33 | Raw prediction retention | y_true, y_score, pair identifier retained per run (never stripped) | §10 schema |
| 34 | Failure & exclusion policy | Fail-closed rules of §13; no exclusions, replacements, rescues | §13 |
| 35 | Software/environment provenance | Record Python/torch/torchvision/CUDA/driver/GPU identifiers per run manifest | standard provenance |
| 36 | Screen seed list | {42, 43, 44, 45, 46} | §9 |
| 37 | Full seed list | integers 42 through 67 inclusive | §9 |
| 38 | Statistical unit of replication | Paired attacker seed within the fixed (U_PUBLISHED, D_BDEV) generator pair — optimizer/data-order replicate ONLY | §10 |
| 39 | Primary paired estimand | Arithmetic mean of Δ_s across the 26 paired seeds | §10 |
| 40 | Provisional SEOI | δ = 0.03 AUC — PROVISIONAL_SEOI_PENDING_HUMAN_RATIFICATION | §7 |

Governance note: the current scientific `DevAttacker` hard-refuses any `attacker_seed != 42` (`dev_attacker.py:96`). Future P0 execution therefore requires a prior governance step minting a new frozen attacker-config artifact carrying the seed list (the predeclare-then-hash-gate pattern already used elsewhere), explicitly documented as an open feasibility item in S2_CONFIRMATORY_DESIGN_PROPOSAL.md §7. This is a human protocol decision, not something to route around. VERIFIED_FROM_SOURCE

VAL role limitation: the same 2000 VAL pairs serve early stopping (field 24), attacker selection, and final privacy scoring (field 25). Repeated VAL use makes P0 a CONDITIONAL DEVELOPMENT COMPARISON, not a locked independent confirmation. CURRENT_VAL_CONFIRMATORY_STATUS: NOT_LOCKED.

---

## 9. SEED AND STAGING PLAN

Diagnostic screen: seeds {42,43,44,45,46} × both arms = 10 trajectories. Technical and diagnostic ONLY (protocol failures, runtime failures, output corruption, unexpectedly extreme behavior). It must NEVER produce a scientific PASS/FAIL, superiority, equivalence, promotion, or publication claim.

Full bridge: seeds 42–67 inclusive × both arms = 52 trajectories; the five screen seeds are included; after a valid screen the remaining expansion is 42 trajectories.

Both arms rerun under the same locked runner/environment/seed mapping/data-order contract. The historical B_dev summary (mean 0.8237, SD 0.031, n=26; RESEARCH_DIRECTION_SUMMARY.md §3, S2 proposal §8) may appear ONLY as descriptive context and sanity anchor — it must NOT replace the newly paired D_BDEV arm and must NOT enter the primary paired test. HISTORICAL_BDEV_REUSE: FORBIDDEN_RERUN_BOTH_ARMS.

---

## 10. STATISTICAL ANALYSIS PLAN

Primary analysis: paired attacker-seed differences Δ_s; arithmetic mean; SD and SE across paired seeds; paired t-based CIs; TOST at α=0.05 against ±0.03 provisional bounds; one-sided directional confidence bounds for meaningful improvement (§7 rules).

Predeclared sensitivity analyses: paired sign-flip permutation test (exact if computationally practical; otherwise deterministic Monte Carlo seed and permutation count fixed BEFORE execution); seed-level bootstrap of paired differences ONLY; descriptive summaries — median, IQR, min, max, sign counts, full per-seed paired table.

The currently unvalidated patient-graph bootstrap is NOT used. PATIENT_GRAPH_BOOTSTRAP: UNVALIDATED_NOT_USED.

Inference scope statement: attacker seeds are optimizer/data-order replicates — not independent generator-training replicates and not independent patient-population samples. Any P0 inference is conditional on: two fixed generator checkpoints; one dataset/split; one attacker architecture; one training protocol; the declared attacker-seed-generating mechanism. P0 cannot establish population-level privacy protection or robustness to arbitrary attackers.

---

## 11. OUTPUT AND MANIFEST SPECIFICATION (future run only; NOTHING created now)

Root: `reproduction/p0_bridge/`. Deterministic filenames:

```text
reproduction/p0_bridge/<arm>/<seed>/
    predictions.parquet      # y_true, y_score, pair_id (+ permitted de-identified
                             #  endpoint/patient ids if governance permits)
    attacker_best.pth        # best attacker state dict
    run_manifest.json
manifest: reproduction/p0_bridge/manifest.jsonl   # append-only, atomic append
```

Required per-run retention: raw y_true; raw y_score; pair identifier; arm ∈ {U_PUBLISHED, D_BDEV}; seed; generator role/path/SHA-256; configuration hash; code commit; environment identifier; attacker initialization seed; per-epoch data-order hashes; best epoch; stop epoch; termination reason; selection loss at best epoch; final RAW AUC; start/end timestamps; output-freshness marker; failure status and reason.

Behavioral requirements: atomic write (write temp + fsync + rename); reject pre-existing target directories (collision prevention); stale-output rejection via freshness marker checked before reuse; manifest lines carry the same run identity keys so any mismatch fails closed. NO implementation is authorized in this task.

---

## 12. COST ESTIMATES (planning basis: measured 0.336 GPU-h per completed attacker trajectory)

| Stage | Trajectories | Estimate |
|---|---|---|
| Five-seed paired screen (both arms) | 10 | ≈ 3.4 GPU-hours |
| Full 26-seed paired bridge (both arms, including screen) | 52 | ≈ 17.5 GPU-hours |
| Remaining expansion after a valid screen | 42 | ≈ 14.1 GPU-hours |

These are planning estimates, NOT authorization. Sequential execution only while any other GPU process may exist.

---

## 13. FAIL-CLOSED RULES FOR FUTURE EXECUTION

Immediate stop, before or during execution, upon ANY of:
another training/evaluation process active on the required GPU; checkpoint path/role/SHA mismatch; source-commit mismatch; TRAIN/VAL pair-file or order-hash mismatch; unapproved data or split encountered; seed mapping differing across arms; attacker architecture or initialization-provenance difference; training/selection/scoring geometry difference; missing raw predictions; NaN or Inf anywhere; selection or early-stopping behavior differing between arms; pre-existing or stale output; output collision or partial write; any screen-trajectory failure; any omitted or replaced seed; any analyst proposal — AFTER viewing results — to flip scores, change thresholds, replace seeds, or exclude cases.

No partial-result rescue, post-hoc threshold change, post-hoc seed replacement, score inversion, or selective rerun is permitted. POST_HOC_SCORE_FLIP: FORBIDDEN.

---

## 14. INTERPRETATION LIMITS

State plainly: P0 equivalence establishes only similar behavior of TWO FIXED CHECKPOINTS under THIS EXACT current attacker protocol. A directional result establishes only a conditional difference between them. P0 does NOT validate Direction B; does NOT demonstrate a new privacy method; does NOT establish segmentation improvement; does NOT establish robustness to stronger/alternative/adaptive attackers; does NOT support a Q2-level contribution by itself. A genuine method contribution still requires: a newly trained method; multi-(generator-)seed confirmation with the generator seed as experimental unit; superiority beyond a predeclared meaningful margin; utility preservation; and independent confirmation resources.

---

## 15. HUMAN APPROVAL CHECKLIST

Explicit ratification required for EACH item; nothing is implicitly approved:
1. Checkpoint roles and SHA-256 values (three-role table).
2. Provisional SEOI δ = 0.03 (or replacement, frozen before execution).
3. Raw-AUC direction rule and score-flip prohibition.
4. Five diagnostic screen seeds {42–46}.
5. Full 26-seed range {42–67}.
6. Rerunning BOTH arms instead of reusing historical B_dev values.
7. The complete protocol-lock table (including fields 8, 12, 31 dispositions).
8. Output schema and retained raw predictions.
9. Separate authorization for the diagnostic screen.
10. Separate authorization for expansion to the full bridge.
11. GPU availability and absence of conflicting processes at execution time.
12. Acknowledged limitations of repeated VAL use (NOT_LOCKED confirmation status).
Plus resolution of BLOCKER field 12 (TRAIN/VAL patient-disjointness) before any execution.

---

## 16. MACHINE-READABLE FINAL VERDICT

```text
P0_CANONICAL_PROTOCOL_STATUS: READY_FOR_HUMAN_RATIFICATION
BASE_COMMIT: a3513e5c3b3b5631838399fc14c9e708909fe923
CLEAN_ROOM_FROM_AUDIT_BRANCH: YES

P0_ANCHOR_U_ROLE: U_PUBLISHED
P0_ANCHOR_U_SHA256: 4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153

P0_ANCHOR_D_ROLE: D_BDEV
P0_ANCHOR_D_SHA256: 18381d92c64bb3d646b62d5fb9d0ed8c208cf2cb3154f8aa1dac4b1baff610cd

P0_PRIMARY_DELTA: AUC_U_PUBLISHED_MINUS_AUC_D_BDEV
P0_SEOI: PROVISIONAL_0_03_PENDING_HUMAN_RATIFICATION
P0_SCREEN_SEEDS: 42_43_44_45_46
P0_FULL_SEEDS: 42_THROUGH_67
P0_SCREEN_INFERENCE: TECHNICAL_DIAGNOSTIC_ONLY
P0_FULL_INFERENCE: CONDITIONAL_FIXED_GENERATORS_ONLY

HISTORICAL_BDEV_REUSE: FORBIDDEN_RERUN_BOTH_ARMS
RAW_AUC_ONLY: YES
POST_HOC_SCORE_FLIP: FORBIDDEN
CURRENT_VAL_CONFIRMATORY_STATUS: NOT_LOCKED
PATIENT_GRAPH_BOOTSTRAP: UNVALIDATED_NOT_USED

P0_IMPLEMENTATION_AUTHORIZATION: NONE
P0_GPU_AUTHORIZATION: NONE
MODEL_OR_CHECKPOINT_EXECUTION: NONE
TEST_OR_SIMULATION_EXECUTION: NONE
FILES_CREATED: 1
EXISTING_FILES_MODIFIED: 0
COMMIT_OR_PUSH: NONE
ACTIVE_TRAINING_PROCESS_TOUCHED: NO
NEXT_REQUIRED_ACTION: EXTERNAL_HUMAN_REVIEW
```

*End of protocol lock. Stop and wait for external human review.*
