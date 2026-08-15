# M1 — C4 Development Protocol Lock

- **Protocol version:** **1.1.0** (M1.1: scientific promotion gate repair; seed/cost consistency repair; segmentation provenance recovery attempt)
- **Task:** Freeze a scientifically defensible TRAIN/VALIDATION-only protocol to compare **restored PriCheXy-Net baseline (B_dev)** vs **C4 (feature-preservation term, `L_feat`, DenseNet-121 penultimate-pooled 1024-d, MSELoss, detached source, feature_loss_weight=1.0)** before any re-run.
- **Absolute execution lock:** NO training of B_dev / C4 / C2 / C2+C4 / C3 / anonymizer / attacker; NO TEST access; NO mu or feature-loss-weight sweeps; NO architecture change. This document only READs code/configs, runs M0 suite, and performs tiny non-scientific probes.
- **Reproducibility index:** R6 (READ + tiny probes) → "R5-SCI/READ" boundary (finalize → RUN with same configs).

---

## §0 Preconditions (verified at lock time)

| Check | Status |
|---|---|
| Branch `research/method-restart` created from `original-upstream` `29245d1f71571898d9527417df4ae3f63a8695f6` | OK |
| M0 committed `6d9b5ba76e1312e54ec1c10934d3a9bfd7f42f43` (17 files) | OK |
| Branch manifest committed `14f6715c922e2fc27338d265fcede1f48eb58298` | OK |
| M0.1 provenance-hash repair committed `9af21e0189d7b57d5575d3ffd8fc4604fea56ffd` | OK |
| M1 protocol lock committed `7156d8506468e7adec037d62143923a33cbce3e4` | OK |
| M0 test suite PASS (27/27) at M1 lock; re-verified 28/28 at M1.1 lock | OK |
| Tracked working tree modified at M1.1 | ONLY M1.1-related files staged for this commit |
| `OFFICIAL_TEST_LOCK.md` respected (TEST never read/touched) | OK |

> Note: untracked `archive/`, `chexnet/results/`, `reproduction/`, `logs/`, `data/`, `networks/corrected_baseline/` and assorted `research_agent/*` working files predate this restart (earlier sessions). They are NOT part of the M1 freeze and must NOT be consumed for any M1 decision. Only the tracked M0/M0.1 artifacts plus the configs finalized below are authoritative.

---

## §1 Precondition report (M1 step 1)

- Re-ran full suite at lock HEAD `9af21e0`: **27/27 PASS** (8 test files, including M0.1 hash test).
- All M0/M0.1 artifacts verified present and committed.
- No TEST data, no TEST eval code, no official results touched.

---

## §2 Baseline lineage and environment facts

### 2.1 Restored baseline (frozen, legacy, released generator)

| Artifact | SHA-256 | Verifier |
|---|---|---|
| `networks/generator_lowest_total_loss_mu_0.01.pth` (released) | `4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153` | M0.1 |
| `networks/pretrained_classifier.pth` | `8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663` | M0.1 |
| `networks/pretrained_verification_model.pth` | `331efaed0c0433c69941ddc003a14a936c688d94fd4ecfbefd34e53bfa7c051a` | M0.1 |
| `networks/pretrained_generator_prichexy_net.pth` (upstream anonymizer-train INIT, distinct from released) | `101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb` | sha256sum + git index |
| pair-file training | `3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268` | prior audit |
| pair-file validation | `9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7` | prior audit |

The released generator (`generator_lowest_total_loss_mu_0.01.pth`) is the TRAINED OUTPUT of the pristine pipeline; upstream anonymizer training INITIALIZES from `pretrained_generator_prichexy_net.pth` (Agent.py `torch.load('./networks/pretrained_generator_prichexy_net.pth')`). Tensor-level comparison confirms the two files differ (maxdiff ≈ 0.067 on shared keys). B_dev/C4 retraining therefore starts from the upstream init checkpoint, NOT the released generator.

### 2.2 Environment

- GPU: RTX 5070 Ti 16GB (15.46 GiB usable), `torch 2.7.0+cu128`, Python 3.10.20.
- **VRAM probe (full anonymizer-training graph at 256px: UNet+gauss+grid_sample+ACLoss-DenseNet-deepcopy+Siamese+3 optimizers):**

| batch | result | peak_alloc |
|---|---|---|
| 64 | OOM | — |
| 32 | OOM (even with `expandable_segments`) | — |
| 24 | OK | 12,330 MiB (79%) |
| 16 | OK | 8,294–8,722 MiB (54–56%) |

- **Decision: freeze `batch_size = 16`** (preference hierarchy 64→32→16; 64 and 32 OOM, 24 rejected as >75% headroom risk under real data loading; 16 is the prescribed safe fallback and leaves 44% headroom). Timing probe: 0.296 s/iter → 625 iters/epoch (10000/16) → 3.1 min/epoch → ~12.8 h per 250-epoch anonymizer run at batch 16 (compute-only).
- Attacker recipe (`config_retrainSNN.json`, frozen): SiameseNetwork (ResNet-50), `BCEWithLogitsLoss`, Adam, **batch 32**, lr 1e-4, max_epochs 100, early_stopping 5, image 256, flow_field mu=0.01, fresh ImageNet init, checkpoint = lowest validation loss. Confirmed in `agents/AgentSiameseNetwork.py:95-98,122-127`.

---

## §3 Potential ambiguities and resolutions (frozen, pre-result)

| # | Ambiguity | Resolution (method-neutral, predeclared) |
|---|---|---|
| A1 | Batch size under 16GB VRAM | **16** (both arms; 64/32 OOM; 24 rejected for headroom). B_dev and C4 MUST use identical batch and identical iter order. |
| A2 | Max epochs | **250** (upstream `max_epochs`), 1 accumulation step. No early stopping. |
| A3 | Checkpoint selection | **`lowest_validation_total_loss` with method-neutral rule**: selection uses the **B_dev objective** (`ac_loss_weight*ac_bce + ver_loss_weight*(-log(1-ver))`, i.e. feature term EXCLUDED) applied to BOTH arms. For C4 the selection metric is the SAME formula (feature term not included in the selection metric). Tie-break: earliest epoch. This is the predeclared method-neutral rule that is computable for both arms without TEST. |
| A4 | Auxiliary classifier (critic) semantics | Upstream: `optimizer_ac = SGD(lr=1e-4, momentum=0.9, wd=1e-4)` updates `ac_model` EVERY iteration; ACLoss deepcopies `ac_model` each forward → generator faces current classifier. Repaired `m0_port/ACLoss.py` reproduces this (refresh at forward start). **Same for both arms; C4 must NOT change critic update cadence.** |
| A5 | Feature term for C4 | `L_feat = MSELoss(r_detached_source(224), r_gen(224))` on DenseNet-121 penultimate pooled (1024-d) features; source branch detached (`feature_loss_detach_source=true`). weight `1.0`. Only delta from B_dev. |
| A6 | Privacy critic input | Raw anonymized 256px → Resize 224 → ImageNet normalize (batch norm). Verifier = SiameseNetwork; labels from SiameseDataset (1.0 = same patient). |
| A7 | Seed policy | **S1: anonymizer seed 42 for both arms; attacker seed 42.** **S2: anonymizer seed 42 ONLY (reuse frozen S1 generators, NO retraining); adaptive attacker seeds 42,43,44 per arm.** Paired: B_dev-attacker-42 ↔ C4-attacker-42, 43↔43, 44↔44. Upstream hardcodes `seed_all(42)` (Agent.py `utils.seed_all(42)`); 42 is the faithful primary. Multi-anonymizer-seed robustness (43/44) is NOT part of S2; if ever needed it is a separate later pre-registered stage. |
| A8 | C2/C3 during dev | **OFF** (`ver_ensemble_size=1`, `ver_restart_every=0`, `ver_warmup_iters=0`, `use_budget_map=false`). mu fixed 0.01. |
| A9 | Initial generator for retraining | `networks/pretrained_generator_prichexy_net.pth` (upstream init; SHA `101226890c...`), identical for both arms. Released generator is the baseline OUTPUT reference, not the init. |
| A10 | Verification/Siamese init for attacker | Fresh ImageNet-init ResNet-50 per run (canonical attacker recipe). |
| A11 | Segmentation evaluator | **STILL_BLOCKED after recovery audit** — see §15 and `research_agent/M1_1_SEGMENTATION_PROVENANCE.md`. Segmentation gate is **NOT APPLICABLE while BLOCKED** and must NOT be silently treated as PASS; if the evaluator becomes certified, it becomes REQUIRED with frozen thresholds. |

---

## §4 Paired table (frozen)

| Component | B_dev (control) | C4 | Delta |
|---|---|---|---|
| Initial generator | `pretrained_generator_prichexy_net.pth` | same | none |
| mu | 0.01 | 0.01 | none |
| image_size | 256 | 256 | none |
| batch_size | 16 | 16 | none |
| optimizer (generator) | Adam lr 1e-4 | same | none |
| max_epochs | 250 | 250 | none |
| ac_loss_weight / ver_loss_weight | 1 / 1 | 1 / 1 | none |
| **feature_loss_weight** | **0.0** | **1.0** | **ONLY delta** |
| feature loss term | disabled | MSELoss 1024-d penultimate, detached source | C4 |
| critic update | SGD every iter | SGD every iter (unchanged) | none |
| checkpoint rule | lowest val total (method-neutral) | same metric | none |
| seed | 42 | 42 | none |
| C2/C3 | off | off | none |

---

## §5 Frozen development training protocol

### 5.1 B_dev (control)
`config_files/config_dev_restored_baseline.json` (SHA `1f7df9e5db0c1e1b8e79b540d227d8b143b83a0f0ca6363855120c870aee221b`):
- Pristine PriCheXy-Net objective, ACLoss `feature_loss_weight=0.0`, `ac_pos_weight=null`.
- Run exactly the restored-baseline objective on TRAIN, checkpoint by validation rule in §3-A3.

### 5.2 C4 (method)
`config_files/config_dev_c4.json` (SHA `e63f0a9ee0ecba2dd949a4037e113b2b7b5ab0d3688f196f7a7a62cc8548fb40`):
- Same as B_dev EXCEPT `feature_loss_weight=1.0` with `L_feat` term (§3-A5).
- Only acceptable difference from B_dev.

### 5.3 Evaluators (VALIDATION only; TEST CLOSED)
- **Classification:** `config_eval_classifier.json` pattern, DenseNet-121 `pretrained_classifier.pth`, run on anon train/val images (NOT test images). Metric: **mean ROC-AUC over 14 classes** (primary).
- **Privacy/Re-ID:** attacker recipe §2.2 run on train/val pairs only; metric mean AUC. Exact protocol (incl. seed) frozen pre-run.
- **Segmentation: STILL BLOCKED (see §15 and `M1_1_SEGMENTATION_PROVENANCE.md`).** Checkpoint `archive/train_seg_unet/best.pth` (SHA `2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0`, UNet 1→3 ch, init_features 16, epoch 20, mean_dice 0.9548). Recovery audit found the training script (`train_seg.py`), dataset class (`chexnet/seg_dataset.py`), mask loader (`utils/segmask.py`), model (`networks/UNetSeg.py`), metric code (`eval_seg.py`) and documentation chain (HANDOVER/GOALS.md, RESEARCH_BRIEF.md, 03H audit) recoverable from git history with checkpoint self-metadata matching documented val dice. **Still BLOCKED because:** (1) the actual training log for the checkpoint run is NOT preserved — the only segmenter log (`logs/train_seg.log`) is a different broken run (best mean val dice ≈ 0.06); (2) the exact training CLI/config for the checkpoint run is not reconstructable (checkpoint records only init_features=16 ≠ code default 32, epoch, dice; not lr/batch/seed/epochs). Does NOT block the privacy/classification protocol. Repair: commit canonical seg source onto the branch and/or re-run under a certified protocol with preserved config+log+split, then re-certify with a new checkpoint SHA.

---

## §6 Registries & fingerprints

- Frozen config SHAs (lock-time): B_dev `1f7df9e5...`, C4 `e63f0a9e...`. Re-verify at RUN start.
- M0/M0.1/M1 commits: `6d9b5ba`, `14f6715`, `9af21e0`, `7156d85`. Branch HEAD at M1.1 lock: `7156d8506468e7adec037d62143923a33cbce3e4`.
- Data hashes fixed (§2.1). No dataset files tracked (LFS).
- `M1_C4_PROTOCOL_LOCK.json` (v1.1.0) carries machine-readable frozen values.
- `M1_1_SEGMENTATION_PROVENANCE.md` carries the segmentation recovery audit.

---

## §7 Pre-registered expectations & FALSE POSITIVE CONTROL

- Do NOT use context-only numbers for any decision: paper 0.577±0.040; faithful reproduction R1-FINAL 0.6080±0.0435 (context-only, seed-swept, n=10).
- Control target: adaptive patient Re-ID **AUC < 0.57** (full SI-CLIP) while classification ROC-AUC ≈ baseline utility. All SI-CLIP infra is TEST-gated and will run under the same TEST lock in the R-phase, NOT now.
- **FALSE POSITIVE CONTROL (must be recorded in same RUN block as C4):** re-verify B_dev reproduces the M0-frozen pristine semantics by (a) config hash match to §6, (b) ACLoss repair == pristine upstream (already proven by T4e), (c) identical batch/seed with C4. If B_dev under M1 protocol fails these checks, the paired comparison is invalidated, not "passed".

---

## §8 S1 / S2 gates (v1.1.0 — RUN VALIDITY separated from SCIENTIFIC PROMOTION)

### 8.1 S1 validity (RUN VALIDITY ONLY — NOT scientific promotion)
Both arms must: complete normally; have no NaN/Inf; use frozen config hashes (§6);
use identical paired training semantics (same batch/iter order/seed); produce valid
checkpoints; respect the TEST firewall.
- If not: **S1 = INVALID**. Fix infrastructure only; do **NOT** interpret metrics.
- A valid run can scientifically FAIL. A valid run and scientific promotion are separate.

### 8.2 S1 scientific promotion (attacker seed 42; frozen pre-result, thresholds immutable after results)
- **Privacy non-regression:** `AUC_C4_VAL <= AUC_Bdev_VAL + 0.03`.
- **Classification utility:** `macro_AUC_C4_VAL >= macro_AUC_Bdev_VAL` (primary metric =
  mean ROC-AUC over 14 pathologies, DenseNet-121 `pretrained_classifier.pth`, VAL split).
- **Segmentation:** NOT silently treated as PASS. Segmentation is REQUIRED-IF-CERTIFIED:
  `Dice_C4 >= Dice_Bdev - 0.005` AND no material HD95 degradation — but while the evaluator
  is BLOCKED (§15), this criterion is **NOT APPLICABLE**.
- **Promotion decision:** C4 progresses only if **RUN_VALID AND privacy non-regression AND
  classification utility non-regression/improvement** (plus segmentation criterion only if
  certified).
- If `Delta_priv <= -0.03` (i.e. `AUC_C4_VAL <= AUC_Bdev_VAL - 0.03`), record that as a
  **STRONG PRIVACY SIGNAL** — but it is NOT mandatory, because C4's primary hypothesis is
  utility preservation.

### 8.3 S2 seed semantics and promotion (frozen before results)
- **S2 anonymizer seeds:** `[42]` ONLY (same frozen S1 generators; NO anonymizer retraining).
- **S2 attacker seeds:** `[42, 43, 44]` per arm — 3 fresh adaptive attacker restarts per arm.
- **Paired:** B_dev-attacker-42 ↔ C4-attacker-42, 43↔43, 44↔44.
- **S2 reuses frozen S1 generator checkpoints** (one frozen B_dev generator, one frozen C4
  generator); seed-42 attacker reused from S1 when valid.
- **Primary S2 statistic:** mean adaptive VAL Re-ID AUC over seeds 42,43,44. Also report:
  sample SD, median, min, max, paired per-seed delta.
- **No hidden factorial seed expansion:** anonymizer seeds 43/44 are NOT introduced in
  M2/S2. Multi-anonymizer-seed robustness, if eventually needed for publication, is a
  separate later pre-registered stage.

### 8.4 S2 promotion rule (frozen before results)
- `Delta_priv = mean(AUC_C4_VAL seeds 42,43,44) - mean(AUC_Bdev_VAL seeds 42,43,44)`.
- Promotion to later stronger-deformation/C2 investigation requires: **mean privacy does not
  materially regress** AND **classification advantage/non-regression remains**.
- Privacy non-regression ceiling: **`Delta_priv <= +0.03`**.
- Strong privacy improvement: **`Delta_priv <= -0.03`**.
- **If `Delta_priv > +0.03`, C4 FAILS promotion even if classification is better.**
- Do not redefine these rules afterward.

---

## §9 Final verdict format (for the M1.1 step itself)

```
M1.1 VERDICT: {PASS|BLOCKED}
- Protocol version: 1.1.0
- S1 validity separated from scientific promotion (valid run can scientifically FAIL)
- S1 privacy ceiling: AUC_C4_VAL <= AUC_Bdev_VAL + 0.03 (attacker seed 42)
- S1 classification gate: macro_AUC_C4_VAL >= macro_AUC_Bdev_VAL (mean ROC-AUC over 14 labels)
- S1 segmentation gate: REQUIRED-IF-CERTIFIED (Dice_C4 >= Dice_Bdev - 0.005, no material HD95 degradation); NOT APPLICABLE while BLOCKED
- S2 anonymizer seeds: [42] only (reuse frozen S1 generators; no retrain)
- S2 attacker seeds: [42, 43, 44] per arm, paired per seed
- S2 privacy promotion ceiling: Delta_priv <= +0.03 (Delta_priv > +0.03 => FAIL even if classification better)
- Segmentation: STILL_BLOCKED (recovery audit in M1_1_SEGMENTATION_PROVENANCE.md)
- S1 GPU cost: ~32.6 h; S2 incremental: ~12 h; total through S2: ~44.6 h
- Protocol lock files updated to v1.1.0, committed+ pushed on research/method-restart @ <new-commit>
- Next step (NOT started): M2-S1 — paired B_dev vs C4, anonymizer seed42 only, TRAIN/VALIDATION ONLY.
```

---

## §10 Cost estimate (corrected for M1.1 design, from probe at batch 16)

### S1
| Item | GPU time |
|---|---|
| Anonymizer B_dev seed42 (250 ep, 625 it/ep @0.296s) | ~12.8 h |
| Anonymizer C4 seed42 (same budget) | ~12.8 h |
| Adaptive attacker per arm (seed 42) | ~3 h each |
| Classification VAL per arm | ~0.5 h each |
| **S1 total** | **~32.6 h** |

### S2 incremental (after S1 PASS; NO anonymizer retrain)
| Item | GPU time |
|---|---|
| Reuse frozen S1 generator checkpoints (both arms) | 0 h |
| Reuse S1 seed-42 attacker (both arms, when valid) | 0 h |
| Attacker seed 43 × 2 arms | ~6 h |
| Attacker seed 44 × 2 arms | ~6 h |
| **S2 incremental total** | **~12 h** |

**Total through S2: ~44.6 h.** Feasible on the 5070 Ti 16GB. No under-reporting; S2
incremental contains only additional attacker runs + evaluator cost.

---

## §15 Segmentation evaluator (recovered provenance, STILL BLOCKED)

See `research_agent/M1_1_SEGMENTATION_PROVENANCE.md` (added in M1.1) for the full audit.
Summary: `archive/train_seg_unet/best.pth` (SHA `2dfdcf9b…`) self-records epoch 20,
init_features 16, mean_dice 0.9548, dice [0.9544, 0.9639, 0.9462] which exactly match the
documentation in `HANDOVER/GOALS.md` and `RESEARCH_BRIEF.md`. Training script, dataset
class, mask loader, model and metric code are recoverable from git history (commit
`9eaa5fd`). Certification remains **BLOCKED** because the checkpoint run's own training log
is not preserved (the only log `logs/train_seg.log` is a different broken run, best dice
≈ 0.06) and the exact training CLI/config is unreconstructable. Historical TEST results
(03H fold==test) are explicitly not certification. Segmentation is therefore NOT silently
treated as PASS; the gate stays REQUIRED-IF-CERTIFIED and is NOT APPLICABLE while BLOCKED.
This does not block the privacy/classification protocol.