# P0–P1.2 ANCHOR-ROLE CORRECTION ADDENDUM

**Date:** 2026-08-21
**Status:** NARROW SOURCE-BASED ADDENDUM ONLY. No checkpoint loading, no model execution, no testing, simulation, training, evaluation, GPU work, commit, or push.

---

## 1. EXECUTIVE CORRECTION

The P0–P1.1 erratum designated Checkpoint A (`networks/pretrained_generator_prichexy_net.pth`, SHA `10122689…`) as Anchor U for P0, reasoning from governed current-code SHA gating. That role assignment was **wrong**. The public upstream source at commit `29245d1` establishes two distinct roles:

- **Checkpoint A is the PRETRAINED INITIALIZATION** loaded by `agents/Agent.py` *before* the adversarial PriCheXy training loop begins — correctly named **I_M2 (INITIAL M2 GENERATOR)**. It must not be called the published trained PriCheXy endpoint merely because current governed code SHA-gates it as an initialization artifact.
- **Checkpoint B (`generator_lowest_total_loss_mu_0.01.pth`, SHA `4d82dcdd…`) is the RELEASED TRAINED PRICHEXY ENDPOINT AT μ=0.01** — explicitly designated as the perturbation model by the upstream README, the attacker-retraining config, and the classification-evaluation config. Correctly named **U_PUBLISHED**.

Consequences:
- **P0 must compare U_PUBLISHED versus D_BDEV** (the selected current B_dev generator checkpoint at epoch 13), retaining the corrected sign rule and TOST rules from P0–P1.1. Checkpoint A is NOT the primary upstream endpoint for this bridge.
- The P1a anchors are unchanged: Anchor 1 = I_M2 (Checkpoint A), Anchor 2 = D_BDEV. P1a still cannot test the historical live-critic gap.
- The P0–P1.1 semantic audit remains fully valid (BYTE_DISTINCT / SEMANTICALLY_DIFFERENT / PROVENANCE_RESOLVED); only the experimental ROLE assignment is corrected. GOVERNANCE_CORRECTION

---

## 2. PUBLIC UPSTREAM EVIDENCE

All citations are to files at upstream commit `29245d1f71571898d9527417df4ae3f63a8695f6` (inspected read-only via `git show <commit>:<path>`; byte-identical to the working-tree copies verified previously). VERIFIED_FROM_SOURCE

### 2.1 Checkpoint A = pretrained initialization (I_M2)
- `agents/Agent.py:65`: `self.generator.load_state_dict(torch.load('./networks/pretrained_generator_prichexy_net.pth'))` — executed in `__init__`, i.e., BEFORE the complete adversarial training loop (`run()` → `utils.train`/`validate`).
- `agents/Agent.py:166–178`: during that loop the code SAVES checkpoints named `generator_lowest_ac_loss.pth`, `generator_lowest_ver_loss.pth`, `generator_lowest_total_loss.pth`. The released file `generator_lowest_total_loss_mu_0.01.pth` is manifestly a TRAINED OUTPUT of this loop under μ=0.01, not its input initialization.
- `config_files/config_anonymization.json`: training contract (`"generator_type": "flow_field"`, `"mu": 0.01`, batch 64, lr 1e-4, max_epochs 250) under which such lowest-total-loss endpoints were produced.

### 2.2 Checkpoint B = published trained endpoint at μ=0.01 (U_PUBLISHED)
- `README.md:55`: *"perturbation_model_file": The perturbation model, e.g. "./networks/generator_lowest_total_loss_mu_0.01.pth".* (attacker-retraining instructions)
- `README.md:75`: same designation for classification-evaluation instructions.
- `config_files/config_retrainSNN.json`: `"perturbation_type": "flow_field", "perturbation_model_file": "./networks/generator_lowest_total_loss_mu_0.01.pth", "mu": 0.01`.
- `config_files/config_eval_classifier.json`: identical designation for the classifier-evaluation path.

The upstream release therefore ships exactly one trained flow-field endpoint at μ=0.01 and points all downstream threat-model configurations at it. VERIFIED_FROM_SOURCE

---

## 3. THREE-ROLE CHECKPOINT TABLE

| Role name | Artifact | SHA-256 | Source basis |
|---|---|---|---|
| **I_M2** — INITIAL M2 GENERATOR | `networks/pretrained_generator_prichexy_net.pth` | `101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb` | `agents/Agent.py:65` loads it before training; current governed code uses it as SHA-gated initialization (`evaluator_common.py:56–57`) |
| **U_PUBLISHED** — RELEASED TRAINED PRICHEXY ENDPOINT AT μ=0.01 | `networks/generator_lowest_total_loss_mu_0.01.pth` (local LFS-materialized copy: `reproduction/workspace_prichexy/networks/generator_lowest_total_loss_mu_0.01.pth`) | `4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153` | `README.md:55,75`; `config_retrainSNN.json`; `config_eval_classifier.json` |
| **D_BDEV** — CURRENT SELECTED CHECKPOINT | B_dev selected generator, epoch 13 of the certified 250-epoch run (`research_runs/M2_S1/B_dev/seed_42/generator_best_method_neutral.pth`) | `18381d92c64bb3d646b62d5fb9d0ed8c208cf2cb3154f8aa1dac4b1baff610cd` | `M2_S1_C4_RESULT.md` §1, §4 |

Semantic relationships from P0–P1.1 (unchanged): I_M2 vs U_PUBLISHED are BYTE_DISTINCT and SEMANTICALLY_DIFFERENT (all 118 tensors differ, including all BatchNorm buffers — consistent with U_PUBLISHED being a trained endpoint and I_M2 its initialization). VERIFIED_BY_CPU_SEMANTIC_AUDIT (inherited, remains valid)

---

## 4. CORRECT P0 ANCHOR DEFINITION

P0 protocol-equivalence bridge compares:

```text
Anchor U := U_PUBLISHED  (generator_lowest_total_loss_mu_0.01.pth, SHA 4d82dcdd…)
Anchor B := D_BDEV       (B_dev selected epoch-13 checkpoint, SHA 18381d92…)
```

with the unchanged paired-seed delta definition:

```
Δ_bridge,s = AUC_U_PUBLISHED,s − AUC_D_BDEV,s
```

Sign rule and decision table from P0–P1.1 §7–§8 are retained verbatim (Δ<0 ⇒ upstream endpoint has better privacy; TOST α=0.05 equivalence; directional outcomes require predeclared confidence-bound exclusion of ±δ with δ = 0.03 labeled `PROVISIONAL_SEOI_PENDING_HUMAN_APPROVAL`). MATHEMATICAL_CORRECTION (retained)

Checkpoint A (I_M2) is not the primary upstream endpoint for this bridge. GOVERNANCE_CORRECTION

---

## 5. CORRECT P1a ANCHOR DEFINITION

Unchanged anchors, now with correct names:

```text
P1a Anchor 1 = I_M2   (Checkpoint A — initial M2 generator, pretrained, before adversarial fine-tuning)
P1a Anchor 2 = D_BDEV (selected B_dev epoch-13 checkpoint)
```

Naming discipline from the original P0–P1 report remains binding (no "random"/"untrained"/"epoch-250" labels; two development anchors, not a trajectory). P1a still cannot test the historical live-critic gap; P1b remains BLOCKED_NO_LIVE_CRITIC_CHECKPOINT per P0–P1.1 §12.

Optional internal comparison: a future comparison between I_M2 and D_BDEV may be proposed as a separate internal fine-tuning diagnostic. It must NOT be substituted for the P0 published-upstream bridge and is NOT authorized in this task. PROPOSED_DESIGN / UNAUTHORIZED_HERE

---

## 6. SUPERSEDED P0–P1.1 STATEMENTS

The following P0–P1.1 statements are superseded WITHOUT modifying that file:

1. The designation of Checkpoint A as Anchor U (§6 of P0–P1.1).
2. The claim that governed current initialization determines the published upstream endpoint ("Choose the operational Anchor U path based on the governed current code/provenance").
3. The statement that the paper/release evaluation role of the two checkpoints cannot be determined from public code (the "residual descriptive question" passage) — public upstream code DOES determine it.
4. The suggestion that Checkpoint B is merely an optional secondary variant for human consideration.
5. The P0 human-approval checklist item asking humans to ratify Checkpoint A as Anchor U (P0–P1.1 §14, second item).
6. The earlier machine-readable field `P0_ANCHOR_STATUS: RESOLVED` insofar as it resolved the role to Checkpoint A; corrected to RESOLVED_CORRECTED_TO_CHECKPOINT_B.

Remains fully valid: the CPU semantic audit results (BYTE_DISTINCT, SEMANTICALLY_DIFFERENT, PROVENANCE_RESOLVED), all safety attestations, the corrected P0 sign/TOST mathematics, the five-seed screen limitation, the Git-topology correction, the P1a/P1b split, and the cost bases. Only the experimental role assignment changes. GOVERNANCE_CORRECTION

---

## 7. COST AND AUTHORIZATION STATUS

Because P0 still compares exactly TWO generators (U_PUBLISHED vs D_BDEV), no additional dual-anchor cost arises beyond the existing planning estimates:

| Item | Estimate | Status |
|---|---|---|
| P0 5-paired-seed screen, two generators | ≈ 3.4 GPU-hours | planning estimate only; DIAGNOSTIC_ONLY inference; NOT AUTHORIZED |
| P0 26-paired-seed bridge, two generators | ≈ 17.5 GPU-hours | planning estimate only; NOT AUTHORIZED |

Basis remains the measured ≈0.336 GPU-h per attacker trajectory (50 completed runs, previous reports; unaffected finding). No execution of any kind is authorized by this addendum. GPU_AUTHORIZATION: NONE. MATHEMATICAL_CORRECTION (retained figures)

---

## 8. MACHINE-READABLE FINAL VERDICT

```text
ADDENDUM_STATUS: PASS
P0_P1_1_SEMANTIC_AUDIT: REMAINS_VALID
CHECKPOINT_A_ROLE: I_M2_INITIAL_GENERATOR
CHECKPOINT_B_ROLE: U_PUBLISHED_TRAINED_ENDPOINT_MU_0_01
BDEV_ROLE: D_BDEV_CURRENT_SELECTED_CHECKPOINT

P0_PRIMARY_COMPARISON: U_PUBLISHED_VS_D_BDEV
P0_ANCHOR_STATUS: RESOLVED_CORRECTED_TO_CHECKPOINT_B
P0_SCREEN_INFERENCE_STATUS: DIAGNOSTIC_ONLY
P0_EXECUTION_AUTHORIZATION: NONE

P1A_ANCHOR_1: I_M2_CHECKPOINT_A
P1A_ANCHOR_2: D_BDEV
P1A_EXECUTION_AUTHORIZATION: NONE
P1B_LIVE_CRITIC_GAP: BLOCKED_NO_LIVE_CRITIC_CHECKPOINT

DUAL_ANCHOR_GPU_REQUIREMENT: NOT_REQUIRED_FOR_P0
GPU_AUTHORIZATION: NONE
MODEL_OR_CHECKPOINT_EXECUTION: NONE
FILES_CREATED: 1
EXISTING_FILES_MODIFIED: 0
CURRENT_TASK_COMMIT_OR_PUSH: NONE
ACTIVE_TRAINING_PROCESS_TOUCHED: NO
NEXT_REQUIRED_ACTION: EXTERNAL_HUMAN_REVIEW
```

*End of addendum. Stop here; do not begin P0 or P1.*
