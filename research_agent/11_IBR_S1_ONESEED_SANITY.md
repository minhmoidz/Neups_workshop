# STEP 6C — IBR S1 One-Seed Sanity: STOP BEFORE TRAINING

**Status: STOPPED BEFORE TRAINING — CHECKPOINT SELECTION UNSPECIFIED**
**Commit:** `8f085d6` (clean tree) · branch `main` · no training, no attacker, no TEST.

---

## 1. Gate invoked

STEP 6C's **TRAINING SCHEDULE** section contains an explicit gate:

> **IMPORTANT:**
> If `09_IBR_PREIMPLEMENTATION_LOCK.md` does NOT explicitly define:
>     checkpoint-selection rule
> then **STOP BEFORE TRAINING** and report:
>     **CHECKPOINT SELECTION UNSPECIFIED**
> Do NOT invent a checkpoint-selection rule.

The lock was re-read in full. It does not explicitly define an S1 checkpoint-selection rule, an S1 epoch count/schedule, or a pretraining phase. Therefore the STEP 6C gate requires stopping before training. **No rule was invented.**

## 2. What the lock DOES specify (S1)

From `research_agent/09_IBR_PREIMPLEMENTATION_LOCK.md`:

- §2 Corrected computational graph (`x_self = G(z_id, z_med)`, `x_anon = G(z_id_donor, z_med)`, `L_rec` only on `x_self`).
- §3 `z_id` pairwise verification objective (BCE, `V` head).
- §4 `z_med` GRL-adversarial suppression (`H_med`).
- §5 Donor protocol (deterministic, patient-disjoint, no TEST donors, `donor != source`).
- §6 S1 objective: `L_S1 = λ_rec·L_rec + λ_path·L_path + λ_anat·L_anat + λ_zid·L_zid_pair + λ_adv·L_zmed_adv`, all λ=1.0, GRL λ=1.0.
- §7 Validation references: class VAL `0.793827`, Dice VAL `0.955019`, adaptive Re-ID VAL `0.8382`.
- §8 1-seed gates: adaptive Re-ID `≤ 0.750`; class drop `≤ 0.030` (≥ 0.7638); Dice drop `≤ 0.025` (≥ 0.9300); mechanism gates.
- §11 Architecture spec + optimizers: Adam `lr=1e-4` for E/G/V, Adam `lr=1e-4` for `H_med`; frozen classifier DenseNet-121 SHA `8ad15b38…`, frozen seg teacher `UNetSeg(1,3,16)` SHA `2dfdcf9b…`.
- §12 Batch size: **16** (feasible; no gradient accumulation).

## 3. What the lock does NOT specify

Searched full text of `research_agent/09_IBR_PREIMPLEMENTATION_LOCK.md` (all 298 lines) for schedule/selection terms:

| Item | Present in lock? |
| ---- | ---------------- |
| S1 epoch count (`max_epochs`/`num_epochs`) | **NO** — §12 is a memory probe only ("No optimizer, no epochs"). No S1 epoch count appears anywhere. |
| Pretraining phase (autoencoder/self-reconstruction) | **NO** — the only "pretrain" text in the repo is the unrelated STEP 0F flow-field `pretrain` operator. The STEP 6C prompt's "expected design" (pretraining then S1) is **not** present in the lock. |
| S1 checkpoint-selection rule (what to pick among S1 epochs) | **NO** — §10's "checkpoint-selection re-score" concerns the STEP 3D **attacker** convention (best validation AUC vs lowest validation loss, Δ≈0.0022), not S1 model selection. It explicitly cannot serve as an S1 rule. |
| S1 checkpoint-selection metric on which to select | **NO** |

The referenced Phase-II design documents (`08_PHASE2_POSITIVE_METHOD_DESIGN.md`, `08A_ORIGINAL_PAPER_CODE_RECONCILIATION.md`, `04_METHOD_SUCCESS_CRITERIA.md`, `04B_SUCCESS_CRITERIA_AMENDMENT.md`) remain absent from the repository (this absence was already flagged in the lock §13 blocker #1).

## 4. Decision

Per STEP 6C's explicit gate, training must not start until the S1 checkpoint-selection rule (and epoch schedule) are explicitly defined and locked. I do not have authority to define them.

**No S1 training was started. No adaptive attacker was trained. No TEST path was touched. No seed 42 run was executed.**

## 5. Required to unblock

1. Explicit S1 training schedule: epoch count (and whether a self-reconstruction pretraining phase precedes S1, with its own epoch count).
2. Explicit S1 checkpoint-selection rule: which per-epoch signal selects the final S1 checkpoint (e.g., lowest validation total loss, lowest validation privacy-relevant loss, a fixed epoch, or the epoch whose post-anonymization adaptive validation Re-ID is lowest), including tie-breaking.
3. Re-issue STEP 6C (or an amendment locking these two decisions) with the same budget: ONE S1 run (seed 42) + ONE attacker (seed 42), no tuning.

---

## 6. Artifact audit

| Artifact | State |
| -------- | ----- |
| S1 model checkpoint | NOT CREATED (no training) |
| Attacker diagnostics | NOT CREATED (no attacker) |
| Training curves | NOT CREATED |
| Any TEST access | NONE (TEST split remains rejected at argparse) |
| Git | `8f085d6`, clean working tree, branch `main` |

---

## 7. Reproducible check command

The absence can be reproduced:

```
grep -n -i "epoch\|schedule\|selection\|pretrain" research_agent/09_IBR_PREIMPLEMENTATION_LOCK.md
```

- No S1 epoch count, no S1 schedule, no S1 checkpoint-selection rule appear.

---

## 8. Conclusion

STOPPED BEFORE TRAINING — **CHECKPOINT SELECTION UNSPECIFIED**.

The S1 mechanism implementation from STEP 6B remains intact and verified; nothing was trained or changed in this step. A short owner decision (schedule + checkpoint-selection rule) is required before any S1 epoch or attacker run may begin.

---

CHECKPOINT SELECTION UNSPECIFIED — NO TRAINING STARTED