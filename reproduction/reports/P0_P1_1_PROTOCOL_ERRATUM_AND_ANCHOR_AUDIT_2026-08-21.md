# P0–P1.1 MATHEMATICAL, GOVERNANCE, AND CHECKPOINT-IDENTITY ERRATUM

**Date:** 2026-08-21
**Status:** ERRATUM + NARROW CPU-ONLY CHECKPOINT AUDIT ONLY. No P0/P1a/P1b execution, no generator training, no attacker training, no model evaluation, no simulation, no GPU use.

---

## 1. EXECUTIVE ERRATUM VERDICT

1. **Retraction (governance):** The previous P0–P1 report improperly referenced prohibited frozen-evaluation material in parts of Sections 2 and 5. Those references and all conclusions depending on them are retracted. No underlying prohibited resource was opened or inspected in this erratum task. The user-reported approximately 0.60 value is retained only as unverified contextual motivation and is not treated as repository-verified evidence. GOVERNANCE_CORRECTION
2. **Checkpoint identity resolved:** The two byte-distinct candidate upstream generators are **SEMANTICALLY DIFFERENT** at the tensor level (all 118 tensors differ), but **both trace directly to the upstream release commit `29245d1`** — one as a regular blob (`networks/pretrained_generator_prichexy_net.pth`), one as an LFS object (`networks/generator_lowest_total_loss_mu_0.01.pth`, LFS oid `4d82dcdd…`). Provenance is RESOLVED; the operational Anchor U is designated by governed current code as Checkpoint A (`networks/pretrained_generator_prichexy_net.pth`, SHA `10122689…`), which is the SHA-gated `INITIAL_GENERATOR_PATH` of the certified pipeline. VERIFIED_BY_CPU_SEMANTIC_AUDIT / VERIFIED_FROM_SOURCE
3. **P0 sign rule corrected:** with Δ_bridge = AUC_U − AUC_B, negative Δ means Anchor U has better privacy. The previous directional outcome table used the wrong sign and is replaced by a four-outcome TOST-based table. MATHEMATICAL_CORRECTION
4. **Git topology corrected:** `a3513e5` and `c2ee268` are sibling documentation commits sharing the same parent `34bca1f`. There was never an unexplained divergence of the canonical method history. VERIFIED_FROM_SOURCE
5. **P1 split:** P1a (frozen-generator critic budget/geometry probe) is DESIGN_READY_BUT_NOT_AUTHORIZED; P1b (historical live-critic-gap test) is BLOCKED_NO_LIVE_CRITIC_CHECKPOINT and would require a separately authorized full co-training trajectory with snapshot capture — not a micro-run. GOVERNANCE_CORRECTION / PROPOSED_DESIGN
6. All unaffected source-code findings of the previous report remain usable (see §9). No execution authorization is granted. NEXT_REQUIRED_ACTION: EXTERNAL_HUMAN_REVIEW.

---

## 2. SCOPE AND SAFETY CONFIRMATION

- Worktree used (created this task under explicit human authorization after the initial branch mismatch was reported): `/tmp/p0p1-review-worktree`, branch `research/method-restart-p0p1-review`, HEAD `c2ee268d3d9af1f1a5b589a040db178b46b17763`. Verified by `git rev-parse --abbrev-ref HEAD` / `git rev-parse HEAD` before any work. VERIFIED_FROM_SOURCE
- Active training process check performed at start and end: PID 1875303 (`run_hardened_verifier.py --arm B_dev --k_extra_verifier_steps 2 --seed 42 --max_epochs 250 --tag k3`) active throughout; untouched. The erratum work was performed exclusively in the separate review worktree, not in the training process's worktree. VERIFIED_FROM_SOURCE
- Exactly one new file created (this report). No existing file modified. No commit, push, merge, rebase, cherry-pick, or PR in this task. VERIFIED_FROM_SOURCE
- Checkpoint access note: Checkpoint B is not git-tracked, so it was read via its absolute path in the canonical worktree (`/home/minhtt/Neups_workshop/reproduction/workspace_prichexy/networks/generator_lowest_total_loss_mu_0.01.pth`). Read-only; SHA verified unchanged before/after. Checkpoint A was read inside the review worktree (tracked file, identical blob to canonical). VERIFIED_BY_CPU_SEMANTIC_AUDIT

---

## 3. GENERIC GOVERNANCE CORRECTION

The following permitted statement supersedes the previous report's prohibited-material passages:

> The previous P0–P1 report improperly referenced prohibited frozen-evaluation material in parts of Sections 2 and 5. Those references and all conclusions depending on them are retracted. No underlying prohibited resource was opened or inspected in this erratum task. The user-reported approximately 0.60 value is retained only as unverified contextual motivation and is not treated as repository-verified evidence.

Explicitly superseded, without restating their content:
- previous report §2.1, only where prohibited material was listed;
- previous report §5.0.1, prohibited-evaluation evidence;
- previous report §5.2 item 5, any proposed prohibited cross-check.

P0 must use only authorized TRAIN/VAL resources. GOVERNANCE_CORRECTION

---

## 4. CPU-ONLY CHECKPOINT AUDIT METHOD

Environment and safety (all verified in-run):
- `CUDA_VISIBLE_DEVICES=""`; torch 2.7.0+cu128; `map_location="cpu"`; `torch.load(..., weights_only=True)` succeeded for BOTH files on first attempt (no unrestricted-pickle fallback needed or attempted).
- `torch.cuda.is_initialized()` asserted `False` immediately before loading and verified `False` after all comparisons.
- No model instantiation, no forward pass, no optimizer, no image/dataset loading, no metric computation, no temporary checkpoint conversion, no repo output other than this report.
- File SHA-256 recorded before and after the audit; both unchanged. VERIFIED_BY_CPU_SEMANTIC_AUDIT

Normalization applied: both containers were recognized as raw string-key→tensor mappings (no wrapper key); no `module.` prefix present on either side, so none removed; no keys renamed or discarded. VERIFIED_BY_CPU_SEMANTIC_AUDIT

---

## 5. TENSOR-LEVEL CHECKPOINT RESULTS

| # | Quantity | Checkpoint A | Checkpoint B |
|---|---|---|---|
| 1 | File SHA-256 | `101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb` | `4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153` |
| 2 | File size | 31,112,853 bytes | 31,112,853 bytes |
| 3 | Container type | raw mapping str→tensor | raw mapping str→tensor |
| 4 | Normalized key count | 118 | 118 |
| 5 | Key-set equality | YES (exact) | |
| 6 | Missing keys | none in A vs B; none in B vs A | |
| 7 | Shape equality per shared key | YES (all 118) | |
| 8 | Dtype equality per shared key | YES (all 118) | |
| 9 | Total tensor count | 118 shared | |
| 10 | Total elements | 7,768,404 | |
| 11 | Exact `torch.equal()` | FALSE for every shared tensor | |
| 12 | Different tensors | 118 / 118 | |
| 13 | Different elements | 7,768,404 / 7,768,404 | |
| 14 | Max abs diff (float tensors, n=100) | 3.6720677614212036 | |
| 15 | Mean abs diff (over all elements) | 0.005152789663339106 | |
| 16 | Integer/Boolean exact-diff elements | 18 (= the 18 `num_batches_tracked` buffers) | |
| 17a | BN `running_mean` buffers | 0 equal / 18 different | |
| 17b | BN `running_var` buffers | 0 equal / 18 different | |
| 17c | BN `num_batches_tracked` buffers | 0 equal / 18 different | |
| 18 | Canonical semantic SHA-256 | `6bd60726411b69cf35099688dbdcad01ee74b9619cee41a75dacab43777b4d14` | `764be7490cbaed55e5d00797096901ecc6920df1ca9804ee17892fb1e41fc7a5` |

No tensor contents printed. VERIFIED_BY_CPU_SEMANTIC_AUDIT

**Classification: `SEMANTICALLY_DIFFERENT`.** Every parameter, every BatchNorm running statistic, and every `num_batches_tracked` counter differs; the two files represent two different generator training states, not serialization variants of one state. The differing `num_batches_tracked` values indicate different accumulated update exposure. VERIFIED_BY_CPU_SEMANTIC_AUDIT

---

## 6. ANCHOR U IDENTITY / PROVENANCE DECISION

Read-only Git provenance audit (no privacy AUC used anywhere in this decision):
- Checkpoint A path `networks/pretrained_generator_prichexy_net.pth` exists at the upstream anchor commit `29245d1` itself ("Pushed code", 2023-10-05) as regular blob `0f199e83b79bf71fb50699bc2146d5eaa9aab998`, byte-identical to today's working-tree copy (`git rev-parse HEAD:…` = same blob). VERIFIED_FROM_SOURCE
- Checkpoint B's content hash `4d82dcdd…` and size 31,112,853 exactly match the LFS pointer of `networks/generator_lowest_total_loss_mu_0.01.pth`, which has existed since the same upstream anchor commit `29245d1`. The local workspace copy is therefore the materialized (LFS-smudged) form of an upstream-released LFS object. VERIFIED_FROM_SOURCE
- A similarly named but distinct archived checkpoint added later (`archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth`, commit `4bb0d28`, LFS oid `01287c0c…`, size 31,116,675) is NOT either candidate and plays no role here. VERIFIED_FROM_SOURCE

**Provenance classification: `PROVENANCE_RESOLVED`** — both candidates link directly to the upstream release; they are two different released generator states under two names.

**Anchor U designation (by governed current code, not historical performance):** `research_agent/m2_dev/evaluator_common.py:56–57` defines and SHA-gates `INITIAL_GENERATOR_PATH = networks/pretrained_generator_prichexy_net.pth` (SHA `10122689…`) as the certified pipeline's canonical initial generator. Therefore:

```text
Anchor U := networks/pretrained_generator_prichexy_net.pth (SHA 10122689…)   [Checkpoint A]
Recorded variant: reproduction/workspace_prichexy/networks/generator_lowest_total_loss_mu_0.01.pth
                  (SHA 4d82dcdd…; upstream-linked LFS object; SEMANTICALLY_DIFFERENT from Anchor U)
```

Residual descriptive question (does NOT gate P0): which of the two released states corresponds to the paper-era evaluation context behind the user-reported ≈0.60 value cannot be determined from code alone. UNRESOLVED (contextual only). Any future comparison against the B variant is a separate human decision and is not authorized here.

**P0 anchor status: RESOLVED.**

---

## 7. CORRECT P0 DELTA SIGN

Definition unchanged:

```
Δ_bridge,s = AUC_U,s − AUC_B,s
```

- Δ < 0 ⇒ Anchor U has LOWER Re-ID AUC ⇒ **better privacy for Anchor U**.
- Δ > 0 ⇒ Anchor B has LOWER Re-ID AUC ⇒ **better privacy for Anchor B_dev**.

The previous report's Outcome "P0-B" stated the direction incorrectly and is retracted in full, replaced by §8 below. MATHEMATICAL_CORRECTION

---

## 8. CORRECT EQUIVALENCE AND DIRECTIONAL DECISION TABLE

Margin: δ = 0.03, labeled `PROVISIONAL_SEOI_PENDING_HUMAN_APPROVAL`. Reusing the earlier non-inferiority margin does not automatically make it the correct smallest effect of interest for this bridge; it is inherited for continuity and must be confirmed or replaced by a human BEFORE execution. It must never be tuned after seeing P0 results. MATHEMATICAL_CORRECTION / GOVERNANCE_CORRECTION

Analysis basis: paired attacker-seed deltas (n=26 planned), conditional on the two fixed generators; attacker seeds are optimization replications, not generator-level method replications.

| Outcome | Predeclared statistical requirement | Interpretation |
|---|---|---|
| **A — Practical equivalence** | TOST at α = 0.05 passes; equivalently the **90% CI** for mean paired Δ lies entirely inside [−0.03, +0.03]. Two-sided 95% CI also reported descriptively. Overlap with an arbitrary baseline-AUC band is NOT acceptable evidence. | Under the authorized locked TRAIN/VAL protocol, the two fixed generator checkpoints are practically equivalent within the provisional margin. The user-reported ≈0.60 value is protocol-dependent or otherwise not transportable into this protocol. |
| **B — Anchor U meaningfully better for privacy** | Δ_bridge < −0.03 with the predeclared confidence-bound rule excluding −0.03 in the appropriate direction (one-sided 95% upper bound < −0.03). | B_dev shows a meaningful privacy regression relative to Anchor U under the same authorized protocol. |
| **C — Anchor B meaningfully better for privacy** | Δ_bridge > +0.03 with the predeclared confidence-bound rule excluding +0.03 in the appropriate direction (one-sided 95% lower bound > +0.03). | B_dev meaningfully improves privacy relative to Anchor U under the same authorized protocol. |
| **D — Inconclusive** | Equivalence not established AND neither directional rule established; or protocol equivalence fails; or Anchor U identity unresolved; or required data/provenance unavailable. | No baseline comparison or method-regression claim is authorized. |

MATHEMATICAL_CORRECTION

---

## 9. FIVE-SEED SCREEN LIMITATION

The five-seed P0 screen may check ONLY: runtime feasibility; manifest correctness; paired-seed execution; obvious numerical or protocol failure; gross unexpected effects. It must NOT issue an equivalence, regression, or improvement verdict. Only the predeclared full paired bridge (26 seeds) may produce a conditional fixed-generator P0 conclusion. Attacker seeds remain conditional optimization replications, not generator-level method replications. GOVERNANCE_CORRECTION

---

## 10. CORRECT GIT COMMIT TOPOLOGY

Verified read-only via parent-revision and merge-base commands:

```text
34bca1f74662275ced0418ea183a3d9d5ef81f88
    ├── a3513e5c3b3b5631838399fc14c9e708909fe923   (parent = 34bca1f, verified)
    └── c2ee268d3d9af1f1a5b589a040db178b46b17763   (parent = 34bca1f, verified)
```

Corrections:
- `a3513e5` is exactly ONE commit ahead of `34bca1f`;
- this is NOT evidence that the canonical method history had diverged;
- `c2ee268` is a separate sibling documentation commit created from the same base;
- integrating both documentation commits is a future human Git-curation decision, deliberately NOT performed in this task.

The previous report's claim of "unexplained divergent histories" between the strategy-report commit and the locked method tip is RETRACTED. VERIFIED_FROM_SOURCE

---

## 11. P1a — FROZEN-GENERATOR CRITIC BUDGET/GEOMETRY PROBE

Status: `DESIGN_READY_BUT_NOT_AUTHORIZED`.

P1a may study, against frozen generator anchors only:
- fresh-critic optimization trajectories;
- short-versus-long budget taken from the SAME trajectory;
- anon/anon versus anon/real training geometry;
- geometry transfer (train×eval matrix);
- critic-seed variability.

P1a can test budget and geometry mechanisms. **It cannot prove that the historical live critic was weak**, because no live-critic reference exists (§12). PROPOSED_DESIGN

### Short-budget correction
The previously proposed five attacker epochs are NOT "live-equivalent." Two distinct quantities must be distinguished and neither reproduces live co-adaptation against a moving generator:
- `evaluation-attacker early-budget proxy` — e.g., five attacker-loader epochs derived from Stage-A best-epoch statistics;
- `live-verifier exposure-matched budget` — matched to the certified verifier's cumulative updates (≈625 updates/epoch × selection epoch).

The exact P1a budget choice requires human approval and separate justification. GOVERNANCE_CORRECTION

### Statistical limitation
Three critic seeds per cell are mechanism-screening replication ONLY. Rules such as "two of three seeds exceed 0.03" or "eight of twelve differences are positive" must not be presented as confirmatory statistical evidence. Required reporting: complete per-seed effects; paired within-trajectory differences; uncertainty intervals; effect sizes; explicit screening-only language. GOVERNANCE_CORRECTION

---

## 12. P1b — HISTORICAL LIVE-CRITIC-GAP TEST

Retained verified finding: there is NO paired live-verifier checkpoint corresponding to the selected generator anchors (the certified runner persists only generator state at the selected epoch; resumable state is unit-test-only). Consequently the quantity

```
G_live = AUC_fresh_converged − AUC_historical_live_critic
```

cannot currently be computed. BLOCKED

Rules for the corrected framing:
- Do NOT substitute the critic-initialization checkpoint and call the result a live-critic gap.
- Do NOT describe the initialization comparison as a lower or upper bound on the unavailable live gap. It is only a different descriptive quantity.

Status: `BLOCKED_NO_LIVE_CRITIC_CHECKPOINT`.

A valid P1b would require a separately authorized 1:1 control co-training trajectory that:
1. reproduces the canonical B_dev training semantics;
2. saves verifier snapshots at predeclared updates/epochs;
3. saves the corresponding generator snapshots;
4. preserves all governed invariants (mode-topology/buffer-state guards around any frozen-generator forward);
5. uses a fixed external analysis protocol.

This is NOT a micro-run. Provisional budget: approximately ONE FULL B_dev generator trajectory plus critic evaluations — anchored to the measured certified B_dev runtime of 31.57 GPU-hours (M2_S1_C4_RESULT.md §1) — subject to measured timing before authorization. The currently active confounded Direction B process must NOT be reused for this purpose. PROPOSED_DESIGN / BLOCKED

---

## 13. CORRECTED COST IMPLICATIONS (planning estimates, measurement-anchored)

| Item | Corrected figure | Basis |
|---|---|---|
| One attacker trajectory | ≈ 0.336 GPU-h | measured mean 1209.5 s over 50 completed runs (previous report §2.6; unaffected finding) |
| P0 5-seed screen (both anchors, diagnostic-only) | ≈ 3.4 GPU-h | 2 × 5 × 0.336 |
| P0 26-seed bridge (both anchors) | ≈ 17.5 GPU-h | 2 × 26 × 0.336 |
| Dual-anchor GPU evaluation justified? | Single-anchor designation (§6); evaluating the B variant would be an ADDITIONAL optional arm (~+8.7 GPU-h per 26 seeds) requiring separate human approval | governance correction |
| P1a factorial (12 fresh-critic trajectories) | ≈ 4–5 GPU-h expected; quoted 4.2–10.5 h range retained as planning bound | measured attacker-runtime distribution; short/converged checkpoints share trajectories |
| P1b live-gap enablement | ≈ 31.6 GPU-h + snapshot-evaluation overhead (NOT AUTHORIZED, not a micro-run) | measured certified B_dev trajectory 31.57 h |

All figures are planning estimates, not runtime guarantees. MATHEMATICAL_CORRECTION / PROPOSED_DESIGN

---

## 14. HUMAN APPROVAL CHECKLIST

- [ ] Confirm or replace the provisional SEOI δ = 0.03 BEFORE any P0 execution (no post hoc tuning).
- [ ] Ratify Anchor U = Checkpoint A per §6 (and decide whether the upstream-linked B variant ever gets a descriptive secondary arm).
- [ ] Approve the locked P0 protocol manifest (20 lock fields of the previous report, §5.2 items 1–20 excluding the superseded item 5 cross-check; primary metric on authorized VAL pairs only).
- [ ] Authorize P0 screen (diagnostic-only) then full bridge.
- [ ] Approve P1a design including an explicitly justified short-budget definition (evaluation-attacker proxy vs exposure-matched choice).
- [ ] Decide whether to authorize P1b's full snapshot-capturing control trajectory (~31.6 GPU-h class).
- [ ] Approve external-selector design before any causal ablation arms.
- [ ] Curate the sibling documentation commits (integrate `a3513e5` and `c2ee268` lineage) as a human Git decision.
- [ ] Confirm segmentation remains BLOCKED until a certified evaluator exists.
- [ ] Confirm the standing prohibitions: no post-hoc score flipping, no post-hoc threshold edits, no use of the active confounded k3 run as causal evidence.

---

## 15. MACHINE-READABLE FINAL VERDICT

```text
ERRATUM_STATUS: PASS
PREVIOUS_REPORT_STATUS: SUPERSEDED_ON_LISTED_POINTS
PROHIBITED_EVALUATION_REFERENCES: RETRACTED
PROHIBITED_RESOURCE_ACCESSED_IN_THIS_TASK: NO

CHECKPOINT_A_FILE_SHA_UNCHANGED: YES
CHECKPOINT_B_FILE_SHA_UNCHANGED: YES
CUDA_INITIALIZED: NO
MODEL_INSTANTIATED: NO
MODEL_FORWARD_EXECUTED: NO

ANCHOR_U_FILE_IDENTITY: BYTE_DISTINCT
ANCHOR_U_SEMANTIC_IDENTITY: SEMANTICALLY_DIFFERENT
ANCHOR_U_PROVENANCE: RESOLVED
P0_ANCHOR_STATUS: RESOLVED

P0_DELTA_DEFINITION: AUC_U_MINUS_AUC_B
P0_SIGN_RULE: CORRECTED
P0_EQUIVALENCE_RULE: TOST_ALPHA_0_05
P0_SCREEN_INFERENCE_STATUS: DIAGNOSTIC_ONLY
P0_PROTOCOL_SPECIFICATION: DESIGN_READY
P0_EXECUTION_AUTHORIZATION: NONE

P1A_FROZEN_CRITIC_FACTORIAL: DESIGN_READY_NOT_AUTHORIZED
P1B_LIVE_CRITIC_GAP: BLOCKED_NO_LIVE_CRITIC_CHECKPOINT
WEAK_LIVE_CRITIC_HYPOTHESIS: UNTESTED
DIRECTION_B_V1_CAUSAL_EVIDENCE: INVALID_REFERENCE_ONLY

GIT_TOPOLOGY_STATEMENT: CORRECTED
CURRENT_VAL_CONFIRMATORY_STATUS: NOT_LOCKED
PATIENT_GRAPH_BOOTSTRAP_STATUS: UNVALIDATED
SEGMENTATION_EXECUTION_STATUS: BLOCKED

GPU_AUTHORIZATION: NONE
GENERATOR_TRAINING_AUTHORIZATION: NONE
ATTACKER_TRAINING_AUTHORIZATION: NONE
FILES_CREATED: 1
EXISTING_FILES_MODIFIED: 0
CURRENT_ERRATUM_TASK_COMMIT_OR_PUSH: NONE
ACTIVE_TRAINING_PROCESS_TOUCHED: NO
NEXT_REQUIRED_ACTION: EXTERNAL_HUMAN_REVIEW
```

Previous report commit: `c2ee268d3d9af1f1a5b589a040db178b46b17763`
Previous report status: `SUPERSEDED_ON_GOVERNANCE_REFERENCES, P0_SIGN, P0_EQUIVALENCE_RULE, ANCHOR_U_IDENTITY, GIT_TOPOLOGY, AND_P1_LIVE_GAP_STATUS`
Previous report commit status: `COMMITTED_AND_PUSHED`
Current erratum task commit/push: `NONE`

Unaffected findings that remain usable: Direction B v1 BatchNorm drift; same-batch repetition; method-dependent checkpoint selection; Direction B v2 design-only status; attacker-seed versus generator-seed distinction; current VAL not being an untouched confirmation resource; patient bootstrap remaining unvalidated; segmentation-coordinate requirements.

*End of erratum. Do not begin P0, P1a, or P1b without explicit external human approval.*
