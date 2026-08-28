# ADDENDUM A to V2_CONFIRMATION_PREREGISTERED_HYPOTHESES_2026-08-26

**Date:** 2026-08-28
**Amends:** `reproduction/reports/V2_CONFIRMATION_PREREGISTERED_HYPOTHESES_2026-08-26.md`
(hereafter "the V2 prereg"), which states: *"This file must not be edited after
unblinding; corrections go into a dated addendum."* That file is unmodified.
Also amends the interpretive gloss (not the numeric classification) of
`P0_3_PREREGISTERED_IM2_HYPOTHESES_2026-08-23.md` on branch
`review/p0-runner-attacker-loop-20260823`.

**Disposition recorded here:** the V2 confirmation is **ABANDONED BEFORE
EXECUTION**. It is not classified, because its candidate was invalid for a
reason discovered before the confirmation ran. The successor experiment is
preregistered separately in
`PHASE_B_CORRECTED_OBJECTIVE_PREREGISTRATION_2026-08-28.md`.

---

## 0. EXECUTION STATUS AT TIME OF ABANDONMENT

The V2 confirmation protocol was **never executed**. Verified 2026-08-28:

- The §2 nested pools (`P_gen_train`, `P_gen_select`, `P_att_train`,
  `P_att_select`, `P_confirm`) **do not exist**; `ls image_pairs/` contains only
  the three legacy upstream files.
- No confirmation run directory exists under any `runs_*` root.
- Therefore **no confirmatory outcome variable was ever observed**, and nothing
  in the V2 prereg was unblinded in its own terms.

Development-stage numbers *were* observed. They are disclosed in full in §5 so
that the successor preregistration cannot be accused of being written to fit
them.

---

## 1. CORRECTION 1 — the candidate architecture was inert (fatal to §1)

The V2 prereg §1 selects the confirmation candidate as the argmin of
development-stage privacy VAL AUC over `{run1 attention, arm acc1, arm acc2}`
(plus Direction C if complete). Two of those arms are attention runs.

**The attention mechanism was never active in any of them.**

`AttentionGate.__init__` zero-initialized `W_g`, `W_x` **and** `psi.weight`
simultaneously. That is a gradient fixed point, not an initialization:

```
a = relu(W_g(g) + W_x(x)) = relu(0) = 0
dL/d(psi.weight)  proportional to  a           = 0   -> psi.weight pinned at 0
dL/da             proportional to  psi.weight  = 0   -> W_g, W_x get NO gradient
```

so no gate weight can ever leave zero, and each gate degenerates to the single
learnable scalar `sigmoid(psi.bias)`.

**Confirmed empirically, not inferred.** In the completed 250-epoch run
`archive/v2_attention_feat1_Uinit_run1/generator_lowest_total_loss.pth`, all
87,000+ entries of every `W_g`/`W_x`/`psi.weight` tensor are still exactly
`0.0`; only the four `psi.bias` scalars moved (6.0 -> 5.953, 6.017, 6.061,
6.076). Fix and verification: commit `10d9212`.

**Consequences for the record:**

1. Every completed "V2 attention" run is, in fact, a plain U-Net with four
   learnable scalars (~0.9974) on its skip connections. The architectural
   hypothesis the V2 prereg was written to test **was never tested**.
2. The §1 argmin would therefore have selected among mislabeled arms. Running
   the confirmation as written would have produced a correctly-executed test of
   the wrong object.
3. Existing V2 results are **not** evidence that attention fails to help. Any
   future attention claim requires retraining under `10d9212` or later.
4. The V2 prereg's own §1 sanity apparatus could not have caught this:
   `_initial_gate_sanity_check` verified gates ~= sigmoid(6) and structural
   parity with the plain U-Net, **both of which PASS in the dead state**, and
   printed `Init sanity OK`. A gradient-liveness check was added in `10d9212`
   and regression-tested against the old init (20/24 tensors dead).

---

## 2. CORRECTION 2 — anchor citation error (§1)

The V2 prereg §1 describes the anchor as *"the released upstream generator used
by the P0 screen (mean screen AUC 0.6336 ± 0.0551)"*.

`0.6336 ± 0.0551` is **the n=5 early screen subset (seeds 42–46)**, not the P0
screen as a whole. Recomputed independently from the sealed run manifests in
`reproduction/p0_bridge/runs_screen/`:

| Subset | n | mean U_PUBLISHED AUC | SD |
|---|---:|---:|---:|
| seeds 42–46 (early screen; the figure cited) | 5 | 0.6336 | 0.0551 |
| seeds 47–67 (**declared PRIMARY**, P0.3 §3) | 21 | **0.7139** | 0.0520 |
| seeds 42–67 (full, secondary/descriptive) | 26 | 0.6985 | 0.0608 |

The cited figure is the **lowest of the three**, in a document whose §8 commits
to anti-anchoring. This was a citation error, not an analysis error: the V2
prereg's comparisons are paired and within-harness, and §1 explicitly states
that "absolute cross-split band transfer is NOT assumed", so no decision rule
consumed the wrong number. It is corrected here so no successor document
inherits it.

**Binding for all successor work:** the P0 U_PUBLISHED anchor is **0.7139
(n=21, seeds 47–67)** as primary, 0.6985 (n=26) as descriptive. The 5-seed
figure must not be cited as "the P0 screen".

The same 5-seed bands appear as the reference bands in P0.3 §1, where they are
correctly labelled `(from already-unblinded screen, n=5 each)`. That labelling
is accurate and is not amended.

---

## 3. CORRECTION 3 — the H-SIBLING interpretive gloss inverts the conclusion

P0.3 §2 classifies the I_M2 diagnostic by the numeric rule
`|M − 0.8275| ≤ 0.03`. With `M = 0.8366` (n=10, dist 0.0091) the rule selects
**H-SIBLING**. **That numeric classification was applied correctly and is NOT
amended.**

What is amended is the prose attached to that label:

> "H-SIBLING supported — the upstream release contains two generator states with
> drastically different privacy; D_BDEV faithfully inherited its weak init;
> **no fine-tuning blame**"

The final clause does not survive a provenance fact established afterwards
(§4 below): `pretrained_generator_prichexy_net.pth` (`10122689…`), the I_M2
anchor, is the **pre-adversarial** output of upstream README §1 pre-training —
a generator that has had **no** privacy training at all.

Read with that fact, the same numbers say the opposite of "no fine-tuning
blame":

| Generator | Adaptive Re-ID AUC (P0 harness) |
|---|---:|
| I_M2 — pre-adversarial init | 0.8366 (n=10) |
| D_BDEV — after the full certified adversarial training | 0.8244 (n=26) |
| U_PUBLISHED — upstream's own released final model | 0.6985 (n=26) |

The project's entire adversarial privacy phase moved the metric by **−0.011**,
while upstream's own training of the same architecture from the same init moved
it by **−0.138**. The fine-tuning path is therefore squarely implicated. The
correct reading is *"our adversarial training produced essentially no privacy"*,
not *"the init was weak, so training is exonerated"*.

**Downstream effect.** P0.3 §5 maps H-SIBLING to *"adopt C1 (fine-tune-from-U)
as primary method candidate"*. That action was derived from the inverted gloss.
It was in fact attempted (`v2_attention_feat1_Uinit_run1`, initialized from
U_PUBLISHED) and came out **worse** than the anchor — see §5. Successor work
must not treat "fine-tune from U" as a preregistered-favoured route on the
strength of the H-SIBLING label.

---

## 4. CLOSURE — blocker B-1 (Anchor U identity) is RESOLVED

`P0_P1_PREEXPERIMENT_PROTOCOL_REVIEW_2026-08-21.md` §1 item 5 and §5.0.1 record
B-1 as an **UNRESOLVED** execution blocker: two byte-distinct artifacts
(`10122689…` and `4d82dcdd…`) are both described in project documents as the
released upstream generator, and "Anchor U identity must be fixed before any P0
run."

B-1 is resolvable from the upstream tree and is closed here. Both files are
tracked in upstream commit `29245d1` on branch `original-upstream`, and the
upstream README distinguishes them explicitly:

- **§1 "Pre-training of the flow field generator"** produces
  `pretrained_generator_prichexy_net.pth` (`10122689…`) — the **pre-adversarial**
  generator. The certified pipeline uses it as `INITIAL_GENERATOR_PATH`.
- **README lines 55 and 75** instruct that evaluation uses
  `perturbation_model_file = "./networks/generator_lowest_total_loss_mu_0.01.pth"`
  (`4d82dcdd…`) — the **released, fully adversarially-trained** model.

**Determination: Anchor U = `4d82dcdd…` is correct as executed.** The P0 screen
and bridge used it (`runs_screen/U_PUBLISHED/*/run_manifest.json`,
`generator_sha256: 4d82dcdd…`), and the independent 10-seed upstream-protocol
reproduction used the same bytes
(`FINAL_10SEED_PRICHEXY_REPRODUCTION.md` §6). No P0 result requires revision.

I_M2 (`10122689…`) is correctly understood as the **pre-adversarial anchor**,
which is what §3 above turns on.

---

## 5. FULL DISCLOSURE OF DEVELOPMENT-STAGE NUMBERS ALREADY OBSERVED

Required so that the successor preregistration is honest about what its author
already knew. All are development-stage, P0 harness, fold=val, raw ROC AUC.

| Arm | Checkpoint | AUC | n seeds |
|---|---|---:|---:|
| V2_UINIT | `generator_lowest_total_loss.pth` (`2f285743`) | 0.7258 | 3 |
| V2_UINIT_VER | `generator_lowest_ver_loss.pth` (`9796bf3d`) | 0.6969 | 3 |
| U_PUBLISHED anchor | `4d82dcdd` | 0.6985 / 0.7139 | 26 / 21 |
| D_BDEV | `18381d92` | 0.8244 | 26 |
| I_M2 | `10122689` | 0.8366 | 10 |

Also observed (`privacy_objective_diagnosis.json`, commit `09046f7`): the
co-adapted training verifier reaches AUC 0.9147 on generator `2f285743` where
fresh adaptive attackers reach 0.7258; and the logged `ver_loss` varies 9x
(0.034 -> 0.309) across four checkpoints whose true AUC stays pinned near 0.90.

**Two hazards this creates, declared now:**

1. **Direction of the V2 candidate is known.** Both V2 arms are *above* the
   anchor (+0.091 and +0.062 paired on seeds 42–44), i.e. heading toward
   H-NOT-SUPPORTED under the V2 prereg's own rules. Abandonment in §0 is
   therefore recorded as being for the reason in §1 (inert architecture),
   **and** with the disclosure that the candidate was also losing on the
   numbers. Both facts belong in the record; neither is offered as the sole
   reason.
2. **Selection multiplicity.** `generator_lowest_ver_loss.pth` (0.6969) scores
   better than the prereg-locked `generator_lowest_total_loss.pth` (0.7258).
   The V2 prereg §1 locked `lowest_total_loss`. **Switching to the ver-selected
   checkpoint post hoc is forbidden** and is not done anywhere.

---

## 6. CODE DEFECTS AFFECTING PRIOR ARTIFACTS

Recorded so no prior number is silently re-used. Full detail: commit `10d9212`.

| # | Defect | Artifacts affected |
|---|---|---|
| 1 | Dead-gradient attention init | every V2 attention run to date |
| 2 | Sanity check could not detect #1 (passed and printed OK) | as above |
| 3 | Provenance manifest recorded the DEFAULT generator's SHA regardless of the file actually loaded | `v2_attention_feat1_Uinit_run1/v2_provenance_manifest.json` reports `path=…mu_0.01.pth` with `sha256=10122689…` — path and hash contradict, across exactly the two checkpoints §3–§4 turn on |
| 4 | AC critic left in `train()` across accumulation micro-batches | `config_anonymization_v2ctrl_b16acc2.json` (`accumulation_steps: 2`) only; acc1 arms unaffected |
| 5 | Frozen anonymizer not `requires_grad=False` in the V2 attacker | wasted compute only; no trained weights affected |

**Not affected:** `research_agent/m2_dev/anonymizer_runner.py`, the certified
B_dev/C4 path, is untouched by all five (last modified by `c5bc118`, unrelated).
Defects 1–2 are confined to `UNetAtt`; 4 requires `accumulation_steps > 1`,
which the certified config does not use. **No certified S1 or P0 result requires
revision on account of these defects.**

---

## 7. WHAT THIS ADDENDUM DOES NOT CHANGE

- No numeric classification is altered. P0.3's H-SIBLING selection stands;
  only its prose gloss is corrected (§3).
- No P0 screen, bridge, I_M2, batch-diagnostic or utility measurement is
  altered, recomputed, or withdrawn.
- No decision threshold is edited. δ = 0.03 remains frozen.
- The V2 prereg file itself is unmodified.

*End of Addendum A. Corrections to this addendum require Addendum B, dated.*
