# EXECUTION PLAN — InfPriv @ NeurIPS 2026 submission

**Created:** 2026-08-29. **Deadline:** 2026-09-08, 12:00 UTC (**10 days**).
**Venue:** InfPriv — *Beyond Private Training: The New Landscape of AI Privacy*.
**Target topic:** 06, *Benchmarks and Evaluation*. Secondary: 05.

Working rule for the whole plan: **the paper does not depend on any experiment
that has not already finished.** Everything still running is upside.

---

## 0. THE CONTRIBUTION, IN ONE SENTENCE

> An adversarial privacy objective that is a monotone function of a
> discriminator's logit optimizes the adversary's **calibration**, not its
> **discriminability**; since privacy is reported with a ranking metric, such an
> objective is invariant to the quantity it is credited with improving.

The affected form, `−log(1 − σ(z)) = softplus(z)`, is the standard
non-saturating GAN generator loss, so the statement reaches any obfuscation
method that fools a discriminator with a confidence loss and reports a retrained
attacker's AUC.

**The deliverable that makes this topic 06** is a three-test audit any such paper
can run:

| Test | Cost | What it answers | Our result |
|---|---|---|---|
| **T1 Invariance** | pencil | is the objective monotone in the logit? | yes → AUC-invariant |
| **T2 Decoupling** | inference only | do the logged metric and true AUC move together? | no: 9x vs pinned at ~0.90 |
| **T3 Best-response gap** | one attacker | is the training critic weaker than the eval attacker? | **no, stronger**: −0.189 |

T3 matters because the intuitive answer is wrong, and two method directions in
this project were built on the wrong answer.

---

## 1. WHAT IS ALREADY DONE (no further GPU needed)

| # | Result | Artifact |
|---|---|---|
| A | Objective is AUC-invariant; `L_priv` 1.058 → 0.000000 with AUC fixed at 0.9808 | `privacy_objective_diagnosis.json` |
| B | Logged metric spans 9x (0.034→0.309) while true AUC stays 0.892–0.915 | same |
| C | Best-response gap −0.189 (co-adapted 0.9147 vs fresh 0.7258) | same |
| D | Published privacy curve shifts **+9.3 / +12.3 / +12.1** under an adaptive attacker | `mu_curve_summary.json` |
| E | Utility reproduces with a **constant +1.0** offset at every row | `utility_results*.json` |
| F | Four independent interventions, none improving | P0 manifests, Stage-A pilot |
| G | Objective **self-terminates**: gradient 2.7e-8 at mean logit −17.42, while separability gap is still 0.4392 | pre-flight, commit `dadeef9` |
| H | Deformation is uniform (1.08x) while attacker saliency is concentrated (1.89x); spatial overlap 1.03 ≈ independent | saliency measurement |

D+E together are the load-bearing pair: same pipeline, same checkpoints, same
fold, same evaluator — utility reproduces, privacy does not.

---

## 2. SCHEDULE

| Date | GPU | Writing |
|---|---|---|
| **Aug 29** | Phase B2 arm A running (~5.5 h) | Sections 1–3 drafted |
| **Aug 30** | **P1: protocol-vs-fold decomposition (~3 h)** → then arm B (~9 h) | Sections 4–5 |
| **Aug 31 – Sep 1** | evaluate B2 arms on P0 (~4.3 h/arm) | Results assembled |
| **Sep 2** | **HARD STOP — all GPU work ends** | Full draft complete |
| **Sep 3–5** | none | Revision; every number re-verified against its artifact |
| **Sep 6–7** | none | Internal review, buffer |
| **Sep 8** | none | Submit |

The Sep 2 hard stop is declared in `PHASE_B2_ADDENDUM_A` §4 and is not
negotiable by a good result.

---

## 3. PRIORITY CHANGE — DEFENCE BEFORE ATTACK

**P1 (protocol-vs-fold decomposition) is promoted ahead of arm B.**

Our adaptive numbers are VAL; the paper's are TEST. The +9…+12 shift therefore
mixes *protocol* with *fold*, and we have not separated them. A reviewer can say
"your effect is a fold artifact" and the headline collapses.

Fix: run the **upstream protocol on the VAL fold** (retrained SNN, 5 seeds,
~3 GPU-h, TEST untouched). Same fold, only the protocol differs.

- lands near 60 → the whole shift is protocol. Claim is strong.
- lands near 68 → most of it is fold. Claim weakens **and we learn it before
  submitting rather than in review.**

Rationale for the reorder: if arm B fails we lose nothing we already have; if
the fold confound stays open we lose the main contribution. Defence first.

---

## 4. STEPS, WITH EXIT CRITERIA

**S1 — Phase B2 arm A** *(running)*
Exit: futility at epoch 25 (~2.7 h) or completion at epoch 50.
Epoch-0 signal: `gap` 0.4392 → 0.0929, co-adapted AUC 0.5222.
**This is a diagnostic and is explicitly non-decisive** (prereg §4). It cannot
distinguish "identity removed" from "this one critic fooled". Only the P0 fresh
attacker decides. Do not report it as a result.

**S2 — P1 decomposition** *(~3 h, after S1)*
Upstream protocol, VAL fold, 5 seeds, on `U_PUBLISHED`.
Exit: a number. Either outcome is reported.

**S3 — Phase B2 arm B, w=3.0** *(~9 h)*
Runs regardless of arm A's outcome (Addendum A §2: futility is per-arm).
Skipped only if the Sep 2 stop arrives first.

**S4 — P0 evaluation of surviving B2 arms** *(~4.3 h/arm, screen seeds 42–46)*
Anchor: `U_PUBLISHED` paired on the same seeds, where it averages **0.6336** —
a harder bar than its 26-seed mean of 0.6985. n=5 gives a smallest detectable
effect of ≈ −0.041.
Primary arm chosen **on utility**, never on privacy (prereg §5).

**S5 — Mechanism confirmation** *(~0 GPU, only if a B2 arm wins)*
Re-run the saliency measurement on the winning checkpoint. The diagnosis
predicts the corrected objective reallocates deformation toward identity-bearing
pixels. Confirming it turns a number into a mechanism; failing to confirm it is
reported too.

**NOT DONE before the deadline, declared now:** Stage 2 (n=26) and the TEST
confirmation. A positive screen does not buy them.

---

## 5. FOUR STATEMENTS THAT GO NEAR THE TOP OF THE PAPER

Transparency is a design requirement here, not a virtue signal: the paper
indicts selective reporting, so selective reporting would discredit it.

1. **We do not claim their number is wrong.** 57.7 is correct for their threat
   model. We reproduced that same checkpoint under that same protocol at
   **60.80 ± 4.35**, inside their own ±4.0.
2. **Every comparison uses 60.80, never 57.7.** The +3.1 is an environment
   offset; comparing a number produced here against their published value is a
   cross-environment comparison.
3. **The utility control sits beside the privacy table**, not in an appendix.
   Constant +1.0 versus divergent +9…+12 is what rules out a broken pipeline.
4. **The fold confound is stated by us, not extracted by a reviewer** — with
   S2's result if we have it, as an open limitation if we do not.

---

## 6. PAPER SKELETON

1. **Introduction** — privacy is reported with a ranking metric; a common class
   of objectives cannot see it.
2. **The invariance** — proposition, numerical demonstration (A), and that the
   affected form is the standard GAN loss.
3. **The audit** — T1/T2/T3 as a reusable procedure; results B and C.
4. **Case study** — the published curve under an adaptive attacker (D), with the
   utility control (E) alongside.
5. **Mechanism** — self-termination (G) and misallocated deformation (H); four
   interventions that do not help, and why (F).
6. **A corrected objective** — *conditional on S4*; otherwise folded into §5 as
   an attempted fix with its measured outcome.
7. **Limitations** — one generator seed; one attacker family; VAL selection and
   scoring share a pool; fold confound status; screen-level power at n=5.

Sections 1–5 and 7 are writable **today** and depend on nothing still running.

---

## 7. RISK REGISTER

| Risk | Mitigation |
|---|---|
| B2 fails | §1 A–H stand alone; the paper is written to not need it |
| Fold confound unresolved | S2, promoted to first priority |
| "One paper, not general" | the proposition + the reusable three-test audit |
| "No method contribution" | topic 06 asks for auditing frameworks; B2 is upside |
| Success creates more work | Stage 2 and TEST pre-declared out of scope |
| GPU overruns the writing window | Sep 2 hard stop, already committed |

---

## 8. STANDING RULES

- No result is reported from a protocol we did not also run on the baseline.
- The co-adapted critic's AUC is never reported as a privacy result.
- If a B2 arm beats the anchor under one protocol and not another, **both are
  reported**.
- Any number entering the manuscript is re-verified against its sealed artifact
  during Sep 3–5, not trusted from a note.
