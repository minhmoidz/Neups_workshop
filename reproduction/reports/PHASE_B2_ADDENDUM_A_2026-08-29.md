# ADDENDUM A to PHASE_B2_PREREGISTRATION_2026-08-29

**Date:** 2026-08-29, hours after the parent lock (commit `3d2b7a1`).
**Amends:** §7 of the parent — the scope of the futility rule.
**Type:** pre-execution clarification of an ambiguity. The parent is unmodified.

## 0. LOCK CONDITION

No Phase B2 artifact exists. The runner still cannot be configured as §3
requires. **Zero candidate values, of either arm, have been observed.** This
clarification is therefore made blind to any outcome, which is the only
condition under which it is legitimate.

## 1. THE AMBIGUITY

Parent §7 states:

> "If at epoch 25 the running minimum of the pooled VAL `gap` has not fallen by
> at least 25% from its epoch-0 value, stop and classify
> `H-B2-MECHANISM-FAILED`."

It does not say whether "stop" means *stop that arm* or *stop the experiment*.
Two arms are declared (§5: `ver_loss_weight` 1.0 and 3.0), so the two readings
differ materially: arm B applies three times the weight to the privacy term, and
could move `gap` where arm A does not.

Left unresolved, the reading would be chosen after seeing which arm stalled.
That is precisely the post-hoc interpretive freedom this project exists to
eliminate, so it is fixed now.

## 2. RESOLUTION — THE FUTILITY RULE IS PER-ARM

1. The futility check is evaluated **independently for each arm**, at that arm's
   own epoch 25, against that arm's own epoch-0 pooled VAL `gap`.
2. An arm that trips it is stopped and recorded
   `H-B2-MECHANISM-FAILED (arm)`. It contributes no checkpoint, no attacker
   runs and no privacy value.
3. **A tripped arm does not stop the other arm**, and does not by itself
   classify the experiment.
4. Experiment-level classification, applied once to the surviving arms under the
   parent's §5 selection rule and §7 table:
   - **both arms trip** -> `H-B2-MECHANISM-FAILED` for the experiment;
   - **exactly one arm survives** -> it is the primary by default; the §5
     utility rule is vacuous with one arm but the §8 utility gate still applies
     in full;
   - **both survive** -> the parent's §5 rule selects the primary unchanged.
5. A tripped arm is **reported**, with its `gap` trajectory, in every case. It
   is not omitted from the record because it failed.

## 3. EXECUTION ORDER, DECLARED NOW

Arms run **sequentially, arm A (`ver_loss_weight=1.0`) first**, for scheduling
reasons only: the futility gate then bounds the worst case to roughly 3 GPU-h
rather than a full run, which matters against a 2026-09-08 deadline.

Order carries no scientific weight and confers no priority. Arm A being first
does **not** make it the primary; §5 selects on utility regardless of order.
Arm B runs whatever arm A does, unless arm B is itself abandoned for the
schedule reason in §4.

## 4. SCHEDULE STOP, DECLARED NOW

All Phase B2 GPU work stops on **2026-09-02**, whatever its state. Any arm not
finished by then is reported as `NOT_RUN` or `INCOMPLETE`, never as a null
result and never quietly dropped.

If Stage 1 classifies `H-B2-SUPERIOR`, **Stage 2 (n=26) and the §9 TEST
confirmation are NOT executed before the deadline.** The Stage 1 result is
reported as what the parent §6 already calls it — a screen, at n=5, with a
smallest detectable effect of about −0.041. Declared now so that a positive
result cannot later justify spending the writing window on confirmation runs.

## 5. WHAT THIS DOES NOT CHANGE

No decision rule, threshold, seed list, arm definition, objective, weight,
anchor or utility gate. δ = 0.03 remains frozen, the futility threshold remains
25% at epoch 25, and every anti-anchoring commitment of parent §11 applies to
this addendum.

*End of Addendum A. Corrections require Addendum B, dated.*
