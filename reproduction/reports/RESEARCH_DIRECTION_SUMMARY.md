# Research Direction Summary (as of 2026-08-21)

**Purpose of this document:** a single entry point for anyone (human or agent) picking up this
project, summarizing what is certified, what is uncertified-but-real pilot data, what the current
open research direction is, and exactly where to look for the underlying evidence. Written for an
external reader with no prior context on this conversation.

---

## 1. What is certified (frozen, governed, do not touch)

- `research_agent/M2_S1_EXECUTION_LOCK.json` + `research_agent/PROTOCOL_AUTHORITY.md` — the sole
  scientific-method authority for M2-S1.
- `research_agent/M2_S1_C4_RESULT.md` / `research_runs/M2_S1/M2_S1_summary.json` — the completed,
  certified S1 result: **B_dev vs C4, seed 42, VAL fold only, TEST never touched.**
  - Privacy gate (Δpriv ≤ +0.03): PASS, Δpriv = −0.0109 (B_dev AUC 0.8132, C4 AUC 0.8023).
  - Classification gate (Δclass ≥ 0.0): PASS, Δclass = +0.0216 (B_dev 0.7841, C4 0.8056 macro AUC).
  - Verdict: **"C4 S1: PROMOTE TO S2."**
  - **This is a single-seed result.** See §2 below for why it should not be over-trusted on its own.

Nothing under `research_agent/m2_dev/`, no frozen config in `config_files/`, and no file whose
SHA256 is embedded in `M2_S1_summary.json` has been modified by any of the work described below.

## 2. Stage A pilot — real multi-seed statistics (uncertified, but real GPU-measured data)

**Question asked:** was S1's single-seed "PROMOTE" verdict reliable, or could a different seed have
flipped it? **Method:** re-train only the attacker (Siamese Re-ID network), reusing the exact S1-selected
generator checkpoints for B_dev and C4, for seeds 43–67 (25 additional seeds, n=26 total per arm
including the certified seed 42). Full methodology, fidelity policy, and results:
**`reproduction/reports/S2_CONFIRMATORY_DESIGN_PROPOSAL.md`** (§7–§8). Raw per-seed data:
**`reproduction/s2_pilot/results/pilot_summary_final.json`** (50 pilot runs) and
**`reproduction/s2_pilot/run_stage_a_pilot.py`** (the script; never touches any frozen file, reuses
the real unmodified `evaluate_reid_val()` for every AUC).

**Headline finding:** with n=26 real seeds,
- Non-inferiority (Δ < 0.03 margin) is **confirmed** with p = 0.005 (one-sided 95% upper bound on
  mean Δ = 0.0218 < 0.03).
- **Superiority is NOT confirmed** — mean Δ = +0.0089, 95% CI [−0.0066, +0.0245], contains zero.
  C4 does **not** demonstrably reduce Re-ID AUC vs. B_dev; the point estimate is even slightly
  unfavorable.
- 7 of 26 seeds (27%) would have individually **failed** the S1 gate had they been drawn as "the"
  seed — i.e. the certified single-seed S1 result was closer to a lucky draw than a robust finding.
- Attacker-seed correlation between the two arms (despite generator-level common-random-numbers
  design): ρ ≈ 0.17 — much lower than commonly assumed for CRN designs.

**Implication:** C4's classification-utility improvement is real and safely non-inferior on privacy,
but C4 alone is **not a privacy-improving method**. Any claim of *lower* Re-ID AUC than baseline needs
a genuinely new mechanism — see §3.

## 3. Current open direction: closing the train/eval adversary strength gap

**Mechanism identified (from reading `research_agent/m2_dev/anonymizer_runner.py` closely):** the
verifier critic used during generator training (`SiameseNetwork`, same architecture family as the
eval-time attacker) is updated **1:1 with the generator** every batch — it never gets to converge
against a *fixed* generator. The eval-time S1 attacker, by contrast, is trained to convergence
(100 epochs, patience 5) against the **final, frozen** generator. This asymmetry is a plausible
explanation for why measured Re-ID AUC (B_dev mean 0.8237, SD 0.031, n=26) is far higher than
train-time `privacy_term` would suggest — the generator has never been tested during training against
an adversary with a real convergence advantage.

**Two candidate fixes were designed** (not both tried yet):
- **Direction A — periodic hard-restart:** periodically freeze the generator, train the verifier to
  full convergence from scratch, then resume generator training against the hardened verifier.
  Not yet implemented.
- **Direction B — k:1 verifier:generator update ratio (in progress):** give the verifier k=3 updates
  per generator update (2 extra verifier-only steps per batch, same batch reused, fresh no_grad
  generator forward for the extra steps — chosen specifically to keep the *generator's* per-epoch
  data traversal and update count byte-identical to the certified B_dev run; confirmed via matching
  `order_sha256` per epoch in the smoke test). Script:
  **`reproduction/method_dev/run_hardened_verifier.py`**. Currently running a full 250-epoch B_dev
  seed=42 k=3 training (~34 GPU-hours estimated from the observed ~492s/epoch pace); check
  `reproduction/method_dev/hardened_verifier_k3/B_dev/seed_42/train_log.jsonl` for live progress and
  `reproduction/method_dev/run_bdev_k3.log` / `checkpoint_manifest.json` for completion status.

**What "success" looks like:** once this generator finishes training, its Re-ID AUC must be measured
the same rigorous way as §2 (re-run `run_stage_a_pilot.py`-style attacker retraining across enough
seeds, compare mean Δ against the B_dev baseline of 0.8237 with a proper significance test) — a
single-seed AUC number, even a good-looking one, would repeat exactly the mistake §2 was designed to
catch. **This has not happened yet** — the generator training itself is still in progress as of this
writing.

## 4. What a Q2-level contribution would require (explicit, not yet achieved)

Discussed and agreed explicitly: the statistical rigor in §2 (real seed-variance quantification,
proper non-inferiority testing) is necessary infrastructure but **is not, by itself, a sufficient
contribution** for a Q2 journal — it validates that nothing broke, not that something improved. The
paper's core claim needs a **measured, statistically significant improvement** in Re-ID AUC (lower
than B_dev's 0.8237 baseline, under the same adaptive-attacker/VAL-fold evaluation protocol used
throughout this project — **not** the original PriCheXy-Net paper's ~0.577, which is a different
threat model: static attacker, TEST fold, not comparable). Direction A/B above are the current
candidate mechanisms to produce that result; neither has produced a validated positive result yet.

## 5. Two directions explicitly not yet started

- **Segmentation improvement over baseline:** blocked — `research_agent/M1_1_SEGMENTATION_PROVENANCE.md`
  confirms `SEGMENTATION_STILL_BLOCKED` (no certified segmentation evaluator exists yet). Needs that
  evaluator built and certified before any segmentation comparison is meaningful.
- **Stage B (anonymizer-seed variance):** retraining the full generator under multiple seeds to
  measure how much the whole non-inferiority conclusion in §2 depends on the *specific* generator
  training run selected at S1 — expensive (~27–59 GPU-hours per retrain), not started.

## 6. Where everything lives (all outside the certified pipeline)

| What | Path |
|---|---|
| Reproduction of the original PriCheXy-Net baseline + statistical audit | `reproduction/reports/PRICHEXY_PAPER_REPRODUCTION.md`, `FINAL_10SEED_PRICHEXY_REPRODUCTION.md` |
| Predeclaration/execution erratum | `reproduction/reports/PAPER_REPRO_PROTOCOL.md` §8 |
| S2 design + Stage A pilot results | `reproduction/reports/S2_CONFIRMATORY_DESIGN_PROPOSAL.md` |
| Stage A pilot script + raw results | `reproduction/s2_pilot/run_stage_a_pilot.py`, `reproduction/s2_pilot/results/` |
| Direction B experiment script + in-progress run | `reproduction/method_dev/run_hardened_verifier.py`, `reproduction/method_dev/hardened_verifier_k3/` |
| Certified S1 result | `research_agent/M2_S1_C4_RESULT.md`, `research_runs/M2_S1/M2_S1_summary.json` |

None of the `reproduction/` paths above are tracked by `.gitignore`'s default `reproduction/` rule —
they were force-added individually (`git add -f`) specifically so this summary and its evidence would
be visible in the pushed repository without also pushing large/binary checkpoint files.
