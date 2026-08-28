# MANUSCRIPT DRAFT — TIER 0

**Status:** working draft, 2026-08-29. Contains ONLY results that already exist,
are sealed, and are reproducible at zero additional GPU cost. Every number below
is traceable to a committed artifact; none is projected or expected.

**Working title:** *An adversarial anonymization objective that is invariant to
the privacy metric it reports*

**Target framing:** evaluation-validity / reproducibility study with a
mechanistic diagnosis. **Not** a method paper — no method claim is made here.

---

## 1. WHAT IS CLAIMED

**C1 (formal).** The privacy objective of PriCheXy-Net (Packhäuser et al.,
MICCAI 2023) is mathematically invariant to the ROC AUC used to report privacy.

**C2 (empirical, same-checkpoint).** On real trained generators the logged
privacy metric varies by 9x while true re-identification AUC stays pinned near
0.90.

**C3 (falsification).** The "the training critic is too weak" hypothesis — the
premise of two separate method directions in this project — is false. The
co-adapted critic is the *strongest* re-identifier measured.

**C4 (consequence).** Four independent interventions, spanning auxiliary loss,
architecture, critic strength and initialization, produce no improvement. The
diagnosis explains why: none of them touches the objective.

**C5 (reproduction).** The published utility result reproduces; the published
privacy result does not. Faithful retraining lands on the paper's own
*no-anonymization* baseline.

Explicitly **not** claimed here: that any corrected objective works. That is a
separate experiment under separate preregistration.

---

## 2. C1 — THE OBJECTIVE IS AUC-INVARIANT

The paper states the privacy loss as `−log(1 − ŷ_v)` with
`ŷ_v = SNN(F(x₁), x₂)`. The released `utils/VerificationLoss.py` returns
`sigmoid(z)`, so

```
L_priv = −log(1 − σ(z)) = softplus(z),     z = verifier logit
```

Minimizing it drives every logit toward −∞, i.e. it asks the verifier to answer
"different patient" on *every* pair. ROC AUC depends only on the **order** of
`z` between positive and negative pairs, and a uniform shift preserves order
exactly.

Numerical demonstration (`D1`, seeded, self-contained):

| logit shift | `L_priv` | `ver_loss` | **ROC AUC** |
|---:|---:|---:|---:|
| 0 | 1.058266 | 0.507210 | **0.9808** |
| −5 | 0.027665 | 0.025584 | **0.9808** |
| −20 | 0.000000 | 0.000000 | **0.9808** |

`auc_is_constant_across_all_shifts: true`. The loss reaches exactly zero with
the ranking untouched.

**This is a property of the published method, not of this project's port.**
`utils/VerificationLoss.py` is byte-identical to upstream commit `29245d1`
(SHA `bececf73266d070b`), and the paper states the same formula in text.

---

## 3. C2 — THE LOGGED METRIC AND THE REPORTED METRIC ARE DECOUPLED

Four checkpoints from two independently trained generators, each scored against
**its own co-adapted verifier**, geometry anon(x₁)/real(x₂), 2000 VAL pairs:

| checkpoint | `ver_loss` (as logged during training) | **true ROC AUC** |
|---|---:|---:|
| run A — selected epoch | 0.0529 | **0.9147** |
| run A — epoch 250 | 0.0784 | **0.8921** |
| run B — selected epoch | **0.0343** | **0.8987** |
| run B — epoch 250 | **0.3092** | **0.9108** |
| *a truly anonymizing generator* | *0.5* | *0.5* |

The training metric spans a **9x range** (0.034 → 0.309) while true
re-identification stays within 2.3 points of 0.90. A log reading 0.034 — which
presents as "the adversary is 96.6 % confident these are different patients" —
coexists with the adversary ranking same-patient pairs correctly 89.9 % of the
time.

Two cheap objections are closed in advance:
- **Memorization:** train/val patient overlap is exactly **0**.
- **Base rate:** the VAL pool is exactly **1000/1000** positive/negative, so a
  low mean sigmoid is not an artifact of class imbalance.

---

## 4. C3 — THE WEAK-CRITIC HYPOTHESIS IS FALSE

Both Direction A (periodic best-response refresh) and Direction B (k:1
critic/generator update ratio) rest on the premise that the training verifier is
too weak. This project's own protocol review recorded it as
`WEAK_CRITIC_HYPOTHESIS_STATUS: UNTESTED`.

Measured on generator `2f285743`, identical bytes on both sides:

| | AUC |
|---|---:|
| co-adapted **training** verifier | **0.9147** |
| fresh adaptive attackers (n=3, sealed P0 manifests) | 0.7258 |
| **best-response gap** | **−0.1888** |

The training critic is the **strongest** re-identifier measured, not the
weakest. `weak_critic_hypothesis_supported: false`.

Direction B additionally carried an independent implementation confound
(BatchNorm drift from `no_grad` forwards while the generator was in `train()`
mode) and was classified `DIRECTION_B_V1_CAUSAL_EVIDENCE: INVALID`.

---

## 5. C4 — FOUR INTERVENTIONS, NO IMPROVEMENT

All paired by attacker seed on the governed harness; lower AUC is better.

| Intervention | Anchor | n | Δ | Result |
|---|---|---:|---:|---|
| C4 auxiliary feature loss | B_dev | 26 | **+0.0089** | worse; 11/26 seeds |
| Attention-gated generator | released ckpt | 3 | **+0.0906** | worse |
| — its ver-selected checkpoint | released ckpt | 3 | **+0.0616** | worse |
| Hardened critic (k=3) | — | — | — | invalid (confounded) |
| Continuation from released ckpt | — | — | — | never produced a checkpoint |

Two remarks that belong in the paper:

1. **The single-seed result was a lucky draw.** At the certified seed the
   auxiliary-loss arm read Δ = −0.0109 and was promoted. Across 26 attacker
   seeds it is Δ = +0.0089, winning 11/26 — a coin flip.
2. **The attention arm was inert.** Its gates were zero-initialized in a
   configuration that is a gradient fixed point; after 250 epochs all 87k+ gate
   weights were still exactly 0.0 and only four bias scalars had moved. Those
   runs are therefore **not** evidence that attention fails to help; the
   mechanism was never active. Disclosed as an erratum.

The interventions span auxiliary loss, architecture, critic strength and
initialization. **None touches the privacy objective** — which §2 shows is the
component that does not measure what is reported.

---

## 6. C5 — UTILITY REPRODUCES, PRIVACY DOES NOT

| Measurement | Value | Source |
|---|---:|---|
| Paper, baseline (real data, TEST) | 81.8 ± 0.6 | Table 1 |
| Paper, μ=0.01 (TEST) | 57.7 ± 4.0 | Table 1 |
| **Ours: released ckpt, published protocol, TEST, n=10** | **60.80 ± 4.35** | reproduction |
| Ours: released ckpt, adaptive harness, VAL, n=26 | 69.85 | P0 screen |
| **Ours: faithful retrain, adaptive harness, VAL, n=26** | **82.44** | P0 screen |
| Ours: pre-adversarial init, adaptive harness, VAL, n=10 | 83.66 | P0 |

Classification utility: paper 80.5 → 76.2; ours 81.58 → 77.30 (released) /
78.40 (retrain). **The utility side reproduces**, which rules out a
generally-broken pipeline.

The internally-valid statement (same protocol, same fold, same harness):

> Our full adversarial training moved re-identification by **−1.2 points**
> (83.66 → 82.44), from the pre-adversarial initialization.

The external landmark, with its caveat stated:

> That retrained anonymizer sits at the paper's own *no-anonymization* baseline
> (81.8). The two numbers are different measurements — theirs on real images
> with the original SNN on TEST, ours on deformed images with a retrained
> attacker on VAL — that coincide in magnitude. Illustrative, not an identity.

One documented recipe deviation exists and must be reported: the paper trains
the anonymizer at **batch 64**; this project's frozen config uses **16**.
Everything else — losses, update order, optimizers, μ, epochs, initial
checkpoints, and the pair files (byte-identical) — matches.

---

## 7. WHY NOTHING IMPROVED — THE SYNTHESIS

The objective can be driven to zero by a uniform shift of the verifier's logits.
A generator that does so must still deform strongly — and it does: the retrained
generator deforms as hard as the released one (mean |flow| 0.898 vs 0.875, mean
pixel change 0.0223 vs 0.0236) — but it is never asked to remove identity, only
to move a score. Whether the deformation *incidentally* destroys identity is a
property of the optimization path, not of the objective.

Two independent observations support "incidental":

- the two generators' flow fields are essentially **orthogonal**
  (cosine 0.0497) despite comparable magnitude and spatial concentration;
- in the paper's own Table 1 the baseline row has SD ±0.6 while all three
  anonymized rows have SD ±2.6 to ±5.8 — the privacy effect is **4–10x more
  variable** than the measurement noise floor.

---

## 8. LIMITATIONS (to state, not to hide)

- One generator seed per arm; attacker seeds are nested replicates, not method
  replicates. No generator-seed variance is estimated.
- One attacker family (Siamese ResNet-50).
- Selection and scoring draw on the same VAL pool; this applies identically to
  both arms so it largely cancels in paired Δ, but it is not an independent
  confirmation set.
- Patient-clustered bootstrap CIs are unvalidated; seed-level intervals only.
- Segmentation utility is blocked by evaluator provenance and is excluded.
- §5's attention arm rests on runs whose mechanism was inert (erratum §5).

---

## 9. ARTIFACTS

| Claim | Artifact |
|---|---|
| C1, C2, C3 | `privacy_objective_diagnosis.json` + `diagnose_privacy_objective.py` (`09046f7`) |
| C4 | P0 sealed manifests, Stage-A pilot summary |
| C5 | `FINAL_10SEED_PRICHEXY_REPRODUCTION.md`, P0 screen, utility results |
| Errata | `V2_CONFIRMATION_ADDENDUM_A`, Phase B Addenda B & C, commit `10d9212` |

---

## 10. WHAT WOULD UPGRADE THIS DRAFT

Each is independent; none is a gate on the above.

1. **Released endpoints at μ=0.001 / 0.005 under the adaptive harness** (~8 GPU-h,
   no training). Would extend §6 from one point to the paper's full published
   curve. Unblocked by protocol V1_2; awaiting human approval signature.
2. **Effective batch 64, original objective** (~11 GPU-h). Would convert §6's
   deviation from a caveat into an explanation.
3. **A corrected objective** (separate preregistration). Would make this a
   method paper. Phase B's first attempt failed on gradient saturation
   (Addendum C) and is closed.
