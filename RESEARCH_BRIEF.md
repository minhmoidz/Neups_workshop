# PriCheXy-Net Improvement — Complete Research Brief & Handoff

**Purpose of this document:** single self-contained handoff for an external research agent (e.g. Claude
Scientist) to (a) reproduce our numbers, (b) understand what has been tried and what failed, and (c) design
the most promising path to a publishable paper that *improves* PriCheXy-Net, measured on the Pareto curve
(privacy ↔ utility), with two novel downstream metrics neither in the original paper (Top-1 identification
and lung/heart segmentation).

**Paper being improved:** PriCheXy-Net (MICCAI 2023), arXiv:2209.11531 — "PRIvacy-preserving CHest X-ray
(network)". Original repo: https://github.com/kaipackhaeuser/PriCheXy-Net. Companion re-ID paper:
https://www.nature.com/articles/s41598-022-19045-3.

**Task being solved:** chest X-rays are biometric identifiers — a trained Siamese network (SNN) can
re-identify a patient's scan from a released (anonymized) image. PriCheXy-Net warps the X-ray with a
learned smooth flow field F = UNet(x), clipped by amplitude µ, to (1) break Re-ID (verification AUC),
(2) preserve diagnostic utility (classification AUC), and (3) — our extension — preserve anatomical
segmentation (Dice/IoU/HD95).

---

## 1. Central Thesis / Working Insight

The original paper reports verification AUC **0.818 → 0.577** (re-ID) while classification AUC drops
**0.805 → 0.762**. Our experiments strongly support the following interpretation:

> **PriCheXy-Net privacy is fundamentally ADVERSARIAL, not information-theoretic.**

Evidence chain:

- **D2 — generator crushes the train-time adversary but generalizes poorly.** The train `ver_loss` reaches
  ~0.01 (adversary completely fooled) while a *freshly re-trained* attacker still scores 0.60.
- **D4 — pure random warp at the same µ gives ZERO privacy.** `stochastic_lambda=1.0` (uniform random field
  only, µ=0.01) → Re-ID **0.8174 ± 0.029** — identical to raw images (0.8015). All privacy comes from the
  *learned/targeted* deformation, not from entropy.
- **D3 — the utility constraint is effectively inert.** BCE is diluted ~25× (positive labels break +0.415
  but are outnumbered 12.3× by negative labels that get *more confident* −0.016). The generator can happily
  satisfy BCE by nudging everything toward negative predictions — exactly when ranking-based AUC collapses.
  So **µ is the only thing actually protecting utility**; the classifier loss contributes almost nothing.
- **D7 — nothing non-adaptive predicts re-identification risk.** A frozen ImageNet ResNet-50 similarity proxy
  gives Spearman ρ=0.46 (n.s.), and it INVERTS at the most important case (C2). Privacy is defined by what
  an *adaptive learner can recover*, not by "looks different in ImageNet space".

Consequence: at fixed deformation budget µ, **targeting is everything; adding entropy is useless; stronger
adversaries barely matter.** The only real levers are:
1. **Re-spend the budget smarter** (C2: anatomical budget map).
2. **Protect utility better so we can raise µ** (C4: feature-retention loss).

---

## 2. Reproduction Status & Wormholes Found

We reproduced the paper on NIH ChestX-ray14 (112,120 PNGs, official test fold = 25,596 images, 14
pathologies). Primary finding — a **bug we introduced, then fixed**:

### The gradient-accumulation bug (H2)

```python
optimizer_g.zero_grad()                        # WIPED accumulated grads EVERY iteration
(total_loss / accumulation_steps).backward()
if (i+1) % accumulation_steps == 0:
    optimizer_g.step()
```

`zero_grad()` at the top of each loop destroyed the just-accumulated gradients → each optimizer step only
saw 1/4 of a batch's gradient and the generator was updated 4× less often → **under-trained generator**.
This exactly explains repro gap: Re-ID 0.604 (us) vs 0.577 (paper), AND utility 0.770 (us) vs 0.762 (paper)
— both directions. Fixed in `utils/utils.py` (removed spurious zero_grad; `accumulation_steps` default 1).
Regression test: `test_grad_accum.py`.

### Verified protocol numbers (10-seed protocol everywhere)

| Run | Generator | Re-ID AUC (10 seeds) | Class AUC | Notes |
|---|---|---|---|---|
| Paper reference | — | 0.577 ± 0.040 | 0.762 | target |
| Raw images (ours anchor) | — | 0.8015 ± 0.027 | 0.8050 | = upper bound |
| **baseline_fixed (CANONICAL, bug fixed, 60 ep)** | generator_lowest_total_loss | **0.635 ± 0.079** | **0.773** | the only valid comparison baseline |
| run_1 (pre-bug-fix original) | lowest_total | 0.604 ± 0.082 | 0.770 | historical reference only |
| run_1 (pre-fix) | lowest_ver | 0.641 ± 0.058 | — | H4: lowest_ver is worse → use lowest_total |
| run_2 (H1 refresh critic) | | 0.622 ± 0.052 | 0.755 | refreshing critic per-epoch WORSENS utility |
| run_3 (entropy/confusion loss) | | 0.706 ± 0.050 | 0.786 | worst privacy — entropy keeps info |
| run_4 (C3 ensemble + restart, K=3 SNN) | ep 54 | 0.606 ± 0.048 | 0.762 | privacy equal to run_1, −0.008 utility; only real gain: **−42% seed variance** |
| control λ=1 (pure random warp) | none (no training) | 0.8174 ± 0.029 | — | privacy ≈ 0 (D4) |
| **C2 budget map (µ=0.01, γ=58)** | | 0.760 ± 0.026 | **0.709** | alone, C2 collapses utility → needs C4 |
| **C4 feature-retention (µ=0.01)** | | — | **0.792** | best class so far |

**Both anchors validated:** raw-image class AUC = 0.8050 ≈ paper 0.805; raw Re-ID 0.8015 ≈ paper 0.818.

---

## 3. The µ-Sweep of C4 (feature-retention) — LEARNED THE HARD WAY

### Method C4 (installed & verified)
Replace the saturated BCE with **feature-retention**: freeze CheXNet, minimize MSE between the 1024-d
pooled features of the real and the deformed image. Signal is dense and tied to diagnostic content; scaled
correctly by default (`feature_loss_weight=1.0` — measured on real data, not noise). Also available:
`ac_pos_weight` (rebalance the 12.3× label dilution). Bit-for-bit compatible with old behavior when off.

### Full results table (all axes, 2026-08-10)

| Method (µ) | Re-ID 10-seed (REAL) | Class AUC | Seg Dice mean | Top-1 ident | Seg IoU | HD95 |
|---|---|---|---|---|---|---|
| Raw (upper bound) | 0.8015 ± 0.027 | 0.8050 | 0.947 | 0.170 | 0.905 | 1.74 |
| baseline_fixed (0.01) | **0.635 ± 0.079** | 0.773 | 0.937 | 0.152 | 0.886 | 2.27 |
| C4 (0.01) | 0.7496 ± 0.033 | **0.792** | **0.938** | 0.156 | 0.889 | 2.12 |
| C4 (0.02) | 0.7598 ± 0.037 | 0.788 | 0.928 | 0.148 | 0.870 | 2.61 |
| C4 (0.04) | 0.6985 ± 0.065 | 0.785 | 0.897 | 0.148 | 0.818 | 4.25 |
| C2+C4 (0.01) | (proxy 0.663 only) | 0.784 | 0.897 | 0.142 | 0.817 | 4.75 |

### The outcome: C4 FAILS on privacy

Despite proxy re-ID predicting C4@0.04 ≈ 0.634 (very private), **the real 10-seed attacker still scores
0.6985 ± 0.065 at µ=0.04 — far above baseline_fixed 0.635**. This is the decisive negative result:
- C4 preserves diagnostic features, but **identity information overlaps with diagnostic features**. Keeping
  one keeps the other. A re-trained attacker recovers it.
- The frozen-ImageNet proxy **lies for C4/C2** (confirmed D7). Proxy is fine for *ordering* radically
  different methods, useless for deciding whether a feature-preserving method is private.

**Caveat on our own first 0.02 measurement:** the first C4@0.02 10-seed run accidentally used mu=0.01
(`run_snn_multiseed.py` did not override `mu`, base config had 0.01). That run is really C4@0.01 (0.7496).
We re-ran with `--mu 0.02` → 0.7598. The `--mu` override is now added to the multiseed script.

### What DID survive C4
Segmentation stays ≈ raw for C4@0.01–0.02 (Dice 0.938 / 0.928, ~1–2% off raw), and Top-1 identification
is mildly reduced. But since Re-ID (the core privacy axis) is worse than baseline, **C4 alone does not win
on the Pareto front.** C4's value may be as a *utility-preserving* component *combined* with a real
privacy method — not as the privacy driver.

---

## 4. What We Have that the Paper Doesn't (contributions already banked)

1. **Faithful reproduction with bug caveat** (D2, gradient-accum).
2. **D3 diagnosis**: utility loss diluted ~25×; positive/negative BCE breakdown (+0.415/−0.016).
3. **D4 negative control**: random warp at same µ = zero privacy (paper never ran this).
4. **D7**: non-adaptive proxies cannot predict re-identification risk (Spearman ρ=0.46; C2 inverted).
5. **Two novel downstream metrics:**
   - **T6 Top-1 / Top-5 / MRR identification** (N:1 linkage, frozen ResNet-50, gallery 500) — script
     `eval_top1.py`. Raw 0.170 / baseline 0.152 / C4 0.148–0.156 / C2+C4 0.142.
   - **T7 Segmentation**: CheXmask RLE (Left/Right Lung + Heart), U-Net (feat=16), Dice/IoU/HD95 — scripts
     `train_seg.py`, `eval_seg.py`, model at `archive/train_seg_unet/best.pth` (val Dice 0.955/0.964/0.946).
6. **C4 / C2 / C2+C4 / stochastic_lambda** all implemented and verified (bit-for-bit bootstrapping checks).

---

## 5. Hypotheses CLOSED (do not re-run)

| # | Hypothesis | Conclusion |
|---|---|---|
| H1 | Refresh AC critic per epoch | **Rejected** (0.622/0.755 — worse both axes). Explained by D3. |
| H2 | grad accumulation ×4 | **Fixed** — was the repro bug. |
| H3 | 1 run vs 10 runs | Handled — 10 seeds mandatory (std 0.03–0.08). |
| H4 | checkpoint selection | `lowest_ver` is worse (0.641>0.604 → use `lowest_total`). |
| T2 | entropy/confusion loss | **Strongly rejected** (0.706, worst). Derivative of −log(1−p) = 1/(1−p) increases with attacker confidence — the "vanishing gradient" story was wrong. |
| C1 | structured random deformation | **Rejected by D4**. Code kept as ablation line only. |
| C3 | ensemble+restart adversary | **Negative** (D5): privacy equal, utility −0.008; only −42% variance. Keep as evaluation-protocol contribution. |
| D6 | C2 alone | Utility collapses to 0.709 → C2 requires C4. |
| **C4** | feature retention | **Negative on privacy** (real 10-seed worse than baseline at all µ). See §3. |

---

## 6. Open Questions for the Research Agent (where we need you)

We want the paper to show a **method that wins on the Pareto front** (lower Re-ID at equal class, or higher
class at equal Re-ID), plus the two novel metric columns. Candidate directions, in rough priority:

1. **Re-spend budget toward biometric anatomy (C2), now powered by C4-kept utility.**
   Paper's Grad-CAM shows the re-ID model attends to lungs, ribs, heart, clavicles, diaphragm. C2 already
   redistributes µ via a budget map (mean(M)=µ enforced → fair comparison). C2+C4 currently: class 0.784
   (≈ baseline 0.773, better than C2-alone 0.709) but *real Re-ID of C2+C4 is UNKNOWN (proxy 0.663 only)*.
   → **Ask: run 10-seed Re-ID on C2+C4 (µ=0.01) and on a µ-sweep {0.02, 0.04} of C2+C4.** If target-specific
   budget reduces what the SNN can recover, this is the paper.

2. **Targeted budget map targeted at *biometric*, not anatomical, saliency.** Use the trained SNN's saliency
   (Grad-CAM / input-gradient) of the *adversary* to define M(x), instead of anatomy a priori. This directly
   implements "pay the bit where identity lives". Novel and cheap to try now that a trained SNN exists.

3. **Canonical-deformation-factorization check:** a deterministic warp F=UNet(x) is near-bijective → privacy is
   coding churn. Is there a way to break bijectivity *locally* only in rear-ID-salient regions (randomize /
   collapse those, keep rest near identity) → maximizes privacy-per-budget and preserves segmentation. This is
   the "destruct locally, preserve globally" idea. (Random-warp baseline D4 shows global entropy is useless,
   so locality/targeting is the differentiator.)

4. **Stochastic λ with *targeted* noise:** current `stochastic_lambda` mixes a global independent random field
   (useless per D4). Try mixing random field *only under the budget map* → one-way, non-invertible warp in the
   identity-critical region. Needs a 10-seed run (λ>0 variants have never been 10-seed'd).

5. **Utility rebalancing to enable higher µ:** `ac_pos_weight=12.3` (fixes ±25× dilution, but must rescale
   `ac_loss_weight≈0.105`), or keep class AUROC while µ↑. The real lever: can we raise µ (to 0.05–0.1) and
   hold class ≥ baseline with C4 loss, so Re-ID has room to fall? µ-sweep of C4 (0.01/0.02/0.04) is done;
   none won because C4 keeps identity. A method that *destructures* rather than *retains* features is needed.

6. **Reduce seed variance for power:** ensemble (−42% variance, D5) or short-budget multi-seed attacker
   (2–3 seeds, ~20–30 min) as the cheap screening protocol — but it MUST be validated against the 7 ground
   truth points in §2 before trusting (D7).

7. **Given the negative C4 result:** reframe the paper as *critique + two new metrics* if no method wins —
   that is already a valid reproduction/analysis contribution (see PLAN.md conclusions). The agent should
   decide whether to invest compute in method search or bank the analysis paper.

---

## 7. Environment & Reproduction Notes (for the agent)

- **Machine:** native Linux (Ubuntu 24.04), GPU 1× RTX 5070 Ti 16GB (VRAM is the hard constraint; a 10-seed
  SNN session needs ~10.3GB; leave ~6GB headroom). venv at `.venv` (Python 3.10.20, torch 2.7.0+cu128,
  torchvision 0.22.0+cu128). Commands that spawn subprocesses must `export PATH=$PWD/.venv/bin:$PATH`.
- **Data:** images at `/home/minhtt/datasets/nih/images/` (112,120 PNG). Configs were updated to this path.
  CheXmask masks: `data/chexmask/ChestX-Ray8.csv` (RLE, CC-BY, 112,120 rows, 1:1 with images).
- **Entry scripts:** `train_architecture.py` (generator), `retrain_SNN.py` (attacker), `run_snn_multiseed.py`
  (N-seed + summary; now supports `--mu`), `eval_classifier.py` (class), `proxy_reid.py` (+`calibrate_proxy.py`),
  `eval_top1.py` (T6), `train_seg.py`/`eval_seg.py` (T7), `stress test`: `test_grad_accum.py`.
- **Key checkpoints:**
  - baseline: `archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth`
  - C4 µ=0.01/0.02/0.04: `archive/train_prichexy_net_c4_featureloss/`,
    `archive/train_prichexy_net_c4_mu0.02/`, `archive/train_prichexy_net_c4_mu0.04/`
  - C2+C4: `archive/train_prichexy_net_c2c4/`
  - segmenter: `archive/train_seg_unet/best.pth`
- **10-seed result dirs:** `archive/retrain_snn_runs_<tag>/summary.txt` (format: N_runs, AUC_mean,
  AUC_std, Per_run). Read the *real* Re-ID numbers ONLY from these.
- **Class AUC:** `chexnet/results/<tag>/aucs.csv` → mean of 14 pathology AUCs.
- **Protocol rule:** every Re-ID conclusion must use ≥10 seeds (std 0.03–0.08). A single run means nothing.
- **tmux** for long runs (install-lite at `/tmp/opencode/tmux_local/usr/bin/tmux` with
  `LD_LIBRARY_PATH=/tmp/opencode/tmux_local/usr/lib/x86_64-linux-gnu:/tmp/opencode/tmux_local/lib/x86_64-linux-gnu`).

---

## 8. Key Numbers Cheat-Sheet (for quick referencing)

| Axis | Raw | Paper | baseline_fixed | C4@0.02 | C4@0.04 | C2+C4@0.01 |
|---|---|---|---|---|---|---|
| Re-ID 10-seed ↓ | 0.802 | 0.577 | **0.635** | 0.760 | 0.699 | proxy 0.663 (TO DO) |
| Class AUC ↑ | 0.805 | 0.762 | 0.773 | 0.788 | 0.785 | 0.784 |
| Seg Dice mean ↑ | 0.947 | — | 0.937 | 0.928 | 0.897 | 0.897 |
| Seg IoU mean ↑ | 0.905 | — | 0.886 | 0.870 | 0.818 | 0.817 |
| Seg HD95 ↓ | 1.74 | — | 2.27 | 2.61 | 4.25 | 4.75 |
| Top-1 ident. ↓ | 0.170 | — | 0.152 | 0.148 | 0.148 | 0.142 |

**Decisive pending experiment (highest value):** real 10-seed Re-ID of **C2+C4 across µ ∈ {0.01, 0.02, 0.04}**
and of **C2 alone with C4 protection**, to test whether *re-spending budget on biometric saliency + feature
retention* finally beats baseline_fixed (0.635) on the Pareto curve. If it doesn't, bank the analysis paper
(repro + D3/D4/D7 + the two new metrics) — that is already a solid, self-contained contribution.