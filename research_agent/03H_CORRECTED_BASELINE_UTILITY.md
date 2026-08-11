# STEP 3C — CORRECTED BASELINE UTILITY PROFILE (Classification + Anatomical Segmentation)

## 1. Objective

Measure the **complete utility profile** of the **frozen corrected generator**
(`networks/corrected_baseline/generator_lowest_total_loss_corrected.pth`, SHA-256
`8a489eec…`) across (1) **classification** utility and (2) **anatomical segmentation**
utility, using the canonical fixed classifiers/segmenters already in the repository.

The corrected canonical privacy baseline is locked (STEP 3B, not re-run):

| Privacy axis | Value |
|---|---|
| Adaptive Re-ID mean | 0.739194 |
| sample SD | 0.049847 |
| median | 0.722183 |
| maximum | 0.803662 |
| n valid attackers | 10 |
| n near-chance | 0 |

No proposed method was designed or run. No C2/C4, no stochastic anonymizer, no method
screening, no Top-k, no margins locked. This is a pure inference/evaluation of the fixed
corrected baseline.

## 2. Frozen corrected generator

| Check | Result |
|---|---|
| Path | `networks/corrected_baseline/generator_lowest_total_loss_corrected.pth` |
| Re-computed SHA-256 | `8a489eec036679f6775a2bf0b21fb112f1f968490a513b1f9b6dc012d4e384c2` |
| Provenance SHA-256 (03B) | `8a489eec036679f6775a2bf0b21fb112f1f968490a513b1f9b6dc012d4e384c2` |
| Match | **exact** |
| Loadable | yes (OrderedDict state_dict, UNet flow-field generator) |
| `transform_mode` | `corrected` (config `config_anonymization_baseline_corrected.json`) |
| `mu` | 0.01 |
| `stochastic_lambda` | 0.0 |
| Working tree before evaluation | clean |

All six pre-evaluation checks (recompute SHA / exact match / loadable / config transform_mode /
μ and stochastic settings / clean tree) **PASS**. No generator substitution or retraining.

## 3. Utility protocol audit

Inspected: `eval_classifier.py`, `chexnet/eval_model.py`, `chexnet/cxr_dataset.py`,
`train_seg.py`, `eval_seg.py`, `chexnet/seg_dataset.py`, `utils/utils.py`,
`utils/segmask.py`, `networks/UNet_PriCheXyNet.py`, `networks/UNetSeg.py`, and all
evaluation configs in `config_files/`.

**Audit table (before running):**

| Utility axis | Model/checkpoint | Frozen? | Input | Split | Metric |
|---|---|---|---|---|---|
| Classification | DenseNet-121 (14-output sigmoid head), `networks/pretrained_classifier.pth` | yes (eval mode, no grad) | deformed X-ray (flow field, 224×224, ImageNet norm) | NIH `fold==test` from `chexnet/nih_labels.csv` (n=25 596) | per-label ROC-AUC × 14, mean over 14 |
| Segmentation | UNetSeg(in=1,out=3,init=16), `archive/train_seg_unet/best.pth` | yes (eval mode, no grad) | deformed X-ray (flow field, 256×256, [−1,1]) | NIH `fold==test` (restricted to mask rows, n=25 596) | per-case Dice / IoU / HD95, aggregate mean |

**Ambiguity assessment:** the canonical utility protocol is **unambiguous**. There is exactly
one classifier checkpoint, one segmenter checkpoint, one NIH test fold, and a single metric
implementation path (`eval_classifier.py` → `chexnet/eval_model.py`; `eval_seg.py` →
`hd95()`/Dice/IoU). The only variable that distinguishes the corrected-baseline evaluation
from the historical legacy one is `transform_mode=corrected` + the corrected generator,
which is precisely the semantics locked in STEP 3B. **No scientifically plausible competing
evaluation definition exists**, so evaluation proceeded.

## 4. Classification evaluator

- Architecture: torchvision **DenseNet-121**, final `Linear(1024, 14)` + Sigmoid.
- Trained on **raw** (unperturbed) NIH images (`chexnet/model.py`, `fold=train`).
- **Frozen** during evaluation (`model.train(False)`, no gradients).
- Checkpoint: `./networks/pretrained_classifier.pth`, SHA-256 `8ad15b38…`.
- Labels: 14 NIH labels (Atelectasis … Hernia).
- Test list: `chexnet/nih_labels.csv`, `fold == "test"` → n = 25 596 images.
- Input deformation: **yes, dynamically** via `utils.deform()` with the frozen generator
  (flow-field UNet, Gaussian-smoothing k=9 σ=2, identity grid 256), `mu=0.01`,
  `stochastic_lambda=0.0`, `transform_mode='corrected'`; then expanded to 3 channels,
  resized 224, ImageNet-normalized.
- Mean AUC over 14 labels: simple unweighted arithmetic mean of the 14 per-label ROC-AUCs
  (`sklearn.metrics.roc_auc_score`), same as historical practice.
- Config: `config_files/config_eval_classifier_corrected_baseline.json` (new, correct-mode).

## 5. Classification per-label results

Evaluated the official utility test split exactly once. All 14 labels had both classes present.

| Label | n_positive | n_negative | AUC |
|---|---|---|---|
| Atelectasis | 3255 | 22341 | 0.745692 |
| Cardiomegaly | 1065 | 24531 | 0.856528 |
| Effusion | 4648 | 20948 | 0.813062 |
| Infiltration | 6088 | 19508 | 0.685410 |
| Mass | 1712 | 23884 | 0.784372 |
| Nodule | 1615 | 23981 | 0.717974 |
| Pneumonia | 477 | 25119 | 0.682891 |
| Pneumothorax | 2661 | 22935 | 0.812119 |
| Consolidation | 1815 | 23781 | 0.730899 |
| Edema | 925 | 24671 | 0.821556 |
| Emphysema | 1093 | 24503 | 0.855642 |
| Fibrosis | 435 | 25161 | 0.788169 |
| Pleural_Thickening | 1143 | 24453 | 0.750361 |
| Hernia | 86 | 25510 | 0.892499 |

No label was silently dropped. All 14 AUCs are finite. Per-label positive/negative counts
persisted in `03H_corrected_classification.json`.

## 6. Classification mean AUC

**mean_AUC_14 = 0.781227**

Independent recomputation from preserved `preds.csv` + `nih_labels.csv` label join gives
exactly the same per-label AUCs (max |csv − recompute| = 0.0), so the mean is not a
transcription artifact.

## 7. Segmentation evaluator

- Architecture: **UNetSeg**(in_channels=1, out_channels=3, init_features=16), sigmoid output.
- Anatomical targets: **Left Lung, Right Lung, Heart** (3 channels), ground truth from
  CheXmask RLE (`data/chexmask/ChestX-Ray8.csv`), decoded to 256×256 (nearest).
- Trained on **raw** (unperturbed) images (`train_seg.py`, `fold=train`).
- **Frozen** during corrected-baseline evaluation (eval mode, no grad).
- Checkpoint: `./archive/train_seg_unet/best.pth`, SHA-256 `2dfdcf9b…`.
- Split: `fold==test`, restricted to rows present in mask metadata → n = 25 596 cases.
- Preprocessing: grayscale, resize 256×256 bilinear, `(x/255 − 0.5)/0.5`; deformation via the
  same frozen generator, `mu=0.01`, `stochastic_lambda=0.0`, `transform_mode='corrected'`.
- Metrics computed per-case and per-structure with the **exact canonical implementations**
  from `eval_seg.py` (imported directly), aggregated as in `eval_seg.py` (sum / n_cases).

## 8. Dice / IoU / HD95 definitions

Explicitly verified against the canonical code (`eval_seg.py`):

- **Dice** = `(2·|G∩P| + 1e-7) / (|G| + |P| + 1e-7)`, computed on **binary masks at threshold
  0.5** (prediction `model(x) > 0.5`; ground truth already binary). Foreground convention:
  the structure mask itself. Empty-mask handling: `+1e-7` eps in numerator/denominator.
  Aggregation: **per-case Dice, then mean over cases** (matching the historical
  `eval_seg.py` `evaluate()`); reported aggregate = mean over the 3 structure means.
- **IoU** = `(|G∩P| + 1e-7) / (|G∪P| + 1e-7)`, same mask convention, same aggregation.
- **HD95** = two-sided 95th-percentile **Hausdorff distance in pixel units at 256×256**
  (scipy `distance_transform_edt`; `max(p95 fwd, p95 bwd)`). Empty-mask behavior: returns
  NaN if pred or target empty; **in this run there were 0 empty predictions and 0 empty masks
  on all 3 structures**, so HD95 aggregates are finite and complete. Disconnected components
  are handled implicitly by the EDT on the full binary mask (no component labeling).
- These exact semantics were preserved — **not silently changed** — so later method
  comparisons can use identical metric definitions.

## 9. Segmentation results

Per-case values for all 76 788 (case, structure) pairs are persisted in
`research_agent/03H_artifacts/segmentation/seg_per_case.csv`.

| Structure | Dice | IoU | HD95 (px) |
|---|---|---|---|
| Left Lung | 0.9434 | 0.8971 | 1.782 |
| Right Lung | 0.9527 | 0.9126 | 1.450 |
| Heart | 0.9321 | 0.8767 | 2.786 |
| **MEAN** | **0.9427** | **0.8955** | **2.006** |

Cross-checked against the canonical `eval_seg.py` CLI on the same generator/config:
per-structure values match to printed precision (0.9434/0.8971/1.782 … MEAN 0.9427/0.8955/2.006).

## 10. Complete corrected baseline table

| Metric | Direction | Corrected baseline |
|---|---|---|
| Adaptive Re-ID mean | ↓ | **0.739194** |
| Adaptive Re-ID SD | — | 0.049847 |
| Adaptive Re-ID max | ↓ | 0.803662 |
| Classification mean AUC (14) | ↑ | **0.781227** |
| Segmentation Dice | ↑ | **0.9427** |
| Segmentation IoU | ↑ | **0.8955** |
| Segmentation HD95 | ↓ | **2.006** |

## 11. Historical context

The values below are **context only** (legacy protocol, `transform_mode=legacy`), recorded
from `RESEARCH_BRIEF.md` / `PLAN.md`. They are **not** same-protocol comparisons with the
corrected baseline and are deliberately kept in a separate table.

| Utility axis | Raw images | baseline_fixed (legacy) |
|---|---|---|
| Classification mean AUC | ≈ 0.8050 | ≈ 0.773 |
| Segmentation Dice | ≈ 0.947 | ≈ 0.937 |
| Segmentation IoU | ≈ 0.905 | ≈ 0.886 |
| Segmentation HD95 | ≈ 1.74 | ≈ 2.27 |

## 12. Utility uncertainty

Preserved/estimated with minimal, defensible machinery (no invented hierarchy).

- **Classification:** per-label AUCs retained; mean over 14 labels = 0.781227. A
  straightforward **case-level bootstrap** (500 resamples, seed 42) of mean AUC-14 gives
  **95% CI [0.7765, 0.7854]**.
- **Segmentation:** per-case metrics retained (76 788 rows); case-level bootstrap (1000
  resamples, seed 42) of the reported 3-structure mean gives 95% CIs:
  - Dice **95% CI [0.9424, 0.9431]**, per-case mean 0.94272, SD 0.05186, median 0.95687
  - IoU **95% CI [0.8949, 0.8960]**, per-case mean 0.89545, SD 0.07879, median 0.91731
  - HD95 **95% CI [1.9778, 2.0337]**, per-case mean 2.00612, SD 3.85668, median 1.00000

No cases were selected on performance.

## 13. Margin inputs

Computed empirical inputs for later margin locking (δ/MME deferred to a short Scientist
review **after** STEP 3C; **no margins locked here**).

| Input | Observed value |
|---|---|
| Privacy: Re-ID sample SD (10 seeds) | 0.049847 |
| Privacy: Re-ID range (min 0.7927 … max 0.8037) | ≈ 0.011 spread around mean |
| Classification: mean AUC-14 | 0.781227 |
| Classification: bootstrap 95% CI | [0.7765, 0.7854] (half-width ≈ 0.0045) |
| Segmentation Dice: bootstrap 95% CI | [0.9424, 0.9431] (half-width ≈ 0.0004) |
| Segmentation IoU: bootstrap 95% CI | [0.8949, 0.8960] |
| Segmentation HD95: bootstrap 95% CI | [1.9778, 2.0337] |

Candidate statistically defensible scales for the Scientist review (proposals only, not
locked): privacy effect size on the order of the observed SD (≈ 0.05 AUC); classification
non-inferiority δ on the order of a few bootstrap half-widths (≈ 0.005–0.01 mean AUC);
segmentation margins on the order of the bootstrap half-widths scaled up (≈ 0.001–0.002 Dice
/ IoU, ≈ 0.05 px HD95). Final locking is deferred to the post-STEP 3C review.

## 14. Evidence bundle

All lightweight machine records tracked under `research_agent/03H_artifacts/`:

- `classification/aucs.csv` — per-label AUC (14 rows)
- `classification/preds.csv` — 25 596 rows of per-label probabilities (Image Index key)
- `classification/transform_mode.txt` — `corrected`
- `segmentation/seg_per_case.csv` — 76 788 rows (image_index, structure, dice, iou, hd95)

Machine-readable summaries (tracked):

- `research_agent/03H_corrected_classification.json`
- `research_agent/03H_corrected_segmentation.json`
- `research_agent/03H_corrected_baseline_utility_summary.json`

Emitters (tracked, deterministic, machine-produced):

- `research_agent/emit_03h_classification_machine.py`
- `research_agent/emit_03h_segmentation_machine.py`
- `research_agent/emit_03h_summary_machine.py`
- `research_agent/eval_seg_percase.py` (per-case recorder; imports canonical `eval_seg.py`)

Hashes of every evaluation component (generator, classifier, segmenter, splits, mask
metadata, configs, scripts) are embedded in the summary JSON. No model weights committed
beyond repository policy (only path + SHA-256).

## 15. Integrity checks

| Check | Result |
|---|---|
| Generator SHA unchanged before/after | PASS (`8a489eec…`) |
| Classifier checkpoint unchanged | PASS (`8ad15b38…`) |
| Segmentation checkpoint unchanged | PASS (`2dfdcf9b…`) |
| No training occurred | PASS — no new checkpoints created during STEP 3C |
| No configuration changed after seeing TEST results | PASS — eval config created before running |
| No synthetic/stub metrics | PASS — all values from real forward passes on real data |
| All expected cases/labels included | PASS — 14/14 labels, 25 596/25 596 test cases, 76 788 seg pairs |
| Outputs finite or explicitly explained | PASS — all finite; no empty masks/preds |
| No TEST overwrite/re-evaluation loop | PASS — test evaluated exactly once per axis |
| All pre-existing regression suites | PASS (stage-e, protocol, operator-repair, grad-accum) |

## 16. Problems / deviations

- The full-test empty-mask scan as a standalone pass timed out under the shell tool; empty
  masks were instead detected **during** the per-case segmentation evaluation (0 empty masks,
  0 empty predictions across all 3 structures), which is stronger evidence and needs no
  separate scan. No result impact.
- The classifier checkpoint preds.csv contains probabilities only (ground-truth labels are in
  `nih_labels.csv`); the two were joined for the independent AUC recomputation and bootstrap.
- No other deviations. The corrected-baseline evaluation used `transform_mode=corrected`
  (required by the locked generator semantics); this is the *only* intended difference from
  the legacy baseline_fixed evaluation.

## 17. STEP 3C verdict

STEP 3C CORRECTED UTILITY: PASS
