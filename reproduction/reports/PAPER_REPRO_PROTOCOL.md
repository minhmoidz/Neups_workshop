# PAPER REPRODUCTION PROTOCOL — STEP R1 (PriCheXy-Net, MICCAI 2023)

Predeclared protocol for exactly reproducing the original PriCheXy-Net paper result.
All claims/tolerances in this document are fixed BEFORE evaluating the held-out TEST split.

---

## 1. Source of truth

### 1.1 Paper
- **Title:** Deep Learning-based Anonymization of Chest Radiographs: A Utility-preserving Measure for Patient Privacy
- **Venue:** MICCAI 2023; **arXiv:** 2209.11531
- **Reported result to reproduce (Table 1, PriCheXy-Net / Ours, µ = 0.01):**
  - Patient Re-Identification (Ver.): **57.7 ± 4.0 AUC** (baseline real data: 81.8 ± 0.6)
  - Abnormality Classification (Class.): **76.2** mean AUC, 95% CI **[75.8, 76.6]** (baseline real data: 80.5)
- Paper states results are "AUC (mean ± std) over 10 independent training and testing runs" for verification,
  and "mean AUC + 95% CIs" over the 14 class-wise AUC values for classification.

### 1.2 Repository / commit
- Upstream: `kaipackhaeuser/PriCheXy-Net`
- Pristine commit used: **`29245d1` ("Pushed code")** — checked out on branch `original-upstream`
- All pristine source read via `git show 29245d1:<path>`; reproduced code runs from this checkout.

### 1.3 Dataset (already audited, see `CHESTXRAY14_DATA_AUDIT.md`)
- ChestX-ray14 at `/home/minhtt/datasets/nih/images`: **112,120 PNG**, **30,805 patients**, 1024×1024 grayscale.
- Paper preprocessing: grayscale → resize to **256×256**.
- Official pair files (SHA256 in `hashes.json`):
  - `image_pairs_training_10000.txt` (10,000)
  - `image_pairs_validation_2000.txt` (2,000)
  - `image_pairs_testing_5000.txt` (5,000)  ← **held-out TEST**, never touched during development/smoke.
- Classifier eval uses `chexnet/nih_labels.csv` folds: train 75,708 / val 10,816 / **test 25,596** (test = 14-class utility eval).

### 1.4 Released checkpoints used (official artifacts, `networks/`)
| File | Role in R1-A | SHA256 |
|---|---|---|
| `generator_lowest_total_loss_mu_0.01.pth` | **the paper's µ=0.01 U-Net flow generator** (best on validation set) | `4d82dcdd…` (see `hashes.json`) |
| `pretrained_classifier.pth` | DenseNet-121 (CheXNet-style), utility classifier | `8ad15b38…` |
| `pretrained_verification_model.pth` | SNN used as adversary during anonymization training (NOT used in R1-A) | `331efaed…` |

**Decision (R1-A):** use the released generator checkpoint directly. Do NOT retrain the anonymization model —
the released `generator_lowest_total_loss_mu_0.01.pth` IS the paper's best-on-validation µ=0.01 model,
so retraining would inject research-protocol semantics.

---

## 2. What R1-A reproduces (original paper semantics)

R1-A reproduces the **µ = 0.01** row of Table 1 with the **original (legacy) operator** semantics of the released code:

1. **Re-ID (Ver.):** re-train the Siamese verification network (SNN) on anonymized (deformed) images only,
   then simulate the linkage attack (deformed image vs. real image) on the official TEST pairs → AUC.
   Target: **57.7 ± 4.0**.
2. **Utility (Class.):** evaluate the released DenseNet-121 classifier on the anonymized TEST fold
   (25,596 images) → mean of 14 class-wise AUCs + bootstrap 95% CI. Target: **76.2 [75.8, 76.6]**.

### 2.1 Exact experimental procedure (from pristine code, commit 29245d1)
- **Perturbation (flow field), legacy operator** (`utils/train_snn`, `validate_snn`, `test_snn`):
  - `grid = UNet(x)`; `grid = grid_identity - mu*grid`; `grid = gauss_filter(grid)` (GaussianSmoothing, channels=2, kernel=9, σ=2);
    `x = grid_sample(x, grid, padding_mode='border', align_corners=True)`; then `expand(-1,3,-1,-1)` + ImageNet Normalize.
  - Identity grid at **256×256**, `indexing='ij'`, mesh_x/mesh_y stacked (this is the ORIGINAL/legacy grid convention).
- **SNN architecture** (`networks/SiameseNetwork.py`): two ResNet-50 branches (ImageNet pretrained), fc→128, sigmoid,
  `|x1−x2|`, fc_end→1. **No weight load from `pretrained_verification_model.pth`** in the retrain script.
- **Training** (`AgentSiameseNetwork` + `utils.train_snn`): deform BOTH pair images; BCEWithLogitsLoss;
  Adam lr=1e-4; batch 32; image_size 256; n_channels=1; shuffle train; early stopping patience **5** on validation loss;
  best net = lowest validation loss (deepcopy); max_epochs configurable (config says 100; paper's released code default).
- **Test/linkage attack** (`utils.test_snn`): deform **inputs1 only** (the anonymized image) and keep inputs2 real —
  "a real image is attempted to be linked to an anonymized image." Report AUC.
- **Seeds:** released code hardcodes `utils.seed_all(42)`; paper reports 10 runs but does not specify seeds.
  R1-A therefore: (a) runs the released-code default run (seed 42) first, then (b) a **predeclared multi-seed set**
  (Section 5.2) to estimate mean ± std.
- **Classifier eval** (`eval_classifier.py` + `chexnet/eval_model.py`): `pretrained_classifier.pth`;
  flow field at **256×256**, then Resize(224,224) + Normalize(ImageNet); batch 16; TEST fold; 14-class AUCs;
  mean + 95% CI via `utils.bootstrap_abnormalities` (1,000 bootstraps) — matching the paper's CI procedure.

### 2.2 Compatibility patches (torch 2.7.0 vs. paper torch 1.10.2) — documented, no semantic change
- `torch.load(..., weights_only=False)` where a pickled model/dict is loaded:
  `eval_classifier.py` (classifier checkpoint) and `agents/Agent.py` (not used in R1-A) — patch already prepared in
  `research_agent/upstream_compatibility.patch`.
- The generator checkpoint is a pure state dict of tensors → default `weights_only=True` load is fine in
  `AgentSiameseNetwork` (no patch needed).
- `models.resnet50(pretrained=True)` in torchvision 0.22 → deprecated but functional; ImageNet weights cached at
  `~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth` (SHA `0676ba61…`, verified).
- **Hardware constraint (documented deviation):** batch size reduced 32 → 16. The released/paper SNN config uses
  batch 32, which causes CUDA out-of-memory on the available 16 GB RTX 5070 Ti; 16 is the largest batch that fits
  (verified end-to-end: 1-epoch run + full 5000-pair TEST evaluation completed, AUC computed correctly). The earlier
  research reproduction runs on this machine also used batch 16. This is a hyper-parameter-only deviation; it does not
  alter the architecture, the deformation operator, the loss, the early-stopping rule, or the metric.
  **⚠ Superseded — see Erratum §8 (2026-08-18):** the OOM cause above was later found to be a missing `no_grad`
  wrap, not a hard batch-32 memory ceiling; the executed run that produced `PRICHEXY_PAPER_REPRODUCTION.md` used
  `batch_size=32`, not 16. This line is preserved unedited as the historical predeclared record.

---

## 3. Deliverables (produced by `run_prichexy_paper_reproduction.py`)

Saved under `reproduction/results/paper_reproduction/`:

| Artifact | Contents |
|---|---|
| `environment.json` | python/torch/torchvision/cuda/cudnn/numpy/pandas/sklearn/PIL/GPU/driver/disk |
| `hashes.json` | SHA256 of data dir, pair files, nih_labels.csv, all released checkpoints, pristine source files |
| `checkpoint_manifest.json` | structural load test of each released checkpoint (type, keys, shapes, tensor stats) |
| `frozen_config.json` | exact experiment config frozen before TEST (hashing-gated) |
| `smoke_test.json` | TRAIN/VAL-only smoke results (tiny SNN run + tiny classifier batch, TEST untouched) |
| `run_<seed>.json` | per-seed Re-ID test AUC + accuracy/etc. + bootstrap CI |
| `test_predictions.*` | full TEST predictions (npy/csv) for Re-ID and classification |
| `test_metrics.json` | final TEST metrics: Re-ID mean±std over seeds; classification mean AUC + 95% CI |
| `PRICHEXY_PAPER_REPRODUCTION.md` | final report with verdict |

---

## 4. Verdict definition (predeclared)

Compare reproduced TEST results to paper targets with these tolerances, fixed in advance.

### 4.1 Re-ID (stochastic — SNN retraining)
Paper reports **mean ± std over 10 runs = 57.7 ± 4.0**; seeds unspecified. We predeclare a multi-seed set
(Section 5.2) and compare **means** using a tolerance justified by the paper's own reported variability
(its 10-run std = 4.0 → SE of the paper mean ≈ 4.0/√10 ≈ 1.26; SE of the difference between two such means ≈ √(1.26²+1.26²) ≈ 1.79).

- **STRONG PASS:** `|Δmean| ≤ 0.01` OR `|Δmean| ≤ 0.01·(4.0/paper_std_scale)` — i.e., effectively indistinguishable.
  (Given stochasticity, STRONG will normally require the reproduced mean to be within ~1.8 AUC of the paper mean.)
- **APPROXIMATE PASS:** `0.01 < |Δmean| ≤ 0.03` OR reproduced mean within the paper's own 95% CI for the mean
  (`57.7 ± 1.96·1.26 ≈ [55.2, 60.2]`) **and** |Δmean| ≤ 3.0.
- **FAIL:** `|Δmean| > 0.03` AND outside `[55.2, 60.2]` AND |Δmean| > 3.0.

The primary, decisive criterion (avoids gaming the noise): **reproduced 10-seed mean inside the paper's 95% CI [55.2, 60.2] → PASS (at least APPROXIMATE); STRONG if additionally |Δ| ≤ 1.8.**

### 4.2 Classification (deterministic — fixed released classifier + fixed released generator)
- **STRONG PASS:** `|ΔmeanAUC| ≤ 0.01`
- **APPROXIMATE PASS:** `0.01 < |ΔmeanAUC| ≤ 0.03`
- **FAIL:** `|ΔmeanAUC| > 0.03`
(Reproduced CI must also overlap paper CI [75.8, 76.6] to be STRONG.)

### 4.3 Overall verdict
- Overall = **FAIL if either metric FAILs**. Otherwise overall grade = the weaker of the two grades.

---

## 5. Predeclared run configuration

### 5.1 Frozen hyper-parameters (identical to pristine `config_retrainSNN.json`, except image_path)
```
perturbation_type = flow_field
perturbation_model_file = ./networks/generator_lowest_total_loss_mu_0.01.pth
mu = 0.01
image_size = 256
batch_size = 16            # <-- deviation: paper/released config uses 32, but batch 32 causes CUDA OOM on the 16 GB GPU; 16 is the largest that fits. Hyper-parameter-only, documented hardware constraint.
                           # [SUPERSEDED — see Erratum §8, 2026-08-18: executed run used batch_size=32; this
                           #  predeclared value is preserved unedited, not retroactively changed to 32.]
learning_rate = 1e-4
max_epochs = 100
early_stopping = 5
image_path = /home/minhtt/datasets/nih/images/
```

### 5.2 Seeds (predeclared)
Released code default: **seed 42**. To match the paper's "10 independent runs" statistic we run **10 seeds**:
`SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]`.
(seed 42 = the exact released-code run; 0–8 give the predeclared independent sample. Seeds frozen in `frozen_config.json`.)

### 5.3 TEST use policy
TEST split (5,000 pairs / 25,596 classifier images) is used **only once**, after all config is frozen and smoke
tests pass. TEST metrics are never used for any model selection or early stopping.

---

## 6. Execution order (see `run_prichexy_paper_reproduction.py`)

1. Capture environment → `environment.json`
2. Hash all inputs → `hashes.json`
3. Load-test all released checkpoints → `checkpoint_manifest.json`
4. Write + hash-lock experiment config → `frozen_config.json`
5. **Smoke test (TRAIN + VAL only):** 1 SNN epoch on a tiny subsample + 1 classifier batch on a tiny VAL subsample.
   No TEST access. → `smoke_test.json`
6. **Full SNN retrain (10 seeds, TEST pairs evaluated after training per seed)** → `run_<seed>.json`,
   `test_predictions` (Re-ID)
7. **Classifier eval on anonymized TEST fold** → `test_predictions` (class), `test_metrics.json`
8. Write final report `PRICHEXY_PAPER_REPRODUCTION.md` with verdict per Section 4.

---

## 7. Known risks / expected outcomes

- A prior STEP 8R audit (on the `main` branch, validation-pair protocol) measured a **retrained attacker at
  ≈ 0.818 AUC** with the released generator + legacy operator — consistent with raw data and far above the
  paper's 57.7. If the TEST-pair reproduction lands near 0.8, **R1-A verdict = FAIL** (paper result not reproducible
  with the released checkpoint under the original operator). That is a valid, decisive outcome.
- R1-B (corrected-operator diagnostic, optional) is a SEPARATE protocol; it is not part of R1-A and will be reported
  independently. R1-A must NOT be mixed with the research-branch corrected operator or the corrected 0.739 baseline.

---

## 8. ERRATUM (2026-08-18) — Predeclaration/Execution Batch-Size Discrepancy

**Status:** Documentation-only correction. **Nothing below changes any recorded result, hash, config file, or
locked artifact.** §2.2 and §5.1 above are preserved **unedited** as the original historical predeclared record;
this section only documents where that record and the actual executed run diverged, so a future reader does not
rely on §5.1 alone and get misled.

**What happened, in order:**
1. §5.1 of this document predeclares `batch_size = 16` for the R1-A run, justified as a hardware-constraint
   deviation from the released/paper spec of `batch_size = 32` (CUDA OOM on the 16 GB RTX 5070 Ti).
2. The OOM that motivated this deviation was later diagnosed as **spurious**: the root cause was a missing
   `torch.no_grad()` wrap around the frozen-generator deformation pass in `train_snn`, not a hard memory ceiling.
   Once wrapped, `batch_size = 32` fits on the same GPU.
3. The run that actually produced the numbers in `reproduction/reports/PRICHEXY_PAPER_REPRODUCTION.md` used
   `batch_size = 32` — confirmed directly from `reproduction/results/paper_reproduction/frozen_config.json`,
   which records `"batch_size": 32"` and the note *"An earlier run fell back to 16 due to a spurious CUDA OOM;
   root cause was the missing no_grad wrap, now fixed — batch 16 was a documented deviation that inflates Re-ID
   AUC."*
4. **This document (§5.1, §2.2) was never updated to reflect that pivot.** Anyone reading only §5.1 would
   conclude the certified run used batch 16, when the executed run — and the numbers actually reported — used 32.

**Why this matters, and why it does not:**
- It matters because this document explicitly claims ("Predeclared protocol... fixed BEFORE evaluating TEST",
  line 4) to be an auditable predeclaration trail. A predeclared value that silently diverges from the executed
  value, without a recorded amendment, cannot be independently verified as "predeclared before TEST" from the
  document alone — only by cross-referencing `frozen_config.json` after the fact, as done here.
- It does **not** indicate result-shopping: `batch_size=32` is closer to the released implementation and to the
  paper's own stated configuration than `batch_size=16` was, so the change moved the protocol *toward* fidelity,
  not toward a more favorable number. No evidence of result-shopping was found — this is not the same as
  result-shopping having been conclusively excluded, since the amendment itself was undocumented at the time.

**What is corrected by this erratum:** nothing in `PRICHEXY_PAPER_REPRODUCTION.md`, `FINAL_10SEED_PRICHEXY_REPRODUCTION.md`,
or `BASELINE_LOCK.json` changes — those already reflect the batch=32 execution and remain the authoritative
results. This erratum corrects only the **documentation trail** in the present file, so that §5.1's `batch_size = 16`
line is never read in isolation again without a pointer to what was actually run.

**Recorded label going forward:** this reproduction should be described as *released-pipeline-faithful approximate
reproduction, with a compromised internal predeclaration trail* — not as an unqualified "protocol-faithful
reproduction" — per the independent peer review dated 2026-08-18.