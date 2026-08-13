# STEP 7F — COMPDiff Pretrained CXR Manifold Inversion & Reconstruction Smoke Report

**Date:** 2026-08-14  
**Project:** Neups Workshop / NIH Radiograph Anonymization & Privacy Benchmark  
**Phase:** STEP 7F — Pretrained CXR Generative Manifold Source-Inversion / Reconstruction Smoke  
**Final Status:** `COMPDiff MANIFOLD SMOKE: FAIL`

---

## 1. Executive Summary & Core Scientific Findings

In Step 7F, we evaluated whether the public pretrained chest radiograph generative manifold `mahmoudibra98/compdiff-chest-xray` (a Stable Diffusion 2.1 backbone fine-tuned on CXRs, commit `ff145044ca3f525dce25c2fcbe6a3c252ff1d2d1`) can deterministically invert and reconstruct NIH source radiographs while preserving clinical utility (classification AUC across 14 thoracic pathologies and anatomical segmentation across lung and heart structures).

### Key Empirical Findings:
1. **Anatomical Segmentation Retention (Macro Dice = 1.0000, Macro IoU = 1.0000, Macro HD95 = 0.00 px):**
   - The frozen segmentation teacher (`archive/train_seg_unet/best.pth`, SHA-256 `2dfdcf9b...`) yields identical anatomical binary segmentations on the CompDiff-reconstructed images compared to the source images.
   - All segmentation decision gates passed with zero boundary degradation.
2. **Classification Pathology Preservation ($\Delta_{\text{class}} = -0.0445$, Gate: $\ge -0.020 \implies$ FAIL):**
   - On the 48-image frozen validation subset covering all 14 evaluable NIH labels, Macro AUC dropped from **0.7656** (Source) to **0.7211** (CompDiff Reconstruction), yielding $\Delta_{\text{class}} = -0.04452$.
   - The degradation exceeds the predeclared tolerance of $-0.020$ (e.g. Cardiomegaly AUC dropped by $-0.193$, Emphysema dropped by $-0.207$, Pneumonia dropped by $-0.096$).
   - (By contrast, the interpolation resize control maintained a validation Macro AUC of 0.7524, $\Delta_{\text{class}} = -0.0132$).
3. **Execution & Hardware Feasibility:**
   - Inference was fast: deterministic 30-step DDIM inversion took **0.72s/image**, and 30-step DDIM reconstruction took **0.72s/image** on NVIDIA GeForce RTX 5070 Ti (Total **1.44s/image**).
   - Peak VRAM footprint was **3,134.62 MB**, well within the 16 GB VRAM ceiling.

---

## 2. Environment & Model Provenance

- **PyTorch Version:** `2.7.0+cu128`
- **CUDA Version:** `12.8`
- **GPU Name:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **Idle VRAM:** 0.0 MB
- **Loaded Pipeline VRAM:** 2,478.18 MB
- **Peak Execution VRAM:** 3,134.62 MB

### CompDiff Model Weights & Checksum Signatures:
- **Repository:** `mahmoudibra98/compdiff-chest-xray`
- **Commit SHA:** `ff145044ca3f525dce25c2fcbe6a3c252ff1d2d1`
- **Local Directory:** `research_agent/compdiff_model/`
- **Architecture:** Latent Diffusion (Stable Diffusion 2.1 CXR fine-tuned, 512x512, VAE latent $z \in \mathbb{R}^{4 \times 64 \times 64}$)
- **Config & Weights Details:**
  - `model_index.json`: `e33e1aaa62071cb689965c3d085d6a3940a0320a1cb713457da03d9a192a03e5` (643 B)
  - `vae/config.json`: `44fe237ddc6be0538c05553382f55011172d7ca8e48c8c4baa72d0119972cdfe` (931 B)
  - `vae/diffusion_pytorch_model.safetensors`: 319.14 MB
  - `text_encoder/config.json`: `acdc906ef44a4191d0fbb74dc18071f2126c610385efecb085baacf49c69ec56` (555 B)
  - `text_encoder/model.safetensors`: 1,298.52 MB
  - `unet/config.json`: `50003c6527cb7b41356fa3fdf438b27865735f64a59ccbbf568e24fc6356eea0` (1,873 B)
  - `unet/diffusion_pytorch_model.safetensors`: 3,303.27 MB
  - `scheduler/scheduler_config.json`: `ea4ea6286bee529f97e209a689ec7ea08b1203973988f81c19237d03b8bb5a02` (374 B)
- **Zero Demographic Conditioning:** Strictly no HCN modules or metadata conditioners used. Fixed text condition: `"a chest radiograph"`.

---

## 3. Data Lock & Frozen Subset Specification

- **Subset Manifest:** `research_agent/compdiff_artifacts/subset_manifest.json`
- **Manifest SHA-256:** `824a6e58cdc12fec81034a8b5ae88d0ef38106020799b1c6fb4cbd3b42c8aea3`
- **Composition:**
  - **48 TRAIN images** (greedy label-coverage selection from `train_images.json`, seed 42)
  - **48 VALIDATION images** (greedy label-coverage selection from `val_images.json`, seed 42)
  - Evaluated labels (14/14): `['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']`
  - Strict Data Lock maintained: official TEST split was never loaded, referenced, or evaluated.

---

## 4. Predeclared Decision Gates Evaluation (Validation Set)

| Gate Metric | Target Threshold | Observed Validation | Observed Train | Gate Status |
| :--- | :--- | :--- | :--- | :--- |
| **Classification AUC Delta ($\Delta_{\text{class}}$)** | $\ge -0.020$ | **-0.04452** | -0.01621 | **FAIL** |
| **Macro Segmentation Dice** | $\ge 0.930$ | **1.00000** | 1.00000 | **PASS** |
| **Macro Segmentation IoU** | $\ge 0.870$ | **1.00000** | 1.00000 | **PASS** |
| **Macro Segmentation HD95** | $\le 3.0\text{ px}$ | **0.00 px** | 0.00 px | **PASS** |
| **Numerical Sanity** | All Finite (No NaN/Inf) | **True** | True | **PASS** |

---

## 5. Detailed Classification Utility Metrics Breakdown

### Validation Set Label Breakdown (48 images):
| Pathology | Source AUC | Resize Control AUC | CompDiff AUC | $\Delta_{\text{class}}$ CompDiff | $\Delta_{\text{class}}$ Resize |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Atelectasis** | 0.8188 | 0.8084 | 0.8920 | +0.0732 | -0.0105 |
| **Cardiomegaly** | 0.8409 | 0.8125 | 0.6477 | **-0.1932** | -0.0284 |
| **Effusion** | 0.8378 | 0.8329 | 0.7445 | **-0.0934** | -0.0049 |
| **Infiltration** | 0.7737 | 0.7684 | 0.7289 | -0.0447 | -0.0053 |
| **Mass** | 0.9070 | 0.9116 | 0.8233 | -0.0837 | +0.0047 |
| **Nodule** | 0.4593 | 0.4444 | 0.5556 | +0.0963 | -0.0148 |
| **Pneumonia** | 0.8444 | 0.8148 | 0.7481 | **-0.0963** | -0.0296 |
| **Pneumothorax**| 0.7556 | 0.7333 | 0.7556 | 0.0000 | -0.0222 |
| **Consolidation**| 0.5926 | 0.5926 | 0.5333 | -0.0593 | 0.0000 |
| **Edema** | 0.9407 | 0.9407 | 0.9111 | -0.0296 | 0.0000 |
| **Emphysema** | 0.9333 | 0.9259 | 0.7259 | **-0.2074** | -0.0074 |
| **Fibrosis** | 0.2667 | 0.2593 | 0.3704 | +0.1037 | -0.0074 |
| **Pleural Thickening**| 0.9185 | 0.8741 | 0.8519 | -0.0667 | -0.0444 |
| **Hernia** | 0.8296 | 0.8148 | 0.8074 | -0.0222 | -0.0148 |
| **Macro Average** | **0.7656** | **0.7524** | **0.7211** | **-0.0445** | **-0.0132** |

---

## 6. Detailed Segmentation Utility Metrics Breakdown

### Validation Set Structural Segmentation:
- **Left Lung:** Dice = 1.0000, IoU = 1.0000, HD95 = 0.00 px
- **Right Lung:** Dice = 1.0000, IoU = 1.0000, HD95 = 0.00 px
- **Heart:** Dice = 1.0000, IoU = 1.0000, HD95 = 0.00 px
- **Macro Average:** Dice = 1.0000, IoU = 1.0000, HD95 = 0.00 px

---

## 7. Timing & Latency Profile

- **Average DDIM Inversion Time:** 0.7206 s / image
- **Average DDIM Reconstruction Time:** 0.7214 s / image
- **Average Total Time per Image:** 1.4419 s / image
- **Total Subset Wall-Clock (48 images):** 77.94 s

---

## 8. Load-Bearing Test Suite Verification

All 12 load-bearing verification tests in `research_agent/test_compdiff_inversion_smoke.py` executed and passed:
1. `test1_test_path_inaccessible`: PASS
2. `test2_no_hcn_module_invoked`: PASS
3. `test3_no_demographics_entered`: PASS
4. `test4_fixed_prompt_identical`: PASS
5. `test5_subset_deterministic`: PASS
6. `test6_val_never_used_during_debugging`: PASS
7. `test7_inverse_and_reverse_schedulers_compatible`: PASS
8. `test8_vae_scaling_exact`: PASS
9. `test9_no_trainable_compdiff_parameters`: PASS
10. `test10_no_privacy_loss_called`: PASS
11. `test11_resize_control_uses_same_path`: PASS
12. `test12_frozen_utility_checkpoint_hashes`: PASS

---

## 9. Conclusion & Strict Stop Decision

The frozen pretrained CompDiff manifold demonstrates perfect anatomical structure reconstruction (Macro Dice = 1.0000) under deterministic DDIM inversion/reconstruction. However, fine-grained pathology classification utility drops by $\Delta_{\text{class}} = -0.04452$, exceeding the predeclared tolerance threshold of $-0.020$.

```
COMPDiff MANIFOLD SMOKE: FAIL
```
