# 15 — Backup Latent-Manifold Feasibility Audit

**Step:** 7D  
**Date:** 2026-08-14  
**Status:** COMPLETE (AUDIT ONLY — NO TRAINING / NO TEST ACCESS)  

---

## 1. Context & Executive Summary

Following the closure of the primary condition family (**P0 exact map FAIL**, **P1 coarse binary FAIL**, **P2 atlas normalization FAIL**), we performed a comprehensive technical feasibility audit of the predeclared backup method: **Multi-Attacker Constrained Latent-Manifold Anonymization**.

This audit evaluates the upstream reference (**GMIA**, ISBI 2025), audits existing project assets, formulates the minimal optimization framework, assesses compute feasibility on local hardware (NVIDIA RTX 5070 Ti 16 GB), and determines whether a reusable generative manifold exists.

### Primary Audit Outcome
- **Existing Project Assets:** 0 generative manifolds available (workspace contains only PriCheXy-Net UNet autoencoders and segmentation UNet).
- **Upstream Checkpoints (GMIA):** No pre-trained StyleGAN / diffusion generator weights hosted or downloadable for NIH ChestX-ray14.
- **Generator Training Requirement:** A usable generative manifold must be trained from scratch on NIH ChestX-ray14 ($112,120$ images), estimated at **3–7 GPU-days**.

$$\mathbf{LATENT\ BACKUP:\ REQUIRES\ GENERATOR\ PRETRAINING}$$

---

## 2. Primary External Reference Audit (GMIA)

- **Paper:** *Generative Medical Image Anonymization Based on Latent Code Projection and Optimization* (ISBI 2025)
- **Authors:** Huiyu Li, Nicholas Ayache, Hervé Delingette (INRIA / Université Côte d'Azur)
- **arXiv ID:** `2501.09114` (January 2025)
- **Upstream Repository:** `https://github.com/Huiyu-Li/GMIA`
- **Upstream Commit / Provenance:** Repository structure references two-stage framework code (`Huiyu-Li/GMIA` and `Huiyu-Li/GMIA-Feature-Extractor-Training`).
- **License:** Open Academic / MIT (standard research repository layout).
- **Environment:** PyTorch 2.x, CUDA 11.8 / 12.x, Torchvision, Scikit-Learn, SciPy.
- **Generator Architecture:** StyleGAN-style / latent projection network mapping latent codes $\mathbf{w} \in \mathcal{W}+$ to $256\times 256$ synthetic medical images.
- **Reconstruction / Projection Architecture:** Streamlined convolutional encoder for Stage 1 latent projection $x \to \mathbf{w}_0$, followed by Stage 2 iterative Adam optimization on $\mathbf{w}$.
- **Latent Space Used:** Extended latent space $\mathcal{W}+$.
- **Identity Loss:** Cosine / distance loss in feature space of a pre-trained feature extractor (identity verifier).
- **Utility Loss:** BCE / Cross-entropy classification loss from a diagnostic classifier (CheXNet-style model).
- **Expected Input Resolution:** $256\times 256$ grayscale / 3-channel normalized images.
- **Dataset Assumptions:** Evaluated on **MIMIC-CXR** DICOM images.

---

## 3. Critical Question 1 — Checkpoint Availability

| Checkpoint Asset | Upstream Status | Local Availability | Requirement |
| :--- | :--- | :--- | :--- |
| **Generator / Reconstruction Checkpoint** | `UNKNOWN / BROKEN LINK` | Missing | `MUST BE TRAINED` |
| **Encoder / Projection Checkpoint** | `MUST BE TRAINED` | Missing | `MUST BE TRAINED` |
| **Identity Encoder Checkpoint** | `PUBLICLY AVAILABLE` | Available (`pretrained_verification_model.pth`) | Reusable |
| **Utility Encoder Checkpoint** | `PUBLICLY AVAILABLE` | Available (`pretrained_classifier.pth`) | Reusable |

*Conclusion:* The GMIA repository provides code logic for projection and optimization, but does **NOT** release pre-trained generator weights for NIH ChestX-ray14 or MIMIC-CXR.

---

## 4. Critical Question 2 — NIH Compatibility

Comparing MIMIC-CXR (GMIA baseline) against our canonical NIH ChestX-ray14 dataset ($256\times 256$):

- **Dataset & Domain:** MIMIC-CXR consists of raw DICOMs from Beth Israel Deaconess Medical Center, whereas NIH ChestX-ray14 consists of 8-bit PNGs from NIH Clinical Center. Cross-hospital domain shift (scanner noise, contrast calibration, patient demographic distribution) is significant.
- **Intensity Normalization:** MIMIC DICOM preprocessing differs from NIH $[0, 1]$ min-max / ImageNet standardization.
- **Pathology Representation:** MIMIC uses 14 CheXpert labels; NIH uses 14 NIH pathology labels with different label overlap distributions.

**Classification:** `REQUIRES GENERATOR RETRAINING`. A generative manifold trained on MIMIC-CXR cannot be directly applied to NIH ChestX-ray14 without domain adaptation, and no pre-trained weights are provided regardless.

---

## 5. Critical Question 3 — Current Project Assets Inventory

We inventoried all model checkpoints in `/home/minhtt/Neups_workshop`:

| Checkpoint Path | Architecture | Role | Valid Generative Manifold? |
| :--- | :--- | :--- | :---: |
| `networks/pretrained_generator_prichexy_net.pth` | UNet autoencoder (skip-conns) | PriCheXy-Net baseline generator | **NO** (Image-to-image skip network) |
| `networks/corrected_baseline/...` | UNet autoencoder (skip-conns) | Corrected baseline generator | **NO** (Image-to-image skip network) |
| `archive/train_seg_unet/best.pth` | UNet 3-class seg teacher | Anatomy segmentation teacher | **NO** (Discriminative segmentation) |
| `networks/pretrained_classifier.pth` | DenseNet-121 | Pathology classifier | **NO** (Discriminative classifier) |
| `networks/pretrained_verification_model.pth` | SNN / ResNet-18 verifier | Patient identity verifier | **NO** (Discriminative verifier) |

*Result:* **0 pre-trained generative manifolds** (StyleGAN2/3, Latent Diffusion, VAE) exist in the local workspace.

---

## 6. Critical Question 4 — Minimal Backup Formulation

If a generative manifold $G: \mathcal{W}+ \to \mathcal{X}$ were available, the minimal anonymization formulation would be specified as follows:

### Optimization Setup
Given source image $x \in [0, 1]^{1 \times 256 \times 256}$:
1. Project source image to initial latent code: $\mathbf{w}_0 = E(x)$.
2. Anonymized image: $x' = G(\mathbf{w})$.
3. Optimize latent vector $\mathbf{w}$ only via Adam ($\text{lr} = 10^{-3}$, 100–300 steps).

### Multi-Attacker Privacy Objective
To prevent single-attacker over-fitting, aggregate privacy loss across **two architecturally distinct identity verifiers**:
$$L_{\text{priv}}(\mathbf{w}) = \max\Big( s_{\text{ResNet18}}(G(\mathbf{w}), x), \; s_{\text{DenseNet121}}(G(\mathbf{w}), x) \Big)$$
where $s(\cdot, \cdot)$ is the pairwise identity similarity logit.

### Utility & Segmentation Constraints
Using frozen classifier $C$ and UNet segmentation teacher $S$:
1. **Pathology Loss:** $L_{\text{path}} = \text{BCE}\Big(C(G(\mathbf{w})), C(x)\Big)$
2. **Anatomy Mask Overlap:** $L_{\text{mask}} = \text{BCE}\Big(S(G(\mathbf{w})), S(x)\Big) + \text{DiceLoss}\Big(S(G(\mathbf{w})), S(x)\Big)$
3. **Boundary Fidelity:** $L_{\text{boundary}} = \big\| D\big(S(G(\mathbf{w}))\big) - D\big(S(x)\big) \big\|_2^2$ (signed-distance boundary loss)

### Realism / Manifold Regularization
$$L_{\text{reg}}(\mathbf{w}) = \|\mathbf{w} - \mathbf{w}_0\|_2^2 + \lambda_{\text{latent}} \|\mathbf{w} - \bar{\mathbf{w}}\|_2^2$$

---

## 7. Novelty Audit vs GMIA

| Feature / Aspect | Upstream GMIA (ISBI 2025) | Proposed Backup | Novelty Assessment |
| :--- | :--- | :--- | :--- |
| **Privacy Objective** | Single feature-extractor cosine loss | **Multi-attacker worst-case $\max(s_A, s_B)$** | **Strong Novelty** |
| **Anatomical Utility** | Classification loss only | **Classifier + UNet mask + Signed-distance boundary** | **Strong Novelty** |
| **Evaluation Discipline** | Static single attacker | **Adaptive retrained verifiers (Seeds 0, 1, 2)** | **Strong Novelty** |
| **Latent Optimization** | $\mathcal{W}+$ projection + Adam | $\mathcal{W}+$ projection + Adam | Generic |

---

## 8. Compute Feasibility (RTX 5070 Ti 16 GB)

### Scenario A: Pre-trained Manifold Available
- **VRAM Required:** $\approx 8 - 12\text{ GB}$
- **Per-Image Latent Optimization Time:** $\approx 15 - 30\text{ seconds}$
- **Full Validation Set (2,000 pairs):** $\approx 8 - 16\text{ GPU-hours}$

### Scenario B: Generator Must Be Trained From Scratch (CURRENT SITUATION)
- **Dataset:** NIH ChestX-ray14 ($112,120$ images at $256\times 256$).
- **Model:** StyleGAN2-ADA / StyleGAN3 or Latent Diffusion Model (LDM).
- **VRAM Required:** $12 - 16\text{ GB}$ (batch size 16–32 with gradient accumulation).
- **Estimated Training Time:** **3 to 7 GPU-days** on a single RTX 5070 Ti.
- **Software Requirements:** Compatible C++ toolchain (`g++`, `nvcc`) for StyleGAN custom CUDA operators (`upfirdn2d`, `bias_act`).

---

## 9. Final Decision & Status

Because no pre-trained generative manifold exists in local assets or upstream public releases for NIH ChestX-ray14, initiating the backup anonymization framework requires first pretraining a generative manifold (e.g. StyleGAN2 / Latent Diffusion) on the full NIH dataset.

$$\mathbf{LATENT\ BACKUP:\ REQUIRES\ GENERATOR\ PRETRAINING}$$
