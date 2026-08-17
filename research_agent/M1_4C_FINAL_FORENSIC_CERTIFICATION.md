# M1.4c FINAL FORENSIC CERTIFICATION
## Complete Execution Semantics, Cryptographic Lineage, Split Provenance, Numerical Robustness, Independent Parity, and Artifact Isolation Proof

> **Notice**: Historical M1.4c certification record. M1.4c.3 boundary closeout supersedes its disposition; this document is evidence, not execution authority.

**Date**: August 16, 2026  
**Auditor**: Antigravity Autonomous Research Agent (Forensic Certification Subagent)  
**Host Environment**: Linux x86_64, NVIDIA GeForce RTX 5070 Ti  
**Python Runtime**: 3.10.12 | **PyTorch Runtime**: 2.7.0+cu128 | **CUDA**: 12.8 | **cuDNN**: 90701  
**Target Canonical Commit**: `c6431310061c04e54dce82d30ae6e0ce24440562` (branch `research/method-restart`)  
**Certified Execution Code Lineage**: `851c3f1a6912255c97345a7f53ed138e7ae7981d`  
**Pristine Base Commit**: `29245d1f71571898d9527417df4ae3f63a8695f6`  
**Audit Verification Branch**: `audit/m2-final-certification`  
**Certification Status**: **HISTORICAL EVIDENCE ONLY — NOT A SCIENTIFIC EXECUTION CERTIFICATION**

---

## §1 Executive Summary & Cryptographic Lineage

This document establishes the final, immutable forensic certification for the M2-S1 scientific benchmark experiment in the `Neups_workshop` repository. The purpose of M1.4c is to audit and certify the complete pipeline—from data splits and hash provenance to GPU execution semantics, loss parity with zero-dependency pristine references, gradient ownership, non-vacuous assertions, and zero test-set access—prior to authorizing any scientific execution of M2.

### Cryptographic Repository Lineage
- **Canonical Remote**: `origin` (`https://github.com/minhmoidz/Neups_workshop.git`)
- **Canonical Branch**: `research/method-restart`
- **Frozen Canonical HEAD SHA**: `c6431310061c04e54dce82d30ae6e0ce24440562`
- **Audit Branch**: `audit/m2-final-certification`
- **Safety Invariant**: The canonical scientific branch `research/method-restart` has remained byte-for-byte untouched throughout the audit process.

---

## §2 Protocol Lock Hierarchy & Superseded Authority

To eliminate ambiguity across legacy certification artifacts, the protocol authority hierarchy has been explicitly locked (M1.4c.2 reconciliation):
1. **Supreme Scientific Authority**: `M2_S1_EXECUTION_LOCK.json` (scientific method / frozen scientific execution choices). `M1_4C_CERTIFICATION_MANIFEST.json` is certification evidence derived from those choices and must NOT override scientific method hyperparameters.
2. **Authoritative Artifacts**:
   - `M1_4C_FINAL_PARITY_CERTIFICATION.json` and `M1_4C_CERTIFICATION_MANIFEST.json` provide the authoritative record of all frozen checksums, split definitions, determinism bounds, and architectural invariants.
3. **Superseded Artifacts**:
   - `M1_C4_PROTOCOL_LOCK.json` has been formally marked as **SUPERSEDED** (contains stale checksum pointers superseded by M1.4c).

---

## §3 Python & PyTorch Runtime Determinism

The execution environment enforces deterministic execution invariants:
- **PyTorch Seeding**: `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `torch.cuda.manual_seed(seed)`
- **NumPy & Standard Library Seeding**: `np.random.seed(seed)`, `random.seed(seed)`
- **CuDNN Determinism Policy**: `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`
- **Deterministic Algorithm Support**: Micro-certification verified that strict PyTorch `torch.use_deterministic_algorithms(True)` is unsupported on CUDA due to `upsample_bilinear2d_aa_backward_out_cuda` in `transforms.Resize((224, 224))`. The runtime operates under `cudnn.deterministic = True, cudnn.benchmark = False` with a characterized numerical reproducibility envelope.

---

## §4 CUDA Hardware & Driver Profile

The runtime hardware envelope was characterized on the dedicated GPU host:
- **GPU Device**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **CUDA Runtime Version**: 12.8
- **cuDNN Version**: 90701
- **Driver**: NVIDIA Linux x86_64 Driver
- **Measured Envelope**: Two independent full forward/backward micro-certification passes yielded:
  - Loss differential: `0.0`
  - Max gradient differential: `4.445202648639679e-05`
  - Verifier / Classifier gradient differential: `0.0`

---

## §5 Frozen Checkpoint Integrity Table

All pretrained checkpoints consumed during training and evaluation are cryptographically pinned and verified before execution:

| Checkpoint Name | Relative Workspace Path | SHA-256 Digest |
| :--- | :--- | :--- |
| **Pretrained Initial Generator** | `networks/pretrained_generator_prichexy_net.pth` | `101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb` |
| **Pretrained Verifier Critic** | `networks/pretrained_verification_model.pth` | `331efaed0c0433c69941ddc003a14a936c688d94fd4ecfbefd34e53bfa7c051a` |
| **Pretrained Classifier Critic** | `networks/pretrained_classifier.pth` | `8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663` |
| **Repaired ACLoss Module** | `research_agent/m0_port/ACLoss.py` | `3ed8483718c3ccffb59f76e9dece47e92295a553895e3fd43b1b18cd486b263c` |

---

## §6 Frozen Config Manifest & Parameter Matrix

The frozen hyperparameter configurations for Control (`B_dev`) and Feature-Preserving Anonymizer (`C4`) are:

| Parameter | Control Arm (`B_dev`) | Feature Arm (`C4`) |
| :--- | :--- | :--- |
| **Config Path** | `config_files/config_dev_restored_baseline.json` | `config_files/config_dev_c4.json` |
| **Config SHA-256** | `14d3943f798d5855b4a49d55ecc6af858647f514f30c1cc7c803c7edebab30b6` | `7cbdfce84e41317dac73651d0d7d6080cf68871e0340a68ec0d9191383716a8a` |
| **Base Learning Rate ($\eta_G$)** | `0.0001` (Adam) | `0.0001` (Adam) |
| **Verifier Critic LR ($\eta_V$)** | `0.0001` (Adam) | `0.0001` (Adam) |
| **Classifier Critic LR ($\eta_C$)**| `0.0001` (SGD, momentum=0.9, weight_decay=1e-4) | `0.0001` (SGD, momentum=0.9, weight_decay=1e-4) |
| **AC Loss Weight ($\lambda_{AC}$)** | `1.0` | `1.0` |
| **Verifier Loss Weight ($\lambda_V$)**| `1.0` | `1.0` |
| **Feature Loss Weight ($\lambda_{feat}$)**| `0.0` | `1.0` |
| **Warp Factor ($\mu$)** | `0.01` | `0.01` |
| **Gaussian Smoothing ($\sigma, k$)** | $\sigma=2.0, k=9$ | $\sigma=2.0, k=9$ |
| **Epochs / Batch Size** | `250` epochs / DataLoader batch_size = 16 pair samples (each batch holds up to 16 pairs = up to 32 image tensors) | `250` epochs / DataLoader batch_size = 16 pair samples (each batch holds up to 16 pairs = up to 32 image tensors) |
| **Attacker Architecture** | Fresh ResNet50 Siamese Network | Fresh ResNet50 Siamese Network |
| **Attacker Config Path & SHA** | `config_files/config_dev_attacker_s1.json` (`72923582e659...`) | `config_files/config_dev_attacker_s1.json` (`72923582e659...`) |
| **Attacker Epochs / Patience / LR** | `100` max epochs / `5` patience / `1e-4` / batch `32` / seed `42` | `100` max epochs / `5` patience / `1e-4` / batch `32` / seed `42` |
| **Attacker Training Geometry** | anon($x_1$), anon($x_2$) | anon($x_1$), anon($x_2$) |
| **Attacker Val Checkpoint Geometry** | anon($x_1$), anon($x_2$) | anon($x_1$), anon($x_2$) |
| **Scientific Privacy Geometry** | anon($x_1$), real($x_2$) | anon($x_1$), real($x_2$) |

---

## §7 Split Provenance & Verification

The dataset metadata and split files have been hashed and locked:

| Split Asset | Path | Exact SHA-256 Digest |
| :--- | :--- | :--- |
| **NIH Image Metadata CSV** | `Data_Entry_2017_v2020.csv` | `dc1d2df67fdc1c5a7601d48699cda2b13dc2c4841488b4183dcf04884dbaca11` |
| **CheXNet 14-Pathology Labels CSV** | `chexnet/nih_labels.csv` | `80324996867e73546bd7a09025df4a4cc3243fc00663b753023ccd90a9b5f8b9` |
| **Anonymizer Training Pairs (10,000)** | `image_pairs/image_pairs_training_10000.txt` | `3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268` |
| **Anonymizer Validation Pairs (2,000)** | `image_pairs/image_pairs_validation_2000.txt` | `9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7` |

---

## §8 Patient-Level Disjointness Proofs

To eliminate any risk of data leakage or patient identity contamination:
1. **Anonymizer TRAIN vs Classification VAL**:
   - Anonymizer TRAIN: 10,000 pairs (20,000 images), 9,053 unique patients.
   - Classification VAL: 10,816 images, 3,854 unique patients.
   - Intersection: **Exactly 0 patients ($N=0$)**.
2. **Anonymizer TRAIN vs Anonymizer VAL**:
   - Anonymizer VAL: 2,000 pairs (4,000 images), 1,742 unique patients.
   - Intersection: **Exactly 0 patients ($N=0$)**.
3. **Formal Invariant**: Patient sets across training and evaluation splits are disjoint subsets of the NIH ChestX-ray14 cohort.

---

## §9 Identity Label Semantic Audit

All image pairs in `image_pairs_training_10000.txt` and `image_pairs_validation_2000.txt` were audited:
- Format: `<image_1> <image_2> <label>`
- Verification Rule: Label is `1.0` if and only if `patient_id(image_1) == patient_id(image_2)`, and `0.0` otherwise.
- Result: **100% of pairs (10,000/10,000 TRAIN and 2,000/2,000 VAL) have valid, ground-truth matching identity labels**.

---

## §10 Pathology Label Grammar & Fail-Closed Enforcement

The 14 NIH pathology labels are strictly parsed according to the canonical list:
`['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']`
- Any unknown token, malformed label string, or non-finite label matrix in `LazyPairDataset` immediately triggers an uncatchable `ValueError` / `RuntimeError` (Fail-Closed).
- Evaluated and certified in Test `T169`.

---

## §11 Anonymizer Training Semantics & Architecture

The deformation operator is defined by:
$$\Delta = \mathcal{G}_\theta(I_1)$$
$$\mathcal{T} = \mathcal{T}_{\text{identity}} - \mu \cdot \mathcal{K}_\sigma(\Delta)$$
$$I_{\text{anon}} = \text{GridSample}(I_1, \mathcal{T}, \text{align\_corners}=\text{True}, \text{padding}=\text{border})$$

- $\mu = 0.01$ (deformation amplitude multiplier)
- $\mathcal{K}_\sigma$: $9 \times 9$ Gaussian smoothing filter with $\sigma = 2.0$.
- Optimization Objective:
  $$\mathcal{L}_G^{B\_dev} = \mathcal{L}_{AC}(I_{\text{anon}}, y_{\text{path}}) + \mathcal{L}_{priv}(I_{\text{anon}}, I_2)$$
  $$\mathcal{L}_G^{C4} = \mathcal{L}_{AC}(I_{\text{anon}}, y_{\text{path}}) + 1.0 \cdot \text{MSE}(\phi(I_{\text{anon}}), \phi(I_1)) + \mathcal{L}_{priv}(I_{\text{anon}}, I_2)$$

---

## §12 Paired Ordering & Hash Pipeline

- Epoch 0, 1, and 2 mini-batch pairing and ordering are deterministic across both arms.
- Runtime train order telemetry records the SHA256 digest of semantic pair rows (`image1|image2|label\n`) per epoch.
- Runtime loader fingerprints match the offline precomputed ordering byte-for-byte across epochs (`T153`, `T154`, `T155`).

---

## §13 Checkpoint Selection Invariants & Tie-Breaking

- **Selection Metric**: Method-neutral sum $\mathcal{S} = \mathcal{L}_{AC\_BCE} + \mathcal{L}_{priv}$ evaluated on the 2,000 validation pairs.
- **Exclusion Invariant**: The feature preservation MSE loss term is **strictly excluded** from the selection metric for both $B\_dev$ and $C4$ (`T174`).
- **Tie-Breaking Rule**: In case of identical selection loss scores across epochs, the **earliest epoch** is deterministically selected (`T175`).

---

## §14 S1 Adaptive Attacker Protocol

- **Model**: Fresh ResNet-50 Siamese Network trained end-to-end from scratch (ImageNet weights initialized at seed 42).
- **Input Geometry**: Anonymized image $I_{\text{anon}}$ and candidate image $I_2$.
- **Training Protocol**: 100 max epochs with early stopping patience of 5 epochs on validation BCE loss.
- **Seed Policy**: Locked to seed 42 for primary evaluation.

---

## §15 Privacy Evaluation Geometry (anon/real)

- Evaluated strictly using the trained S1 Adaptive Attacker on the 2,000 validation set pairs using `anon(x1), real(x2)` geometry.
- Primary privacy metric: **Attacker Verification Area Under the ROC Curve (ROC-AUC)**.
- Replay validation: Raw predictions saved in `privacy_val_predictions.npz`, SHA hashed and recomputed during validity checks.

---

## §16 Classification Evaluation Protocol (VAL-only, 14 AUCs)

- Evaluated strictly on the 10,816 images of the CheXNet classification VAL split (`chexnet/nih_labels.csv` fold `'val'`).
- Primary utility metric: **Mean 14-Pathology AUROC (Macro AUC)**.
- Strict Hard-Fail Enforcement: All 14 pathology AUCs must be finite. One-class pathologies are rejected because ROC-AUC is undefined. Valid ROC-AUC values may include the endpoints 0.0 and 1.0. The implementation hard-fails on NaN/Inf AUCs and on one-class pathologies where no AUC can be computed (`T83`, `T84`, `T157`).
- Replay validation: Raw predictions saved in `classification_val_predictions.csv` and `classification_val_aucs.csv`, SHAs verified, and 14 AUCs recomputed during validity checks.

---

## §17 Independent Pristine Reference & Parity Proof

An independent, zero-dependency reference implementation (`m0_tests/pristine_reference.py`, commit `29245d1f71571898d9527417df4ae3f63a8695f6`) was created and compared against the production runner:
- **Anonymized Tensor Parity**: Maximum absolute difference $= 0.0 \le 1.0 \times 10^{-6}$ (`T162`).
- **Generator Loss Parity**: Maximum absolute difference $= 0.0 \le 1.0 \times 10^{-6}$ (`T163`).
- **Generator Gradient Parity**: Maximum absolute difference $= 7.45 \times 10^{-9} \le 1.0 \times 10^{-6}$ (`T164`).
- **Verifier Critic Gradient Parity**: Maximum absolute difference $= 0.0 \le 1.0 \times 10^{-6}$ (`T165`).
- **Classifier Critic Gradient Parity**: Maximum absolute difference $= 0.0 \le 1.0 \times 10^{-6}$ (`T166`).

---

## §18 Gradient & Parameter Ownership Proofs

- During the Generator update step, critic networks (`netSNN`, CheXNet) are set to `eval()` mode and their parameters remain frozen.
- During Critic update steps, Generator parameters are detached (`fakes_1.detach()`), ensuring zero generator gradient leakage into critics (`T167`, `T168`).

---

## §19 Execution Harness & Preflight Hardening

- The scientific launcher `run_m2_s1.py` enforces `--scientific-m2-s1` mode for all non-unit-test executions globally.
- Non-empty or contaminated output directories are rejected before execution begins (`T147`).
- Attacker loss tensors are checked for NaN/Inf at every iteration (`T149`, `T150`, `T151`).
- Numerical validity flags (`'numerical_validity': 'PASS'`) are required across all telemetry manifests (`T152`).

---

## §20 Git Source & Lineage Guard Proof

- Scientific execution enforces source cleanliness via `git diff --quiet` (tracked unstaged changes are rejected), `git diff --cached --quiet` (staged changes are rejected), and rejection of untracked importable/runtime Python/config files. Unrelated non-runtime untracked files are not part of the scientific source guard (`T172`).
- Earlier M1.4c wording about permissive untracked-file handling is superseded by the M1.4c.3 exact source-identity guard; see `research_agent/M1_4C3_FINAL_EXECUTION_BOUNDARY_CLOSEOUT.md` for the current authority.
- Canonical HEAD on `origin/research/method-restart` is verified at `c6431310061c04e54dce82d30ae6e0ce24440562`.

---

## §21 Complete Test Suite Matrix (T1–T176+)

Corrected by M1.4c.1 after independent forensic review. This historical document is no longer the authoritative cross-suite execution record. Current suite totals and promotion blockers must be taken from the source-bound M1.4c.3 inventory and closeout records, not from this historical report.

---

## §22 Measured Determinism Envelopes & Reproducibility

- Multi-pass forward-backward error bounds across identical seeds on CUDA:
  - Total Loss Diff: $0.0$
  - Max Parameter Gradient Diff: $4.45 \times 10^{-5}$
  - Verifier / Classifier Gradient Diff: $0.0$

---

## §23 Scientific Resume Policy Certification (Option B)

- **Policy Choice**: Option B (Scientific runs must restart from epoch 0).
- **Rule**: Resuming an interrupted training run from an intermediate checkpoint is marked as **UNCERTIFIED** for scientific claims (`T156`).
- **Enforcement**: Any publication or scientific verdict must be generated from an uninterrupted run starting from epoch 0.

---

## §24 Zero Test Access Firewall Proof

- The `TestFirewall` subsystem intercepts all dataset initialization and loader requests.
- Mode `'test'`, `'testing'`, `'eval_test'`, and test-split paths are intercepted and blocked with a fatal exception across all modules during dev and evaluation phases (`T40`, `T52`, `T112`, `T176`).
- Test split data has had **zero access** throughout M1.4c certification.

---

## §25 Historical Disposition (Superseded)

The earlier FULL PASS wording is withdrawn. It was not sufficient to establish exact
source identity, direct scientific API closure, strict replay-size provenance, or
fail-closed numerical diagnostics. This file must not be used to authorize M2/S1.

## §26 M1.4c.3 Boundary-Closeout Disposition

M1.4c.3 is an implementation and test closeout only. No real M2, S1, S2, or TEST
execution occurred in this pass. The audit branch is not the certified canonical
branch and the external certified tag is intentionally absent here.

**Disposition: BLOCKED — SMALL CERTIFICATION/PROMOTION CLOSEOUT REQUIRED.**

The authoritative next record is `research_agent/M1_4C3_FINAL_EXECUTION_BOUNDARY_CLOSEOUT.md`.
