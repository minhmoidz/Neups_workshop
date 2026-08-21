# M2-S1 Scientific Report — Paired Baseline Control vs C4 (Feature Preservation)

## Executive Summary
This report documents the definitive findings of the paired **M2-S1** experimental run comparing the restored baseline control (`B_dev`, $\mu=0.01$, legacy operator) against the penultimate feature-preservation method (`C4`, feature loss weight=1.0) under identical training semantics and exact data-order pairing.

---

## 1. Key Results Summary

| Metric | B_dev (Control) | C4 (Method) | Delta (C4 - B_dev) | Gate / Target | Gate Status |
|---|---:|---:|---:|---|:---:|
| **Selected Generator Epoch** | `13` | `8` | — | Method-neutral min total loss | diagnostic |
| **Val Selection Total Loss** | `0.22939` | `0.23430` | `+0.00491` | $L_{AC} + L_{priv}$ | diagnostic |
| **Adaptive Re-ID VAL AUC** | `0.8132` | `0.8023` | `-0.0109` | $\Delta_{priv} \le +0.03$ | **PASS** |
| **Classification Macro VAL AUC** | `0.7841` | `0.8056` | `+0.0216` | $\Delta_{class} \ge 0.0$ | **PASS** |
| **Segmentation Dice** | *BLOCKED* | *BLOCKED* | — | Certified evaluator | **NOT APPLICABLE** |
| **Peak VRAM (MB)** | `8963.1` | `9665.7` | `+702.6` | < 16,000 MB | diagnostic |
| **Training Runtime (Hours)** | `31.57` | `27.74` | `-3.83` | 250 epochs | diagnostic |

---

## 2. 14 Pathology Classification AUCs (Validation Fold)

| Pathology | B_dev AUC | C4 AUC | Delta (C4 - B_dev) |
|---|---:|---:|---:|
| Atelectasis | `0.7828` | `0.8056` | `+0.0228` |
| Cardiomegaly | `0.8544` | `0.8638` | `+0.0095` |
| Consolidation | `0.7988` | `0.8072` | `+0.0084` |
| Edema | `0.8853` | `0.8953` | `+0.0100` |
| Effusion | `0.8803` | `0.8874` | `+0.0071` |
| Emphysema | `0.8461` | `0.8879` | `+0.0419` |
| Fibrosis | `0.7059` | `0.7007` | `-0.0052` |
| Hernia | `0.8629` | `0.8748` | `+0.0119` |
| Infiltration | `0.5758` | `0.5895` | `+0.0137` |
| Mass | `0.7722` | `0.8334` | `+0.0611` |
| Nodule | `0.6781` | `0.7309` | `+0.0529` |
| Pleural_Thickening | `0.7903` | `0.8132` | `+0.0228` |
| Pneumonia | `0.7273` | `0.7379` | `+0.0106` |
| Pneumothorax | `0.8164` | `0.8512` | `+0.0348` |

---

## 3. C4 Feature-Loss Gradient Norm Diagnostics

| Epoch | Base Objective Grad Norm | Feature Loss Grad Norm | Ratio (Feature / Base) |
|---|---:|---:|---:|
| 0 | `2.04146e+00` | `8.62091e-03` | `0.0042` |
| 1 | `6.75146e+00` | `3.10322e-01` | `0.0460` |
| 25 | `7.21092e-01` | `4.56321e-01` | `0.6328` |
| 50 | `2.55318e+00` | `3.76705e-01` | `0.1475` |
| 75 | `5.13850e-01` | `8.59332e-01` | `1.6723` |
| 100 | `4.26404e+00` | `1.10721e+00` | `0.2597` |
| 125 | `9.75283e-01` | `9.54717e-01` | `0.9789` |
| 150 | `1.71930e+00` | `9.91710e-01` | `0.5768` |
| 175 | `1.13085e+01` | `2.04366e+00` | `0.1807` |
| 200 | `4.42320e+00` | `2.00972e+00` | `0.4544` |
| 225 | `4.24892e+00` | `1.08565e+00` | `0.2555` |

---

## 4. Scientific Provenance & Artifact Hashes
- **Branch**: `research/method-restart`
- **B_dev Selected Generator SHA256**: `18381d92c64bb3d646b62d5fb9d0ed8c208cf2cb3154f8aa1dac4b1baff610cd`
- **C4 Selected Generator SHA256**: `366a7dd083c6e6547ca359cb351eed178746b90acdd3567bae423291b2329337`
- **B_dev Attacker Checkpoint SHA256**: `ade8cd52cf69a72be3177cd60b00d35050f7674cf5a4013b3787b3f1db59f5e8`
- **C4 Attacker Checkpoint SHA256**: `3da0408f2fb55abbefbf9517d6418b0c673058da81882f89f74fd3cf365c43f8`
- **Test Firewall**: STRICTLY CLOSED (`test_touched: false`)

---

## 5. Frozen S1 Decision Gate Evaluation
- **Run Validity**: `VALID` (VALID)
- **Privacy Gate ($\Delta_{priv} \le +0.03$)**: `PASS` ($\Delta_{priv} = -0.0109$)
- **Classification Gate ($\Delta_{class} \ge 0.0$)**: `PASS` ($\Delta_{class} = +0.0216$)
- **Segmentation**: `NOT APPLICABLE — evaluator provenance not yet certified`

### Final S1 Verdict: **C4 S1: PROMOTE TO S2**
