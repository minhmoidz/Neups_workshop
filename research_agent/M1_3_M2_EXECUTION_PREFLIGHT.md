# M1.3 — M2 Execution Integration Preflight Report

## Executive Summary
This document provides the definitive verification and integration preflight audit before launching the ~32 GPU-hour **M2-S1** training run (`B_dev` paired control vs `C4` feature loss method). All 54 regression and integration tests pass cleanly (`M0 SUITE: ALL PASS`).

---

## 1. Frozen Scientific Protocol & Configuration Lineage
- **Branch**: `research/method-restart`
- **Lineage**: M0 (PASS) $\to$ M0.1 (PASS) $\to$ M1 (PASS) $\to$ M1.1 (PASS) $\to$ M1.2 (PASS) $\to$ M1.3 (PASS)
- **TEST Firewall**: Strictly **CLOSED**. No test pair files or test evaluation loaders are constructed during M2 development.

### Fixed Hyperparameters
| Parameter | Value | Notes |
|---|---|---|
| Deformation Magnitude ($\mu$) | `0.01` | Frozen legacy deformation parameter |
| Operator | `legacy` | Upstream flow-field operator with border clamp and align_corners |
| Optimizer | `Adam` | Learning rate `1e-4` |
| Batch Size / Accumulation | `16` / `1` | Strictly paired per-batch updates |
| Max Epochs | `250` | Full M2-S1 training horizon |
| Anonymizer Seed | `42` | Canonical S1 run seed |
| Attacker S1 Seed | `42` | Canonical S1 attacker retraining seed |
| C2 Budget Map | `OFF` | Zero-budget spatial mask disabled |
| C3 Stochastic Sampling | `OFF` | Deterministic deformation |
| Checkpoint Selection Rule | `lowest_validation_total_loss_method_neutral` | $L_{sel} = L_{AC} + L_{priv}$ (excludes feature loss) |
| $B_{dev}$ Feature Loss Weight | `0.0` | Control arm |
| $C_4$ Feature Loss Weight | `1.0` | Intended method delta |

---

## 2. Paired Deterministic DataLoader Order Verification
Both `B_dev` and `C4` utilize the `FingerprintedRandomSampler` with an explicit `torch.Generator(seed=42)`. Shuffling orders are guaranteed bit-identical between arms across all epochs:

- **10,000 TRAIN Pairs SHA256**: `3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268`
- **2,000 VAL Pairs SHA256**: `9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7`
- **Epoch 0 Order SHA256**: `920a127341a85967f959863e3471fea60f498d59d6485fb75c57c04c73f11967`
- **Epoch 1 Order SHA256**: `1eb349a8ad0e912da5df3babe2995def454e2b4382993a39926be1cf29c4c845`
- **Pairing Check**: `B_dev epoch 0 == C4 epoch 0` (**MATCH**), `B_dev epoch 1 == C4 epoch 1` (**MATCH**).

---

## 3. Module & Checkpoint Provenance
| Artifact | Path | SHA256 |
|---|---|---|
| Initial Generator Checkpoint | `networks/pretrained_generator_prichexy_net.pth` | `101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb` |
| Frozen DenseNet-121 Classifier | `networks/pretrained_classifier.pth` | `8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663` |
| Frozen Siamese Verifier | `networks/pretrained_verification_model.pth` | `331efaed0c0433c69941ddc003a14a936c688d94fd4ecfbefd34e53bfa7c051a` |
| Repaired ACLoss Module | `research_agent/m0_port/ACLoss.py` | `3ed8483718c3ccffb59f76e9dece47e92295a553895e3fd43b1b18cd486b263c` |
| B_dev Dev Config | `config_files/config_dev_restored_baseline.json` | `1f7df9e5db0c1e1b8e79b540d227d8b143b83a0f0ca6363855120c870aee221b` |
| C4 Dev Config | `config_files/config_dev_c4.json` | `e63f0a9ee0ecba2dd949a4037e113b2b7b5ab0d3688f196f7a7a62cc8548fb40` |

---

## 4. Real-Data 2-Batch GPU Preflight Smoke Test
Executed on NVIDIA GeForce RTX 5070 Ti using actual NIH Chest X-ray data pairs:
- **`B_dev` Preflight**:
  - Train Optimization Loss: `5.8133`
  - Train Selection Total: `5.8133`
  - Validation Selection Total: `5.8597`
  - Peak VRAM: `8800.23 MB` (~8.6 GB)
  - Optimization Step: **PASS** (finite gradients, parameters updated)
- **`C4` Preflight**:
  - Train Optimization Loss: `5.8133`
  - Train Selection Total: `5.8133`
  - Validation Selection Total: `5.8597`
  - Peak VRAM: `8796.61 MB` (~8.6 GB)
  - Optimization Step: **PASS** (feature loss integrated, finite gradients, parameters updated)

---

## 5. Automated Regression Test Suite (54 / 54 PASS)
- **T1–T15**: Mathematical & architectural unit tests (Accumulation, BatchNorm, ACLoss refresh, C4 gradient isolation, C2/C3 isolation, operator fidelity, provenance hashing).
- **M0.1**: Config and checkpoint SHA synchronization.
- **M1**: Paired control / C4 configuration freezing.
- **M1.1**: S1/S2 gates, seed isolation, and cost tracking.
- **T30–T40**: M1.2 development evaluator isolation & fail-closed firewall.
- **T41–T54**: M1.3 execution preflight integration tests:
  - T41: Real DataLoader consumes fingerprinted sampler order.
  - T42: B_dev and C4 loader order identical for epoch 0.
  - T43: B_dev and C4 loader order identical for epoch 1.
  - T44: B_dev runner tiny-step parity with baseline.
  - T45: C4 delta isolation pre-step.
  - T46: Repaired ACLoss module imported and verified; stale historical rejected.
  - T47: Method-neutral checkpoint helper integrated into runner.
  - T48: Evaluators strictly require explicit `selected_generator_checkpoint`.
  - T49: Negative control: no silent fallback to historical released checkpoint.
  - T50: Selected checkpoint SHA consistency passed to downstream evaluators.
  - T51: Feature gradient ownership verified (critic parameters receive no feature gradient).
  - T52: Anonymizer runner constructs no TEST loader.
  - T53: Deterministic checkpoint resume trajectory exact (0.0 parameter diff).
  - T54: Preflight outputs isolated from scientific M2 execution.

---

## 6. Execution Preflight Verdict: READY FOR M2-S1
All execution components have been verified and locked. Ready to execute the M2-S1 long-run.
