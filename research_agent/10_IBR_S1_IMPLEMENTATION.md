# STEP 6B — IBR S1 Implementation + Load-Bearing Tests

**Phase:** implementation only — no full training. **NO TRAINING, NO ATTACKER, NO TEST.**
Repository commit descends from `7958d52`. Source of truth: `research_agent/09_IBR_PREIMPLEMENTATION_LOCK.md`.
S1 code implemented exactly per the STEP 6A lock. Full S1 training run is deferred.

---

## 1. What was implemented

New Phase-II files under `research_agent/ibr/` (no invasive edits to the canonical baseline):

| File | Contents |
| ---- | -------- |
| `ibr/grl.py` | `GradientReversalLayer` (±λ backward, identity forward) |
| `ibr/models.py` | `IBREncoder`, `IBRDecoder`, `ZIdVerifier`, `ZMedAdversary` |
| `ibr/frozen_models.py` | `load_frozen_classifier`, `load_frozen_segmentation_teacher` (SHA hard-fail) |
| `ibr/losses.py` | frozen coefficients + individual loss functions + `FrozenUtility` |
| `ibr/s1_loss.py` | `compute_s1_loss` — the single S1 5-term loss assembly |
| `ibr/ibr_model.py` | `IBRModel` bundle (E, G, V, H_med, GRL) |
| `ibr/donor.py` | `DonorSampler` — deterministic, patient-disjoint, no-TEST |
| `ibr/train_s1.py` | future S1 training entry point (TEST rejected, explicit split) |
| `ibr/dry_run_s1.py` | STEP 6B dry run (1 batch fwd+bwd, VRAM measurement) |
| `test_ibr_s1.py` | load-bearing tests TEST 1–8 |

## 2. Architecture (frozen per STEP 6A lock)

| Component | Specification | Params |
| --------- | ------------- | ------ |
| Encoder trunk `E` | U-Net encoder, init_features=32, 256×256×1 → bottleneck 16×16×512 | 3.48 M |
| `z_id` head | `AdaptiveAvgPool2d` → `Linear(512→128)` | — |
| `z_med` | spatial bottleneck map (B,512,16,16) | — |
| Decoder `G` | U-Net decoder with skip connections; `z_med + Linear_up(z_id)` injection; `tanh` output | 4.35 M |
| `V` (z_id verifier) | `MLP(256→128→1)`, BCE | 33.0 k |
| `H_med` (z_med adversary) | 1×1 conv stack → GAP → Linear; normal training, GRL on encoder path | 131.0 k |
| GRL | `GradientReversalLayer(λ=1.0)` | — |
| **S1 trainable total** | E+G+V+H_med | **8.09 M** |

Tensor shapes (B=2 smoke): `z_id (2,128)`, `z_med (2,512,16,16)`, `x_self (2,1,256,256)`, `x_anon (2,1,256,256)` — all finite.

## 3. Frozen checkpoints + SHA (hard-verified)

| Model | Path | SHA-256 | Architecture | Params | Verified |
| ----- | ---- | ------- | ------------ | ------ | -------- |
| Pathology classifier | `networks/pretrained_classifier.pth` | `8ad15b38…5e663` | DenseNet-121 14-label sigmoid | 6,968,206 | TEST 7 + loader assert |
| Segmentation teacher | `archive/train_seg_unet/best.pth` | `2dfdcf9b…73e0` | `UNetSeg(1,3,16)` (checkpoint `init_features=16`) | 1,942,323 | TEST 7 + prefix assert |

> **Param-count reconciliation with the STEP 6A lock:** the lock (§1) reports 1,945,285 for the teacher. That number is the full `state_dict` total including the 2,962 `num_batches_tracked` buffer elements (18 BatchNorm layers, one scalar each). `module.parameters()` yields 1,942,323; 1,945,285 − 1,942,323 = 2,962. Same checkpoint, different counting convention (buffers vs parameters).

Both loaded with `eval()`, `requires_grad_(False)`. Gradients propagate THROUGH their forwards to `x_anon` (no `torch.no_grad` on those paths — TEST 3/8).

## 4. Loss graph (exactly the STEP 6A lock)

```
L_S1 = λ_rec·L_rec + λ_path·L_path + λ_anat·L_anat + λ_zid·L_zid_pair + λ_adv·L_zmed_adv
```

Frozen coefficients (not tuned): `λ_rec=λ_path=λ_anat=λ_zid=λ_adv=1.0`, `GRL λ=1.0`.

| Term | Definition | Target |
| ---- | ---------- | ------ |
| `L_rec` | `‖x_self − x‖₁` | **ONLY x_self** — verified no x_anon→L_rec path (TEST 2) |
| `L_path` | BCE( classifier(x_anon), y_path_source ) | anon keeps source pathology |
| `L_anat` | MSE( seg(x_anon), seg(x_source) ) | anon keeps source anatomy |
| `L_zid_pair` | BCE( V(z_id, z_id_donor), y_pair ) | identity organization on z_id |
| `L_zmed_adv` | BCE( H_med(GRL(z_med), GRL(z_med_donor)), y_pair ) | adversarial pairwise identity suppression in z_med (NOT called mutual information) |

## 5. Donor protocol (implemented + tested)

- `donor != source` patient: hard assertion inside sampler (assert on patient IDs).
- TRAIN source → TRAIN donor pool; VALIDATION source → VALIDATION donor pool only.
- TEST: no donor pool; TEST source images are REJECTED (RuntimeError) in development.
- Deterministic for fixed seed; different seed → different mapping (TEST 6).
- Donor/source patient IDs persisted via `DonorSampler.provenance()`.

## 6. Tests

| Test | What it verifies | Result |
| ---- | ---------------- | ------ |
| TEST 1 | shapes (z_id 128, z_med 256→512×16×16, x_self/x_anon == input) + finite | PASS |
| TEST 2 | reconstruction isolation: L_rec unchanged when donor changes; grads reach E/G self | PASS |
| TEST 3 | frozen classifier/teacher get NO grads; L_path/L_anat gradients reach x_anon | PASS |
| TEST 4 | GRL gradient sign flipped vs normal adversary (numeric; normal −0.269, grl +0.269) | PASS |
| TEST 5 | z_id verifier: loss fit decreases (0.696→0.090); p_same 0.912 > p_diff 0.077 | PASS |
| TEST 6 | donor safety: donor≠source, deterministic per seed, seed-changes mapping, no TEST touched, TEST source rejected | PASS |
| TEST 7 | checkpoint provenance: paths, SHA, architecture, param counts | PASS |
| TEST 8 | gradient ownership: one synthetic fwd+bwd; all trainable finite; frozen grad-free; no detached utility grad | PASS |

Result: **`IBR S1 LOAD-BEARING TESTS: ALL PASS`** (8/8).

## 7. Dry run — actual peak VRAM

bs=16, RTX 5070 Ti 16 GB, torch 2.7.0+cu128:

| Metric | Value |
| ------ | ----- |
| S1 trainable params | 8.09 M |
| Dataset sanity | CXRDataset(val) sample OK (1×256×256, 14 labels) |
| Loss parts (1 batch) | L_rec 0.708, L_path 2.662, L_anat 0.106, L_zid 0.711, L_adv 0.672, total 4.859 |
| **Actual peak VRAM (bs=16)** | **9.68 GB** |
| Free during run | 5.58 GB |
| Post-step params finite | True |

Vs STEP 6A estimate (6.78 GB): the dry run is higher because the real S1 graph retains the frozen DenseNet-121 and UNetSeg forward activations in-graph (gradients must flow to x_anon; the STEP 6A probe ran them under `no_grad`, undercounting). **bs=16 does not OOM and retains >5 GB safe headroom → keep bs=16. No gradient accumulation introduced.**

## 8. Training entry point

`ibr/train_s1.py --split {train|validation} [--seed] [--bs] [--max_epochs] [--device]` — TEST is rejected at argparse level (`invalid choice`); cannot default to TEST. Logs per-component loss, total, gradient finite flags, donor diagnostics, verifier/adversary performance, classification/segmentation diagnostics, config, seed, checkpoint hashes, git commit. **Not executed** in this step.

## 9. Files added/modified

- Added: `research_agent/ibr/` (10 files), `research_agent/test_ibr_s1.py`, `research_agent/10_IBR_S1_IMPLEMENTATION.md`, `research_agent/10_IBR_S1_summary.json`.
- Modified: none (canonical baseline untouched).

## 10. Known limitations

- `L_anat` uses MSE on the teacher's 3-structure probability maps (equivalent scale to BCE terms); STEP 6B did not run hyperparameter search — coefficients stay frozen.
- `L_zid_pair` in S1 uses the (source, partner) pair; the partner is a same-patient image with p=0.5 (when the patient has ≥2 images in the split) else a different-patient donor (y=0), so BOTH identity classes reach V and H_med. Fix documented in `11A_IBR_S1_MECHANISM_DEBUG.md`.
- Full S1 training, attacker training, and TEST remain strictly out of scope for this step.

## 11. Strict step boundaries respected

No S1 epochs, no attacker training, no TEST inspection, no λ tuning, no z_id dimension change, no V2, no ESSA implemented.