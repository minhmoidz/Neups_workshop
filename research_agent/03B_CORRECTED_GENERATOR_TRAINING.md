# 03B — CORRECTED CANONICAL GENERATOR TRAINING (STEP 3A1)

> Status: **DONE** (2026-08-11). One corrected PriCheXy-Net baseline generator trained,
> validated, frozen with SHA-256. **No attacker run** (deferred to STEP 3A2).

---

## 1. Objective

Train the **canonical corrected-baseline PriCheXy-Net generator** — a baseline
reproduction of the accepted historical `baseline_fixed` recipe that differs ONLY by
replacing the legacy operator `G*(I-u)` with the corrected operator `I-G*u`
(`transform_mode='corrected'`). No method change, no hyperparameter tuning, no C2/C4, no
test-set access.

The 03A audit (committed `7d01e46`) established that no previously existing checkpoint
was trained under `corrected`, hence this reproduction step.

---

## 2. Git / environment

| Item | Value |
|---|---|
| Branch | `main` |
| HEAD at launch | `7d01e46334d456264e85f35f6c6b45f82d4e7195` (`7d01e46`) |
| Working tree at launch | clean (03A report committed) |
| GPU | NVIDIA GeForce RTX 5070 Ti (15.5 GB total) |
| Python | repo `.venv` (`python3.10`) |
| Preflight tests | `test_grad_accum.py` PASS, `test_operator_repair.py` PASS |

---

## 3. Historical baseline recipe audit

The accepted historical generator is
`archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth` produced by
`config_files/config_anonymization_baseline_fixed.json` (`train_prichexy_net_baseline_fixed`).
The recipe was reconstructed from code (`agents/Agent.py`, `utils/utils.py`,
`networks/UNet_PriCheXyNet.py`, `utils/GaussianSmoothing.py`, `utils/ACLoss.py`,
`utils/VerificationLoss.py`) at the historical commit `6c86e8f` and cross-checked against
the current tree (diff shows only STEP 1/1B operator/provenance threading).

| Field | Historical `baseline_fixed` | Corrected baseline (this run) |
|---|---|---|
| architecture | UNet(1, 2, 32), flow_field, tanh head | **same** |
| initialization | load `./networks/pretrained_generator_prichexy_net.pth` | **same** |
| optimizer | generator: Adam; ver models: Adam; AC: SGD(momentum=0.9, wd=1e-4) | **same** |
| learning rate | 0.0001 | **same** |
| scheduler | none | **same** |
| batch size | 16 | **same** |
| number of epochs | 60 | **same** |
| early stopping | none (epoch cap) | **same** |
| gradient accumulation | accumulation_steps=1; H2 fix (zero_grad after step) | **same** |
| gradient accumulation FIX (H2) | active | **same** (verified by test_grad_accum) |
| μ | 0.01 | **same** |
| stochastic_lambda | 0.0 | **same** |
| train data | `image_pairs_training_10000.txt` via `Dataset(phase='training')` | **same** |
| validation data | `image_pairs_validation_2000.txt` via `Dataset(phase='validation')` | **same** |
| loss terms | AC BCE + verification `-log(1-ver)`; total = 1·ac + 1·llh | **same** |
| loss weights | ac_loss_weight=1, ver_loss_weight=1 | **same** |
| adversarial/critic settings | ver_ensemble_size=1, ver_active_per_step=1, ver_restart_every=25 (default), warmup 200 | **same** |
| checkpoint selection rule | min **validation total loss** (`Agent.py:221-224`) | **same** |
| RNG / seed handling | `seed_all(42)` (python.random, numpy, torch CPU+CUDA, `cudnn.benchmark=False`) | **same** |
| image size | 256 | **same** |
| augmentations | none (Resize+ToTensor only) | **same** |
| Gaussian kernel / sigma | kernel_size=9, sigma=2, channels=2 | **same** |
| coordinate convention | identity grid from linspace(-1,1), meshgrid ij, stack (y,x) | **same** |
| align_corners | True | **same** |
| padding_mode | border | **same** |
| transform_mode | legacy | **corrected** (only intended behavioral diff) |

**No other behavioral difference found.** Diff of `agents/Agent.py` and `utils/utils.py`
vs historical commit shows only: `transform_mode` threading, provenance recording, and the
operator refactor (`build_sampling_grid`/`compute_budget_map`) proven bit-for-bit
legacy-equivalent by `test_operator_repair.py` (TEST 12 legacy equivalence).

---

## 4. Corrected configuration

`config_files/config_anonymization_baseline_corrected.json`:

```json
{
  "experiment_description": "train_prichexy_net_baseline_corrected",
  "image_path": "/home/minhtt/datasets/nih/images/",
  "seed": 42,
  "ac_loss_weight": 1,
  "ver_loss_weight": 1,
  "generator_type": "flow_field",
  "mu": 0.01,
  "transform_mode": "corrected",
  "ver_ensemble_size": 1,
  "ver_active_per_step": 1,
  "accumulation_steps": 1,
  "use_budget_map": false,
  "stochastic_lambda": 0.0,
  "image_size": 256,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "max_epochs": 60,
  "show_every_n_epochs": 10,
  "show_every_n_iterations": 100
}
```

Config SHA-256: `f5b83f4ac23326887b5c9ac85f9ce5aea35ff6f7c74e9a55b8dca778b19616da`.

---

## 5. Exact differences from `baseline_fixed`

| Field | Change |
|---|---|
| `experiment_description` | `train_prichexy_net_baseline_fixed` → `train_prichexy_net_baseline_corrected` |
| `transform_mode` | absent (→ legacy default) → `"corrected"` |
| everything else | **identical** |

The new config was **created, not overwritten**; the historical config and checkpoint are
untouched.

---

## 6. Test-set firewall

- `Agent.py` builds only `training`/`validation` loaders (`phase='training'`,
  `phase='validation'`); no `phase='testing'` path is reachable during training/validation.
- `image_pairs_testing_5000.txt` was **never opened**; its mtime remains
  `Aug  8 22:01` (unchanged). Its hash was not computed (hash withheld by protocol §3.1).
- Training log contains no `testing`/`test_`/`test_auc`/`test_metrics` references.
- Output archive contains no test-metric artifacts (`find ... -iname '*test*'` returns only
  `*_latest.pth` / `*_lowest_total_loss.pth` model weights, which are checkpoints, not test metrics).
- No `adaptive re-ID`, `Top-k`, test classifier, or test segmentation evaluation was run.

---

## 7. Training execution

```
python train_architecture.py --config config_anonymization_baseline_corrected.json
```

- Launched 2026-08-11 11:53 (+07), completed 14:53 (epoch 60, normal termination).
- Console header confirmed: `[ Using Seed :  42  ]`, `[transform_mode] resolved mode for this run: corrected`.
- Per-epoch duration ≈ 2.95–3.0 min; 60 epochs.
- Log: `logs/t3a1_baseline_corrected.log` (gitignored).

---

## 8. Training/validation trajectory

`loss_dict.pkl` (validation total loss, monitored quantity):

| Epoch (0-idx) | train total | val total |
|---|---|---|
| 0 | 1.1747 | 0.7630 |
| 10 | 0.2704 | 0.3430 |
| 20 | 0.2467 | 0.2591 |
| **24 (best)** | — | **0.2445** |
| 30 | 0.4923 | 0.4708 |
| 40 | 0.3556 | 0.3994 |
| 50 | 0.7137 | 0.7319 |

The post-best rise is the expected adversary-restart cycle (`ver_restart_every=25`),
identical to the historical recipe. All 60 epochs finite (no NaN/Inf).

---

## 9. Checkpoint selection

- Monitored quantity: **validation total loss**
- Direction: **min**
- Best epoch: **24** (0-indexed; = epoch 25 in 1-indexed log convention) — confirmed both by
  `loss_dict.pkl` and by the training log (`Current generator with lowest total_loss: epoch 24`)
- Selected checkpoint (canonical rule, `Agent.py:221-224`): `generator_lowest_total_loss.pth`
- **No test criterion used.**

---

## 10. Generator provenance

Written to two locations (identical):
- `archive/train_prichexy_net_baseline_corrected/generator_provenance.json` (run archive, gitignored)
- `research_agent/03B_generator_provenance_baseline_corrected.json` (tracked copy)

Contains: experiment_id, git_commit, config_path + sha256, `transform_mode=corrected`,
mu, stochastic_lambda, architecture, initialization, optimizer, LR, scheduler, batch size,
epochs, grad-accum settings (incl. H2 fix flag), train/validation split paths + hashes,
protocol-document hashes (01, 01B, 03A), seed info, checkpoint criterion + best epoch,
checkpoint path + sha256, run start/end timestamps, test-firewall statement. **No TEST metrics.**

---

## 11. Checkpoint SHA-256

```
8a489eec036679f6775a2bf0b21fb112f1f968490a513b1f9b6dc012d4e384c2
```

Frozen (read-only `chmod 444`) at the stable immutable path:

```
networks/corrected_baseline/generator_lowest_total_loss_corrected.pth
```

This path + hash is the D-1 content-pinning anchor that STEP 3A2 attacker runs must bind to.

---

## 12. Operator sanity checks

Run with `check_corrected_generator.py` (train/validation-side only):

| Check | Result |
|---|---|
| transform_mode resolves | `corrected` ✓ |
| μ | 0.01 ✓ |
| stochastic_lambda | 0.0 ✓ |
| checkpoint exists / loadable | ✓ |
| weights changed vs init | 118/118 layers changed ✓ |
| train+val loss curves finite | ✓ |
| **μ=0 corrected grid == identity** | **True** ✓ |
| **corrected == explicit `I - G·u`** (max diff) | **0.0** ✓ |
| corrected vs legacy grid diff (border) | 0.6516 (they differ, as intended) ✓ |

**PASS.** The trained operator is `I - G*u`, not `G*(I-u)`, and the μ=0 identity invariant holds.

---

## 13. GPU / runtime

| Item | Value |
|---|---|
| GPU model | NVIDIA GeForce RTX 5070 Ti |
| Max VRAM observed | ~9.8 GB (9826 MiB), GPU util 99–100% |
| Runtime | 11:53 → 14:53 (+07), ≈ 3 h |
| Termination reason | epoch cap 60, `Finished Training!` (normal) |

---

## 14. Problems / deviations

| # | Issue | Resolution |
|---|---|---|
| 1 | None during training | — |
| 2 | Initial operator-invariant check in `check_corrected_generator.py` fed un-scaled flow (`0.7`) to `build_sampling_grid` instead of `μ·flow` | Fixed to pass `0.0·flow` for μ=0 and `0.01·flow` for μ=0.01; explicit `I−G·u` equality now asserted (max diff 0.0) |
| 3 | `generator_provenance.json` written into `archive/` (gitignored) | Tracked copy added at `research_agent/03B_generator_provenance_baseline_corrected.json` (byte-identical) |

No deviations from the corrected baseline recipe. No C2/C4, no loss-weight change, no μ sweep,
no critic-setting change.

---

## 15. STEP 3A1 verdict

All PASS criteria met:

1. ✅ corrected generator trained with `transform_mode=corrected` (resolved + provenance-verified)
2. ✅ historical `baseline_fixed` recipe otherwise preserved (only operator + provenance/logging fields differ)
3. ✅ H2 gradient-accumulation fix active (`test_grad_accum.py` PASS; provenance flag)
4. ✅ no TEST access (no test loader, test file mtime unchanged, no test metrics in log/archive)
5. ✅ checkpoint chosen by canonical train/validation rule (min validation total loss, epoch 24)
6. ✅ checkpoint loadable (verified)
7. ✅ generator provenance complete (`generator_provenance.json`)
8. ✅ exact checkpoint SHA-256 frozen: `8a489eec036679f6775a2bf0b21fb112f1f968490a513b1f9b6dc012d4e384c2`
9. ✅ no C2/C4 or other method modification introduced

**No attacker was launched** (Task 13 stop honored). The next step is STEP 3A2 (real
attacker TRAIN/VALIDATION smoke) against the frozen corrected generator, bound by path +
SHA-256.

---

## Commit report

| Item | Value |
|---|---|
| Branch | `main` |
| Full commit hash | `6ad8ec7d0ef8ed34be9cb3df37fcdfa1b8f5c5d5` (pushed `65c81ac..6ad8ec7`) |
| Short hash | `6ad8ec7` |
| Generator checkpoint path | `networks/corrected_baseline/generator_lowest_total_loss_corrected.pth` (gitignored, immutable) |
| Generator checkpoint SHA-256 | `8a489eec036679f6775a2bf0b21fb112f1f968490a513b1f9b6dc012d4e384c2` |

**STEP 3A1 CORRECTED GENERATOR: PASS**
