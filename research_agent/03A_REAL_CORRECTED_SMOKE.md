# 03A — REAL Corrected-Baseline Smoke (Attacker training)

> Status: **BLOCKED — CORRECTED GENERATOR REQUIRED** (2026-08-11).
> No attacker training performed. The repository contains **only legacy-trained
> generator checkpoints** (or generators with no operator provenance); no checkpoint
> trained under `transform_mode='corrected'` exists anywhere in the repo.

---

## 1. Hard claims and verdict summary

| # | Claim | Status |
|---|-------|--------|
| 1 | No `transform_mode='corrected'` generator checkpoint exists in the repository | **CONFIRMED** (audit §3) |
| 2 | Every generator `.pth` in the repo predates the corrected operator's existence in code | **CONFIRMED** (§3.1) |
| 3 | No config file pins `transform_mode`; `resolve_transform_mode` therefore resolves every generator to `legacy` | **CONFIRMED** (§3.3) |
| 4 | No operator-provenance artifact (`transform_mode.txt` / `resolved_config.json`) exists for any generator | **CONFIRMED** (§3.4) |
| 5 | STEP 1C smoke was a **proxy** re-ID on a real validation batch — NOT a corrected-trained generator; it is non-evidence for the real baseline | **CONFIRMED** (§3.5) |
| 6 | Attacker training under the corrected operator must be gated on a corrected-trained generator checkpoint | **HOLDING** — the required artifact does not exist |
| 7 | **Final verdict** | **STEP 3A REAL SMOKE: BLOCKED — CORRECTED GENERATOR REQUIRED** |

**What was NOT done (and why):** the attacker was **not trained**. Training a
corrected-baseline attacker against a legacy (or operator-unprovenanced) generator would
re-measure the *legacy* anonymizer, i.e. the exact artifact class STEP 1 removed; any
such result would be scientifically ambiguous and could not back the corrected baseline
(protocol §1.4: proxy/legacy similarity is never evidence). Strict stop applied.

---

## 2. Preconditions — read-only SHA-256 checks

Frozen artifacts verified (unchanged, content hashed; nothing opened or modified):

| Artifact | SHA-256 |
|---|---|
| `research_agent/01_ADAPTIVE_REID_PROTOCOL.md` | `c3aa381ea2136d89d03f6a409a36edd2e1cb5d8f4986b3ae3846028f61cd6741` |
| `research_agent/01B_PROTOCOL_AMENDMENT.md` | `05fcc9fd4cb4181acd9f40603707cd259987a0d1d3750d0f3463fc1fe78b0865` |
| `image_pairs/image_pairs_training_10000.txt` | `3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268` |
| `image_pairs/image_pairs_validation_2000.txt` | `9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7` |

`image_pairs/image_pairs_testing_5000.txt` exists (`190000` bytes) and was **not opened**
(hash withheld to avoid contaminating the test set; protocol §3.1).

GPU available for the (deferred) real run: `cuda True` — NVIDIA GeForce RTX 5070 Ti.

---

## 3. Corrected-generator audit — why every existing checkpoint is legacy

### 3.1 Timeline: all generator checkpoints predate the corrected operator

| Artifact | Created (mtime) | Corrected operator existed? |
|---|---|---|
| `archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth` | 2026-08-08 22:21:59 | **NO** |
| `networks/pretrained_generator_prichexy_net.pth` (+ `generator_lowest_total_loss_mu_0.01.pth`, `_mu_0.001/0.005`) | 2026-08-08 22:21:59 | **NO** |
| `archive/train_prichexy_net_c4_featureloss/generator_lowest_total_loss.pth` | 2026-08-09 09:54:18 | **NO** |
| `archive/train_prichexy_net_c2c4/generator_lowest_total_loss.pth` | 2026-08-09 13:32:23 | **NO** |
| `archive/train_prichexy_net_c4_mu0.02/generator_lowest_total_loss.pth` | 2026-08-09 16:07:29 | **NO** |
| `archive/train_prichexy_net_c4_mu0.04/generator_lowest_total_loss.pth` | 2026-08-09 21:36:14 | **NO** |

The corrected operator (`I - G*u`) was introduced in git commit
`83738bb` **2026-08-11 05:41:04 +0700** ("STEP 1: correct PriCheXy-Net transformation
operator"). The generator training path (`agents/Agent.py`) only received
`transform_mode` support in commit `18f6eb9` **2026-08-11 05:45:59 +0700**
(`self.transform_mode = utils.resolve_transform_mode(...)`). Every generator above was
produced **before** either commit → it was trained with the historical **legacy**
operator `G*(I-u)`.

### 3.2 Checkpoint-content check: no embedded operator metadata

Loaded each generator state dict and inspected top-level keys:

- `archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth`
- `archive/train_prichexy_net_c2c4/generator_lowest_total_loss.pth`
- `networks/pretrained_generator_prichexy_net.pth`

All three are plain `OrderedDict` of weights (`encoder1.enc1conv1.weight`, ...) with
**no** `config`/`transform_mode`/`meta`/`provenance` keys. There is no way to recover the
training operator from the bytes; provenance must come from the training run, and none
exists (§3.4).

### 3.3 No config pins `transform_mode` → resolves to `legacy` everywhere

`utils.resolve_transform_mode(config.get('transform_mode', 'legacy'))` defaults to
`legacy`. Grep of all `config_files/*.json` for the key `transform_mode`:

```
(none)
```

Relevant generator-training configs, all legacy-by-default:
- `config_files/config_anonymization.json` — `generator_type=flow_field`, `mu=0.01`, seed 42
- `config_files/config_anonymization_baseline_fixed.json` — `mu=0.01`, `use_budget_map=false`, `stochastic_lambda=0.0`, max 60 epochs
- `config_files/config_anonymization_c2c4.json` — the c2c4 run

None of these records the operator that produced the archived checkpoints.

### 3.4 No operator-provenance artifact exists for any generator

`find . -name "transform_mode.txt" -o -name "resolved_config.json"` across the whole
repo (excluding `.venv`): **zero results**. No run of the post-remediation generator
training path has ever recorded its resolved operator, because no such run exists.

### 3.5 STEP 1C smoke was proxy, not a corrected-trained generator

STEP 1C (`00F2_OPERATOR_REVIEW_EVIDENCE.md` §8.1) measured a **proxy re-ID AUC** on a
real validation batch, applying `transform_mode='corrected'` **at deform time** to the
*same legacy checkpoint* (legacy AUC 0.6995 vs corrected 0.7208, delta −0.0213). This
measures the operator difference for a fixed legacy generator; it is **not** a generator
trained end-to-end under the corrected operator, and protocol §1.4 declares proxy numbers
non-evidence for the privacy claim.

### 3.6 Generator hashes (content pinning, D-1)

| Checkpoint | SHA-256 |
|---|---|
| `archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth` | `01287c0c0b2475a10659ba25a012cd9a12da8efddd4a71d87ff4084b8a9f2d4e` |
| `archive/train_prichexy_net_c2c4/generator_lowest_total_loss.pth` | `364a9b573d72c3f273442e516b42f76a09fa7a780e1db97be9d5e34e21ca9333` |
| `networks/pretrained_generator_prichexy_net.pth` | `101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb` |

Any future corrected baseline must carry a **different** hash (D-1 content pinning in
`run_adaptive_reid_arm.py::run_signature` and `verify_stage_e_generator_hash` already
bind the attacker to the exact generator bytes).

---

## 4. READ-ONLY provenance — what exists vs what the corrected baseline needs

| Item | Exists today | Required for the corrected baseline |
|---|---|---|
| Corrected-trained generator checkpoint (`transform_mode='corrected'`) | **NO** | YES — see §5 |
| Generator provenance file (`transform_mode.txt` / `resolved_config.json`) | **NO** | YES (resolved `corrected`, generator hash, mu) |
| Attacker arm runner + D-1 content pinning | YES | reused as-is |
| Frozen pair files + protocol docs | YES | unchanged |
| `config_retrainSNN.json` template | YES | must point at the corrected checkpoint path |

---

## 5. What must be produced before STEP 3A can run

A corrected-baseline generator checkpoint, trained end-to-end with the corrected operator
and provable as such:

1. **Train a generator with `transform_mode='corrected'`.** The post-remediation
   training path already supports it (`agents/Agent.py` reads
   `config.get('transform_mode', 'legacy')` and
   `utils.record_transform_mode_provenance(...)` writes
   `transform_mode.txt` + `resolved_config.json`). Produce e.g.:
   - `config_files/config_anonymization_baseline_fixed_corrected.json` —
     clone of `config_anonymization_baseline_fixed.json` plus
     `"transform_mode": "corrected"`,
     `"experiment_description": "train_prichexy_net_baseline_fixed_corrected"`,
     `mu=0.01`, `use_budget_map=false`, `stochastic_lambda=0.0` (same budget semantics as
     the baseline-fixed generator).
   - Output: `archive/train_prichexy_net_baseline_fixed_corrected/generator_lowest_total_loss.pth`
     plus its provenance files (`transform_mode.txt`, `resolved_config.json`) recording
     the resolved `corrected` operator and generator SHA-256.
2. **Verify** the training-side provenance matches the resolved operator and that
   `resolve_transform_mode` on that config returns `corrected`.
3. **Point the attacker arm at it**: `config_files/config_retrainSNN*.json`
   `perturbation_model_file` → the corrected checkpoint path; runner
   `--checkpoint` → same path. The D-1 hash check then guarantees the attacker sees
   exactly the corrected generator bytes.
4. Only then re-run this STEP 3A smoke (attacker training + per-seed
   validity/near-chance checks) and, after it passes, the confirmatory arm.

---

## 6. Failures discovered and fixes

| # | Failure | Status | Fix |
|---|---------|--------|-----|
| F-1 | No corrected-trained generator checkpoint exists anywhere in the repo | **CONFIRMED** | §5 — train `transform_mode='corrected'` baseline generator with provenance before STEP 3A |
| F-2 | No operator provenance artifact for any existing generator | **CONFIRMED** | §5.1 — provenance auto-written by `record_transform_mode_provenance` during corrected training |
| F-3 | `config_retrainSNN.json` template still points at a nonexistent run-1 path (`./archive/train_prichexy_net_run_1/...`) | **CONFIRMED** | §5.3 — point at the corrected checkpoint once it exists |

No code defect was found in the post-remediation infrastructure (D-1 pinning, run-health
classifier, provenance recorder all present and tested in STEP 2B); the blocker is the
missing scientific artifact, not the tooling.

---

## 7. Provenance requirements checklist (deferred until corrected checkpoint exists)

- [ ] Corrected-trained generator checkpoint + `transform_mode.txt` = `corrected` + `resolved_config.json`
- [ ] Generator SHA-256 recorded in every arm provenance record (D-1)
- [ ] `mu`, `stochastic_lambda`, pair-file hashes, protocol-doc hashes in arm record
- [ ] Attacker determinism check (anonymized test set generated twice, bit-identical)
- [ ] Per-seed validity + near-chance state from training/validation stats only (no test AUC in selection)
- [ ] Full 10-attacker (or budgeted) confirmatory arm run after smoke passes

---

## 8. Final verdict

**STEP 3A REAL SMOKE: BLOCKED — CORRECTED GENERATOR REQUIRED**

The repository contains **only legacy-trained generator checkpoints** (all created
2026-08-08 → 2026-08-10, before the corrected operator existed in code, with no
operator provenance and no config pinning `transform_mode`, hence resolved `legacy`).
Per the protocol, no attacker was trained and no corrected-baseline number was invented.
STEP 3A cannot proceed until a `transform_mode='corrected'` generator checkpoint with
full provenance exists (§5). Tooling is ready; the scientific artifact is missing.
