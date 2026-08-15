# M0 — Forensic Audit & Training-Semantics Repair (PriCheXy-Net)

Status: **PASS** (see verdict) · Date: 2026-08-15 · Repo root: `/home/minhtt/Neups_workshop`

This document answers the owner question: *"Is the current training/evaluation stack
sufficiently correct, faithful, isolated, and reproducible that historical ideas such
as C4 and later C2+C4 can be re-tested on the restored PriCheXy-Net baseline?"*

**Scope discipline (enforced):** development-only. No TEST opened (firewall T13),
no dataset-scale training, no method training, no hyperparameter search, no
architecture/BatchNorm/mu/pairs/splits changes, no "fix" of BatchNorm, no deletion of
historical evidence, minimal repairs only. All tests run on CPU with tiny synthetic
tensors.

---

## 1. Pre-repair git state (recorded before any M0 file was created)

- cwd: `/home/minhtt/Neups_workshop`
- branch: `original-upstream`, HEAD: `29245d1f71571898d9527417df4ae3f63a8695f6` (pristine upstream, 49 tracked files)
- remote: `origin → https://github.com/minhmoidz/Neups_workshop.git`
- working tree: **0 modified tracked files**, 31 untracked entries (`archive/`, `chexnet/`,
  `data/`, `logs/`, `networks/`, `reproduction/`, `research_agent/`, `adaptive_reid/`)
- historical research lives on branch `main` (HEAD `641fbc3`, 411 files): `PLAN.md`,
  `RESEARCH_BRIEF.md`, `HANDOVER/*.md`, `research_agent/*.md`. No `git reset --hard`,
  `clean -fd`, `checkout ./`, or `restore .` was run; the dirty tree was preserved.

**M0 post-repair additions (all new untracked files, zero tracked files touched):**
`config_files/config_dev_restored_baseline.json`, `config_dev_c4.json`, `config_dev_c2c4.json`,
`research_agent/test_firewall.py`, `research_agent/m0_port/ACLoss.py`,
`research_agent/M0_historical_evidence_registry.json`, `research_agent/m0_tests/*`.

---

## 2. Restored baseline provenance

| Artifact | Path | SHA-256 (first 12) | Notes |
|---|---|---|---|
| Generator (released, mu=0.01, flow_field) | `networks/generator_lowest_total_loss_mu_0.01.pth` | `4d82dcdd1c1b` | 31,112,853 B; **matches upstream bit-for-bit** (T15 verified unchanged) |
| Classifier (DenseNet-121 pretrained) | `networks/pretrained_classifier.pth` | `8ad15b38286f` | — |
| Verification model (SNN resnet50) | `networks/pretrained_verification_model.pth` | `331efaed0c04` | — |
| Segmentation (historical branch only) | `archive/train_seg_unet/best.pth` | — | not part of restored baseline |

- **Operator:** LEGACY (released == upstream), bit-for-bit grid and image diff = 0.0
  (evidence `operator_audit_results.json`; T11). The 'corrected' operator is a separate
  dev branch, mu=0 identity-invariant (T12), and is NOT the baseline.
- **mu = 0.01**, `transform_mode = "legacy"`, attacker init = fresh ImageNet resnet50
  (pretrained-verification init A/B was rejected: seed-42 0.8148).
- **R1-FINAL locked baseline (TRAIN+VAL only):** batch 32, legacy operator, fresh attacker,
  10 seeds `[42,0,1,2,3,4,5,6,7,8]` → mean **0.6080 ± 0.0435**, median 0.6084,
  t-CI [0.5769, 0.6392], bootstrap CI [0.5814, 0.6321]; **APPROXIMATE REPRODUCTION** of
  paper 0.577 ± 0.040 (z = 0.71). `reproduction/BASELINE_LOCK.json`,
  `reproduction/OFFICIAL_TEST_LOCK.md` (TEST closed).
- `reproduction/workspace_prichexy/` is the restored baseline workspace (pristine +
  minimal no_grad patch in `utils/utils.py::train_snn`).

---

## 3. Upstream differential audit (pristine working tree vs historical `main`)

| Concern | Pristine (restored baseline) | Historical `main` (method branch) | Severity |
|---|---|---|---|
| Generator update semantics | step every batch (no accumulation) | accumulation bug FIXED (`test_grad_accum.py`; backward loss/acc, step+zero_grad at boundary) | OK |
| Batch size | 64 | 16 (+acc) — **T3: batch16+acc4 ≠ batch64 under BatchNorm** | P1 |
| ACLoss classifier state | `deepcopy(ac_model)` **every forward** (correct) | `loss_model` built once, `refresh()` **never called** → **STALE** | **P0** |
| C3 ensemble verifier | none | defaults `ver_ensemble_size=3`, `ver_restart_every=25` → silently ON | **P1** |
| C2 init | n/a | pads 2→3 channels, zero budget channel ⇒ uniform budget, mean==mu (T9/T10) | OK (mechanistic) |
| GaussianSmoothing | kernel 9, sigma 2 | same | OK |
| Dataset loading | preload all at init | lazy load in `__getitem__` | P2 (memory) |
| Operator | legacy == upstream | `build_sampling_grid` legacy/corrected + `compute_budget_map` + `deform` | legacy OK |

---

## 4. P0 — ACLoss stale classifier state (found & repaired)

**Root cause.** Upstream/restored baseline: `loss_model = copy.deepcopy(self.ac_model)`
on every forward ⇒ the generator always fights the **current** auxiliary classifier, and
the deepcopy **detaches** the generator gradient path from `ac_model`'s parameters.
Historical `main` `utils/ACLoss.py`: builds `loss_model` once in `__init__`, provides
`refresh()` but no call site exists anywhere in `utils/utils.py` ⇒ the generator-facing
classifier was **frozen at its init snapshot** for the whole run. Tests T4 prove:
pristine tracks current weights (loss changes after ac update), main-style build-once is
bit-stale (loss identical to init snapshot), repaired refresh tracks current.

**Minimal repair.** New port-ready module `research_agent/m0_port/ACLoss.py`: calls
`self.refresh()` at the start of `forward()`, i.e. loss model is rebuilt from the
current `ac_model` exactly as upstream's per-call deepcopy — preserving both the current
weights and the detached gradient path. The C4 extensions (`pos_weight`, and
`feature_loss_weight` with `detach()` on source features, T6) are kept with documented
baseline-equivalent defaults (None / 0.0 ⇒ baseline, T5). **T4e: repaired ACLoss output
is identical to pristine upstream across repeated ac_model updates** (allclose, 3 steps).

No repair was needed in the pristine restored baseline itself: its ACLoss is already the
correct upstream deepcopy-every-forward implementation (verified identical to
`reproduction/workspace_prichexy/utils/ACLoss.py`).

---

## 5. Test suite (T1–T15) — all PASS, plain-script runnable (no pytest dependency)

`research_agent/m0_tests/` · `python research_agent/m0_tests/run_all.py`

| Test | Checks | Status |
|---|---|---|
| T1 | accumulation_steps=1 ⇒ exactly one generator update per batch | PASS |
| T2 | accumulation equivalence on toy non-BN model (N=2,4) | PASS |
| T3 | batch16+acc4 ≠ batch64 under BatchNorm | PASS |
| T4 | ACLoss: pristine tracks current / main-style build-once STALE / repaired tracks current | PASS |
| T4e | repaired ACLoss == pristine upstream (bit-equivalent) | PASS |
| T5 | baseline compat: pos_weight=None and feature_loss_weight=0 == baseline BCE | PASS |
| T6 | C4: source features must be detached (generator grad only via anonymized branch) | PASS |
| T7 | verifier labels 1.0=same, 0.0=different; loss is BCE | PASS |
| T8 | DEV restored-baseline config forces ver_ensemble_size=1, budget=false, λ=0 (no silent C3) | PASS |
| T9 | C2 zero-budget channel ⇒ uniform budget, deformation == baseline | PASS |
| T10 | spatial budget mean == mu; zero channel ⇒ uniform | PASS |
| T11 | operator audit evidence: legacy==upstream (diff 0), legacy≠corrected | PASS |
| T12 | corrected mu=0 identity-invariant; legacy mu=0 NOT identity (border pinch) | PASS |
| T13 | TEST firewall fail-closed: dev passes, 'test' raises unless allowed | PASS |
| T14 | provenance hashes deterministic (config sha256, git HEAD/branch captured) | PASS |
| T15 | no frozen checkpoint modified (released generator SHA unchanged) | PASS |

Run output: 26/26 PASS (`M0 SUITE: ALL PASS`).

---

## 6. Test firewall + provenance recorder (development infrastructure)

- `research_agent/test_firewall.py`: `TestFirewall.check(mode)` raises for
  `test`/`final_test` unless `allow=True` (reserved for the approved final protocol);
  dev modes (`dev`/`val`/`probe`/`debug`) pass. `provenance_record()` computes a
  deterministic dict: python/torch/cuda/cudnn/gpu, git HEAD+branch, config sha256, and
  optional artifact sha256s — every run becomes reproducible/auditable.
- `config_files/config_dev_*.json`: three templates (restored baseline / C4 / C2+C4)
  carry explicit provenance keys (operator, mu, batch, accumulation, λ, use_budget_map,
  ver_ensemble_size=1, ver_restart_every=0, feature_loss_weight, ac_pos_weight, mode=dev).
- `research_agent/M0_historical_evidence_registry.json`: categorizes every historical
  result as TRUST_AS_MECHANISTIC / REQUIRES_RERUN / INVALID_FOR_METHOD_COMPARISON /
  CLOSED so future method comparisons never rely on buggy-code numbers.

---

## 7. Unresolved ambiguities

1. **Batch 64 vs 32 for method runs.** Upstream config is batch 64; the locked baseline
   (R1-FINAL) used batch 32 because it reproduced the paper closest. A method comparison
   must pick ONE batch size for both baseline and method. This is a frozen-protocol
   decision, not an M0 repair.
2. **Temporal conditioning / pairs** — unchanged, matches upstream.
3. **`ac_model` for C4** — the C4 config references the released pretrained classifier;
   whether C4 trains its own critic or reuses the released one is a protocol decision.
4. **Segmentation branch** — historical only; not part of restored baseline.

---

## 8. Verdict

**PASS.**

The restored baseline (pristine upstream + no_grad patch, released generator, legacy
operator, mu=0.01, fresh-attacker, 10-seed 0.6080±0.0435, APPROXIMATE REPRODUCTION) is
correct, faithful, and locked. The training/evaluation stack is now isolated: TEST is
firewalled, provenance is recorded, and the one P0 semantic drift found in the historical
method stack (stale ACLoss `loss_model`) has a minimal, provably-equivalent repair plus a
regression test. C4 and C2+C4 can be re-tested on this baseline **provided** (a) they run
with the repaired `m0_port/ACLoss.py`, (b) explicit ensemble/batch keys are set (T8), and
(c) the exact frozen protocol (batch size, critic source, seeds) is fixed before TEST opens.

Files changed by M0: **none** in the pristine restored baseline or any tracked file.
All M0 deliverables are new untracked files under `config_files/` and `research_agent/`.