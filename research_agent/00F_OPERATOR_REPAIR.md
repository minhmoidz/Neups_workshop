# 00F — Operator Repair: Gaussian smoothing on the sampling grid

> Status: **DONE** (2026-08-11). Synthetic tests green on CPU; no retraining performed (strict stop).

---

## 1. The scientific defect (audit of `utils.deform`)

The PriCheXy-Net transformation applies a Gaussian smoother to the *whole displaced
sampling grid*:

```python
# legacy (historical behavior, utils/utils.py)
grids = grid_identity - budget * grids      # full displaced grid  (I - u)
grids = gauss_filter(grids)                 # G * (I - u)
grids = grids.permute(0, 2, 3, 1)
```

Because `GaussianSmoothing` is applied with **zero padding**, `G * I != I` near the
image borders: the smoother drags border coordinates toward 0. Consequences:

- **`mu = 0` does not give the identity.** A zero deformation budget should reproduce
  the input exactly (`T(x) = x`); instead, with a nonzero learned flow the legacy
  operator still warps and darkens a ~4 px ring of border pixels (kernel radius 4 for
  `kernel_size=9`). Measured on a synthetic image: **`max |T(x) - x| = 0.977`** at
  `mu=0` (legacy) vs **`7.6e-06`** (corrected).
- The bug silently inflates the effective deformation strength at borders and drops
  diagnostic content in the outer ring of the radiograph.
- Same defect is present in its entirety in the pre-training loops `utils.pretrain` /
  `utils.preval`.

## 2. The fix (minimal, `utils/utils.py`)

Smooth **only the displacement component**, never the identity term:

```python
# corrected
displacement  = budget * grids             # u
sampling_grid = grid_identity - gauss_filter(displacement)   # I - G*u
```

Gaussian convolution is linear, so `G (I - u) = G I - G u`. The legacy artifact is
exactly the term `G I - I`, which is removed. A zero flow / zero budget now returns the
exact identity grid.

### 2.1 Explicit naming (Task 7)

Both operators are kept, selected by a `transform_mode` flag:

- `transform_mode='legacy'`    — historical `G * (I - u)` **bit-for-bit reproducible** (default).
- `transform_mode='corrected'` — `I - G * u` (the scientific fix).

Shared helper `build_sampling_grid(grids, grid_identity, gauss_filter,
transform_mode)` is the single place the operator is implemented. `deform`, `pretrain`
and `preval` all route through it, so train / SNN-retrain / eval paths use the same
operator.

## 3. Where `transform_mode` is threaded (Guarantee: corrected usable end-to-end)

| Entry point                       | File                                   | Mechanism                                   |
|-----------------------------------|----------------------------------------|---------------------------------------------|
| `utils.deform`                    | `utils/utils.py:145`                    | kwargs `transform_mode` (default `'legacy'`) |
| `utils.pretrain` / `utils.preval` | `utils/utils.py:187 / 270`               | kwargs, default `'legacy'`                   |
| `utils.train` / `validate`        | `utils/utils.py`                        | kwargs, default `'legacy'`                   |
| `utils.train_snn/validate_snn/test_snn` | `utils/utils.py`                | kwargs, default `'legacy'`                   |
| `agents/Agent.py`                 | reads `config['transform_mode']`        | default `'legacy'`                           |
| `agents/AgentSiameseNetwork.py`   | reads `config['transform_mode']`        | default `'legacy'`                           |
| `agents/AgentPretrain.py`         | reads `config['transform_mode']`        | default `'legacy'`                           |
| `eval_seg.py`                     | CLI `--transform_mode`                  | default `'legacy'`                           |
| `proxy_reid.py`                   | CLI `--transform_mode`                  | default `'legacy'`                           |
| `chexnet/eval_model.py`           | `make_pred_multilabel(..., transform_mode)` | default `'legacy'`                        |
| `eval_classifier.py`              | reads `config['transform_mode']`        | default `'legacy'`                           |

Because the default is `legacy` everywhere, **all commit-history results stay
reproducible unchanged** (bit-for-bit: the code path for `'legacy'` is exactly the old
`grid_identity - u` → `gauss_filter` sequence).

## 4. Automated tests (Task 4) — `test_operator_repair.py`

All run on **CPU** (no GPU needed → green CI even before GPU). 9 tests:

- **TEST A** `mu=0` (non-trivial flow): corrected returns input (max diff 3.4e-05, float noise of
  `grid_sample`); legacy darkens/corrupts the border (documented).
- **TEST B** zero flow, `mu>0`: corrected returns input exactly.
- **TEST A'** `stochastic_lambda=0`, `mu=0`: identity preserved.
- **TEST C** constant image deformed with a real field: no artificial border corruption (value-span 0).
- **TEST D** identity-grid invariant: `|I - grid| = 0` corrected vs `0.652` legacy.
- **TEST E** legacy vs corrected checkerboard: border deviation `7.3e-01` at identical `mu`/flow.
- **TEST μ>0**: corrected still deforms, correct shape, finite, smooth, no NaN.
- **TEST budget-map (3-channel)**: corrected path works, finite.

Run: `.venv/bin/python test_operator_repair.py`

## 5. Numeric comparison legacy vs corrected (Task 5)

Synthetic 64×64 diagnostic-like image (mean 0.5, std 0.25), constant flow `u=0.4`,
Gaussian `k=9, σ=2`, identity grid `linspace(-1,1)`:

| mu  | mode      | mean|dT| | max|dT| | borderMSE | innerMSE |
|-----|-----------|----------|----------|-----------|----------|
| 0.0 | legacy    | 8.51e-02 | **9.77e-01** | 1.02e-01 | 7.6e-12 (clean interior) |
| 0.0 | corrected | 1.14e-06 | **7.63e-06** | 1.30e-12 | 3.99e-12 |
| 0.01| legacy    | 1.49e-01 | 9.97e-01 | 1.06e-01 | 1.03e-02 |
| 0.01| corrected | 8.06e-02 | 2.36e-01 | 8.39e-03 | 1.03e-02 |
| 0.05| legacy    | 3.13e-01 | 9.97e-01 | 1.52e-01 | 1.30e-01 |
| 0.05| corrected | 2.93e-01 | 8.63e-01 | 1.11e-01 | 1.30e-01 |

Reading: **interior deformation is identical** (`innerMSE` equal) — the operators differ
*only* in what happens at the border. The legacy operator adds a huge **satellite
border corruption** (borderMSE ~0.10–0.15 which is a massive fraction of the observed
range), while the corrected operator removes it. At `mu=0` legacy does not reproduce the
input at all.

## 6. Task 6 — corrected operator sanity at μ>0

- ✅ Still deforms: `mu>0` with a real flow moves pixels (orientation/sign unchanged: a
  positive flow shifts the sampling position in the same direction as legacy).
- ✅ Smoothing preserved: the Gaussian is now applied to the displacement only, output is
  smooth, no salt-and-pepper.
- ✅ No dimension regression: output `(N,1,H,W)` as before.
- ✅ No NaN / Inf at large scale (`mu=0.5, flow=0.8`).
- ✅ Budget-map (3-channel) path works in corrected mode (finite, correct shape).

## 7. Reproducibility of historical runs (Task 3)

- Default `transform_mode='legacy'` everywhere ⇒ existing scripts/configs and all
  archived results are unchanged.
- `stochastic_lambda` path is orthogonal and preserved identically (`mu` budget scaling
  is applied *before* `build_sampling_grid` in both modes; in legacy the sequence maps
  exactly to the old code).
- No training state, checkpoints, configs, or data were modified. No model was trained.

## 8. Strict stop — no data / no training touched

Per task rules, this work did **NOT** train an anonymization model, did not retrain the
SNN, did not download or alter training data. Only operator code + tests + this report
changed.

## 9. Files changed

- `utils/utils.py` — new `build_sampling_grid`; `deform`, `pretrain`, `preval`,
  `train`, `validate`, `train_snn`, `validate_snn`, `test_snn` gain `transform_mode`
  (default `'legacy'`).
- `agents/Agent.py`, `agents/AgentSiameseNetwork.py`, `agents/AgentPretrain.py` — read
  `config['transform_mode']` (default `'legacy'`).
- `eval_seg.py`, `proxy_reid.py`, `chexnet/eval_model.py`, `eval_classifier.py` —
  exposed `transform_mode`.
- `test_operator_repair.py` — NEW regression suite (Task 4, tests A–E + μ>0 + budget map).
- `research_agent/00F_OPERATOR_REPAIR.md` — this report.

## 10. Smoke quantification on real data (STEP 1C) — DONE

Run a one-epoch smoke comparison of `legacy` vs `corrected` on a real NHI-XR validation
batch using the frozen-image proxy (`proxy_reid` machinery, forward passes only — no
training, no SNN retraining). Same generator `baseline_fixed`, `mu=0.01`, identical
inputs, seed 42:

| mode | PROXY_AUC (2000 pairs) | border-mean-disp | interior-mean-disp |
|---|---|---|---|
| legacy    | 0.699515 | **0.2230** | 0.0086 |
| corrected | 0.720785 | **0.0075** | 0.0086 |

- Interior displacement identical (`0.0086`) in both modes — the operator diff is
  border-only, as designed.
- Legacy's border displacement is ~30× higher (`0.2230` vs `0.0075`) — the quantifiable
  `G*I != I` artifact.
- Legacy's lower proxy AUC (`0.6995` vs `0.7208`) is the *proxy* price of destroying
  border diagnostic content, not a privacy gain; the non-adaptive proxy cannot answer
  the real question. Whether the preserved border helps/hurts a **retrained** attacker
  still needs the 10-seed SNN protocol — out of scope for STEP 1.

Full reading in `00F2_OPERATOR_REVIEW_EVIDENCE.md` §8.1.

---

## 11. Scientific Review Remediation (STEP 1B)

The scientific reviewer returned **FAIL** on the initial fix: the reviewer accepted the
mathematics (`legacy: G*(I-u)`, `corrected: I-G*u`) but could not inspect the actual
implementation artifacts and found several regression tests insufficiently
discriminating. Every blocker is resolved below and documented with evidence in
`research_agent/00F2_OPERATOR_REVIEW_EVIDENCE.md`.

### 11.1 BLOCKER 1 — Make the implementation reviewable ✅

The complete repair is now available as auditable artifacts:

- `research_agent/00F_OPERATOR_REPAIR.md` — this report.
- `research_agent/00F2_OPERATOR_REVIEW_EVIDENCE.md` — review evidence (full output).
- `research_agent/STEP1_OPERATOR_REPAIR.diff` — full diff vs pre-repair commit `9eaa5fd`.
- `test_operator_repair.py` — 13-test final suite (all green).
- Dedicated STEP-1 commit: `83738bb`.
- STEP-1B remediation commit hash: recorded in `00F2_OPERATOR_REVIEW_EVIDENCE.md` §1.

`git status`, `git diff --stat`, `git diff`, and current commit hash are captured in
`00F2_OPERATOR_REVIEW_EVIDENCE.md` §1.

### 11.2 BLOCKER 2 — Resolve the μ=0 numerical question ✅

The different image errors (`3.457e-05` vs `6.974e-06`) came from different test images and
image-space float interpolation, NOT from a non-identity grid. The grid invariant is now
tested **directly** with `torch.equal`:

- **TEST 1** nonzero flow (`u=0.7`) + `μ=0`: `torch.equal(grid, identity)==True`,
  max grid diff **0**, mean grid diff **0**.
- **TEST 2** zero flow + `μ>0`: `torch.equal(grid, identity)==True`, max=mean=**0**.
- **TEST 3** same image, case A (arbitrary flow+μ=0) vs case B (zero flow+μ>0):
  both grids equal the identity grid **exactly** (max |A-B| = 0, max |A - Id| = 0).

### 11.3 BLOCKER 3 — Verify pretrain / preval all use the shared operator ✅

The three historical operator sites are `deform()` (`utils/utils.py`), `pretrain()`
(`utils/utils.py`), `preval()` (`utils/utils.py`). All three now call the single
`build_sampling_grid` helper. Verified two ways:

1. **Source-level** (`test_shared_operator_all_three_sites`): `inspect.getsource` asserts
   each of `deform`/`pretrain`/`preval` calls `build_sampling_grid(` and does NOT contain
   the inline `gauss_filter(grids)` on the full displaced grid.
2. **Behavioral**: `test10_pretrain_corrected_path` and `test11_preval_corrected_path`
   construct the grid exactly as `pretrain`/`preval` do (`mu * flow` →
   `build_sampling_grid(..., mode)`): identity at μ=0 (`torch.equal`), deforming at μ>0.

### 11.4 BLOCKER 4 — Replace the constant-image test ✅

The constant-image test is retained **only as a labeled smoke test** (finite / correct
shape) and is explicitly flagged as *NOT evidence of correctness*. It is no longer used
for any claim. The high-value tests added instead: source-pixel coverage (BLOCKER 5) and
legacy/corrected interior-vs-border comparison (BLOCKER 6).

### 11.5 BLOCKER 5 — Test the actual historical defect (source-pixel coverage) ✅

New `_source_coverage` computes the accumulated bilinear sampling weight per source pixel
for a given sampling grid at `μ=0`:

- **corrected**: every source pixel is sampled — unsampled = **0 / 4096 (0.00%)**.
- **legacy**: **736 / 4096 (17.97%)** source pixels receive zero/negligible weight — the
  entire outer ring (~4 px, kernel radius) is dropped, exactly the audited defect.

### 11.6 BLOCKER 6 — Strengthen non-zero-μ test (interior == , border !=) ✅

Identical flow + `μ=0.05`, comparing `legacy` vs `corrected` **sampling grids**:

- interior (strict, kernel radius margin): max grid diff **3.576e-07** (numerical zero).
- border: max grid diff **6.516e-01** (clearly non-zero).

This proves the fix removes *only* the unintended border term caused by `G*I != I` while
preserving the intended interior deformation (which is identical under both operators).

### 11.7 BLOCKER 7 — Strengthen budget-map test ✅

Budget construction is factored into `compute_budget_map` (used by `deform`), so the
semantics can be asserted directly (new `test8`/`test9`):

1. `mean(budget) == μ` per image: measured `0.02` vs `μ=0.02`
   (max |diff| = `1.863e-09`).
2. Spatial effect used: peaked budget channel raises mean displacement in the band to
   `9.407e-03` vs `5.862e-03` flat and `3.996e-03` outside the band (band > flat, band >
   outside).
3. `μ=0` with a non-trivial budget channel still yields the **exact** identity grid
   (covered by TEST 1/2 path; the budget channel multiplies to zero effective
   displacement).

### 11.8 BLOCKER 8 — Verify legacy reproducibility (not just assert it) ✅

- `test12`: runs the exact old inline equations `old = grid_identity - u`;
  `old = gauss(old); old = old.permute(...)` against
  `build_sampling_grid(..., 'legacy')` → `torch.equal == True`, max grid diff **0**.
  Bit-for-bit equivalence is now verified, not claimed.
- `test12b`: all 28 existing configs (none carry `transform_mode`) resolve to `'legacy'`;
  an invalid value is rejected with `ValueError`.

### 11.9 BLOCKER 9 — Mode provenance (legacy/corrected made explicit) ✅

The resolved mode is now deterministic and recorded at runtime:

- `utils.resolve_transform_mode(value)` — canonical resolution, default `'legacy'` for
  missing keys, rejects anything else.
- `utils.record_transform_mode_provenance(mode, save_path, config)` writes `transform_mode.txt` and a
  `resolved_config.json` (full config + explicit `transform_mode`) into the experiment archive, and prints a
  `[transform_mode] resolved mode for this run: ...` console header.
- Wired into `Agent`, `AgentSiameseNetwork`, `AgentPretrain`, `chexnet/eval_model.py`
  (result metadata), plus console headers in `eval_seg.py`, `proxy_reid.py`.
- No mandatory config key was introduced → old configs keep resolving to `legacy` and
  remain bit-for-bit reproducible, but the *resolved* mode is explicit in saved records.

### 11.10 Updated automated suite — 13 required tests ✅

`test_operator_repair.py`:

| # | Test | What it verifies |
|---|------|------------------|
| 1 | arbitrary nonzero flow + μ=0 | exact identity **grid** (`torch.equal`) |
| 2 | zero flow + μ>0 | exact identity **grid** (`torch.equal`) |
| 3 | same-image μ=0 vs zero-flow | both construct the same identity grid |
| 4 | legacy vs corrected coverage at μ=0 | legacy drops 17.97% source px, corrected 0% |
| 5 | legacy vs corrected interior | equal to numerical zero (3.6e-7) |
| 6 | legacy vs corrected border | clearly different (0.652) |
| 7 | corrected μ>0 | deformation active, finite, correct shape |
| 8 | budget `mean ≈ μ` | max |diff| = 1.9e-9 |
| 9 | budget spatial effect | band > flat, band > outside |
| 10 | pretrain corrected path | identity at μ=0, grid correct at μ>0 |
| 11 | preval corrected path | identity at μ=0, grid correct at μ>0 |
| 12 | old-inline == new legacy helper | `torch.equal`, max diff 0 |
| 12b | old configs → legacy | 28/28 configs; invalid value rejected |
| 13 | gradient-accumulation regression | still passes (unchanged) |

### 11.11 Output

```
STEP 1B REVIEW REMEDIATION: PASS
```

(Full per-test output: `research_agent/00F2_OPERATOR_REVIEW_EVIDENCE.md` §8.)

### 11.12 Files changed in remediation

- `utils/utils.py` — `resolve_transform_mode`, `record_transform_mode_provenance`,
  `compute_budget_map` (factored out of `deform`); `json` import.
- `agents/Agent.py`, `agents/AgentSiameseNetwork.py`, `agents/AgentPretrain.py` —
  resolve + record provenance.
- `chexnet/eval_model.py`, `eval_seg.py`, `proxy_reid.py` — provenance header/metadata.
- `test_operator_repair.py` — replaced with the 13-test suite above.
- `research_agent/00F2_OPERATOR_REVIEW_EVIDENCE.md`, `research_agent/STEP1_OPERATOR_REPAIR.diff` — new.
- `research_agent/00F_OPERATOR_REPAIR.md` — this remediation section.