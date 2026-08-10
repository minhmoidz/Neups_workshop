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

## 10. Suggested next step (out of scope here)

Run a one-epoch smoke comparison of `legacy` vs `corrected` on a real NHI-XR validation
batch using `proxy_reid.py --transform_mode corrected` to quantify the border-effect
component of the Re-ID signal before committing to a 10-seed rerun.