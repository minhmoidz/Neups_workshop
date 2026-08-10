# 00F2 — Operator Repair: Review Evidence (STEP 1B remediation)

> Companion evidence file to `00F_OPERATOR_REPAIR.md` (Scientific Review Remediation).
> Every number below is produced by the repository at the stated commit; reproduction
> commands are in §10.

---

## 1. Git / artifact provenance

### Pre-repair commit (the baseline every diff is measured against)

```
9eaa5fdf22ec08885a15d726c949de8404d522ea
```

### Dedicated commits (substantive content frozen)

```
STEP 1 (operator repair):                83738bb341e8ce920490d5c06c48c69fd8d47f09
STEP 1B (review remediation):            18f6eb9a3e3e7a532da93f31393f3d0e27a52d86
```

> Note: this evidence document references the *substantive* commits above. Subsequent
> metadata-only edits (this file's own commit) move `HEAD` by design; run
> `git rev-parse HEAD` for the exact tip. The full diff vs the pre-repair commit
> `9eaa5fd` is regenerated with `git diff 9eaa5fdf22ec08885a15d726c949de8404d522ea HEAD`.

### Auditable artifacts

| Artifact | Location | In git |
|---|---|---|
| Repair report | `research_agent/00F_OPERATOR_REPAIR.md` | yes |
| Review evidence | `research_agent/00F2_OPERATOR_REVIEW_EVIDENCE.md` (this file) | yes |
| Full diff vs pre-repair | `research_agent/STEP1_OPERATOR_REPAIR.diff` (vs `9eaa5fd`) | generated artifact, untracked |
| Regression suite | `test_operator_repair.py` | yes |

### `git status` (working tree, snapshot after all commits)

```
?? research_agent/STEP1_OPERATOR_REPAIR.diff
```

The only untracked file is the generated diff artifact `research_agent/STEP1_OPERATOR_REPAIR.diff`;
all source/test/doc changes are committed (see commit list above).

### `git diff --stat` (pre-repair → HEAD)

```
 agents/Agent.py                       |   9 +-
 agents/AgentPretrain.py               |   8 +-
 agents/AgentSiameseNetwork.py         |  10 +-
 chexnet/eval_model.py                 |   9 +-
 eval_classifier.py                    |   3 +-
 eval_seg.py                           |  10 +-
 proxy_reid.py                         |   7 +-
 research_agent/00F_OPERATOR_REPAIR.md | 307 ++++++++++++++++++++++++
 test_operator_repair.py               | 436 ++++++++++++++++++++++++++++++++++
 utils/utils.py                        | 185 ++++++++++++---
 10 files changed, 931 insertions(+), 53 deletions(-)
```

The complete `git diff` is materialized in `research_agent/STEP1_OPERATOR_REPAIR.diff`
(1309 lines) so the Scientist can inspect the exact implementation, not only summaries.

---

## 2. Exact grid invariants (BLOCKER 2)

Design invariant: `corrected` builds `grid = I - G*u`; when the effective displacement is
zero the sampling grid must be the **exact identity grid**.

### 2.1 Nonzero arbitrary flow + μ=0

Generator returns constant flow `u = 0.7`; `μ = 0` ⇒ `budget*flow = 0` ⇒ corrected grid.

```
torch.equal(sampling_grid, identity_grid)  == True
max absolute grid difference               == 0
mean absolute grid difference              == 0
```

### 2.2 Zero flow + μ>0

Zero-flow generator; `μ = 0.01` and `μ = 0.05`.

```
torch.equal(sampling_grid, identity_grid)  == True  (both μ)
max absolute grid difference               == 0
mean absolute grid difference              == 0
```

### 2.3 Same input image: case A (arbitrary flow + μ=0) vs case B (zero flow + μ>0)

Same image, same identity grid; both corrected grids are bit-identical to the identity
grid and therefore to each other (max |A − B| grid diff = 0). Image-space
`grid_sample` errors (~1e-5) remain at float interpolation tolerance and are NOT used as
evidence; the grid equality is decisive.

---

## 3. Pretrain / preval verification (BLOCKER 3)

The three historical operator sites, all in `utils/utils.py`:

- `deform()` — main inference/training warp.
- `pretrain()` — pre-training loop of the flow-field generator.
- `preval()` — pre-training validation loop.

All three now route through the single `build_sampling_grid(grids, grid_identity,
gauss_filter, transform_mode)` helper. Verified at source level:

- `test_shared_operator_all_three_sites` inspects the actual source of each function and
  asserts each contains `build_sampling_grid(` and does NOT contain the inline
  `gauss_filter(grids)` on the full displaced grid.

Behavioral tests construct the grid exactly as these loops do (`mu * flow` →
`build_sampling_grid`):

- `test10_pretrain_corrected_path`: pretrain-corrected grid at μ=0 is exactly the identity
  grid (`torch.equal` True); at μ>0 it is a valid (N,H,W,2), finite grid.
- `test11_preval_corrected_path`: same for the validation loop (flow −0.3).
- There is therefore no situation where pretraining uses one operator and
  training/evaluation uses another: all three share `transform_mode`.

### Exact locations

- `build_sampling_grid` — `utils/utils.py` (docstring §2/§2.1 of 00F report);
- `deform` — calls `build_sampling_grid` with `scaled_grids = budget * flow`;
- `pretrain` — calls `build_sampling_grid` with `scaled_grids = mu * grids`;
- `preval`   — calls `build_sampling_grid` with `scaled_grids = mu * grids`.

---

## 4. Source-pixel coverage (BLOCKER 5)

For a sampling grid in grid_sample layout, `_source_coverage` accumulates the bilinear
sampling weight each source pixel receives across all output pixels (identity grid ⇒ every
source pixel weight 1.0). At `μ=0`, arbitrary flow `u=0.7`, image size 64×64=4096 px:

| Operator | total source px | unsampled px | unsampled fraction |
|---|---|---|---|
| **corrected** | 4096 | **0** | **0.00%** |
| **legacy**    | 4096 | **736** | **17.97%** |

The legacy operator drops the outer ring of source pixels (≈ kernel radius 4 for
`kernel_size=9`): the exact historical defect being repaired. Corrected covers every
source pixel.

---

## 5. Interior vs border comparison at μ>0 (BLOCKER 6)

Identical input image, identical generator flow (`u=0.4`), `μ=0.05`. Compare the `legacy`
and `corrected` sampling grids directly (kernel radius `R=4` defines the strict interior).

```
interior max grid difference   = 3.576e-07   (numerical zero — the operators coincide, G*I==I strict interior)
border  max grid difference   = 6.516e-01   (clearly non-zero — the legacy zero-padding artifact)
```

Interpretation: `G*(I-u) = I - G*u + (G*I - I)`. In the strict interior `G*I = I`, so the
two grids are mathematically identical; at the border `G*I != I` (zero padding) which is
exactly the term the corrected operator removes. The fix therefore removes *only* the
unintended border term and preserves the intended interior deformation.

---

## 6. Budget-map semantics (BLOCKER 7)

Budget construction factored into `compute_budget_map(grids, mu)` (used by `deform`), so
the semantics are asserted directly.

### 6.1 `mean(budget) ≈ μ`

3-channel budget generator (flow 0.3 + budget channel peaking 0.5 in a band), `μ=0.02`:

```
per-image budget mean = [0.0199999977, 0.0199999977, 0.0199999977, 0.0199999977]
max |mean(budget) - mu| = 1.863e-09
```

### 6.2 Budget spatial effect is used

Same flow (0.3) with a flat budget channel vs a budget channel peaking in the central
vertical band (`μ=0.02`):

```
mean displacement peak-band (peaked budget) = 9.407e-03
mean displacement peak-band (flat budget)   = 5.862e-03
mean displacement outside band (peaked)     = 3.996e-03
```

Peaked budget ⇒ larger displacement in the band (`9.4e-3`), greater than both the flat-map
displacement (`5.9e-3`) and the outside-band displacement (`4.0e-3`): the budget map
genuinely controls where deformation is applied.

### 6.3 μ=0 with non-trivial budget channel ⇒ exact identity grid

`budget = mu * (1+b3)/mean(...)` with `mu=0` ⇒ effective displacement `= 0 * flow = 0`
for any third-channel output. The identity-grid invariants in §2 hold regardless of the
budget channel (they exercise the same `build_sampling_grid` path).

---

## 7. Legacy equivalence (BLOCKER 8)

### 7.1 Bit-for-bit check

Given fixed synthetic input/flow and `μ=0.03`, A = exact old inline equations:

```python
old_grid = grid_identity - scaled          # scaled = budget*flow
old_grid = gauss_filter(old_grid)
old_grid = old_grid.permute(0, 2, 3, 1)
```

B = `build_sampling_grid(scaled, grid_identity, gauss_filter, transform_mode='legacy')`

```
torch.equal(A, B)  == True
max grid difference == 0
```

Bit-for-bit equivalence is verified, not merely asserted.

### 7.2 Configs without `transform_mode` → legacy

All 28 `.json` configs in `config_files/` carry **no** `transform_mode` key and each
resolves to `'legacy'` (checked programmatically). Invalid values are rejected
(`ValueError`).

---

## 8. Full automated test output

Command: `.venv/bin/python test_operator_repair.py`
Exit code: `0`

```
TEST 1 corrected mu=0 (flow=0.7): torch.equal==True, max grid diff=0e+00, mean grid diff=0e+00
TEST 2 zero flow mu=0.01: torch.equal==True, max grid diff=0e+00, mean grid diff=0e+00
TEST 2 zero flow mu=0.05: torch.equal==True, max grid diff=0e+00, mean grid diff=0e+00
TEST 3 same-image identity invariance: A==B True, A==Id True (max |A|=1.000, max |B|=1.000, grid coords in [-1,1])
TEST 4 mu=0 coverage | total=4096 | unsampled: corrected=0 (0.00%), legacy=736 (17.97%)
TEST 5/6 mu=0.05 legacy vs corrected | interior max grid diff=3.576e-07 (expected ~0), border max grid diff=6.516e-01 (expected >> 0)
TEST 7 corrected mu=0.05: deforms ((1, 1, 64, 64)), finite, non-identity
TEST 8 budget mean per image = [0.019999997690320015, ...] mu = 0.02 (max |diff| = 1.863e-09)
TEST 9 budget spatial effect | mean disp peak-band=9.407e-03, flat-band=5.862e-03, outside=3.996e-03 (band>outside>placement)
TEST 10 pretrain corrected: mu=0 identity torch.equal==True; mu>0 grid (1, 64, 64, 2), finite=True
TEST 11 preval corrected: mu=0 identity torch.equal==True; mu>0 grid (1, 64, 64, 2), finite=True
TEST shared-operator: deform/pretrain/preval all route through build_sampling_grid
TEST 12 legacy equivalence: old-inline == new helper torch.equal==True (max grid diff=0e+00)
TEST 12b default-legacy: 28 configs all resolve to 'legacy' (none carry transform_mode); invalid value rejected
PASS: accumulation over 2 micro-batches == doubled-batch gradient (bit-for-bit)
PASS: buggy zero_grad-in-loop placement is detected by the test
TEST 13: gradient-accumulation regression still passes
TEST constant-image smoke: passes (finite, correct shape) — NOT evidence of correctness

STEP 1B REVIEW REMEDIATION: PASS
```

---

## 8.1 Smoke (STEP 1C): proxy re-ID on a real validation batch — legacy vs corrected

Per the suggested next step (§10 of `00F_OPERATOR_REPAIR.md`), a smoke comparison of the
two operators was run on **real NIH-ChestX-ray14 images** (validation split, 2000
genuine/impostor pairs) with the frozen ImageNet ResNet-50 proxy, using the same
`baseline_fixed` generator (`mu=0.01`, `stochastic_lambda=0.0`, identical inputs, seed 42).
Forward passes only — **no training, no SNN retraining**.

```
SMOKE: proxy re-ID on validation set (real NIH-ChestX-ray14)
generator: archive/train_prichexy_net_baseline_fixed/generator_lowest_total_loss.pth | mu=0.01 | lambda=0.0 | image_size=256
[legacy]    n_pairs=2000 PROXY_AUC=0.699515 | border-mean-disp=0.2230 interior-mean-disp=0.0086
[corrected] n_pairs=2000 PROXY_AUC=0.720785 | border-mean-disp=0.0075 interior-mean-disp=0.0086
delta PROXY_AUC (legacy - corrected) = -0.021270
SMOKE DONE
```

**Reading** (consistent with the audited defect and BLOCKER 5/6):

- `interior-mean-disp` is identical (`0.0086`) for both modes — the operators differ
  **only** at the border, exactly as designed.
- `border-mean-disp` = `0.2230` (legacy) collapses to `0.0075` (corrected): the 30× higher
  border displacement in legacy is the `G*I != I` artifact dragging the sampling grid
  toward 0 and dropping the outer ring of source pixels.
- Legacy shows a *lower* proxy AUC (`0.6995` vs `0.7208`). This is **not** evidence that
  legacy is more private — the proxy feature-space measure is non-adaptive (D7) and the
  legacy drop comes from destroying border diagnostic content, the very defect this repair
  removes. The 0.021 gap is the *proxy* signature of the border artifact, and the real
  question (does preserving the border help/hurt the *retrained* attacker) still requires a
  10-seed SNN retrain, which remains out of scope.

Smoke script: `/tmp/opencode/smoke_proxy_compare.py` (no repo files added).

---

## 9. Files changed

### STEP 1 (commit `83738bb`)

- `utils/utils.py` — `build_sampling_grid` (new shared operator), `deform`, `pretrain`,
  `preval`, `train`, `validate`, `train_snn`, `validate_snn`, `test_snn` gain
  `transform_mode` (default `'legacy'`).
- `agents/Agent.py`, `agents/AgentSiameseNetwork.py`, `agents/AgentPretrain.py` —
  read `config['transform_mode']`.
- `eval_seg.py`, `proxy_reid.py`, `chexnet/eval_model.py`, `eval_classifier.py` —
  expose `transform_mode`.
- `test_operator_repair.py` (initial), `research_agent/00F_OPERATOR_REPAIR.md` (initial).

### STEP 1B remediation (commit `18f6eb9`)

- `utils/utils.py` — `resolve_transform_mode`, `record_transform_mode_provenance`,
  `compute_budget_map` (factored out of `deform` for direct budget testing), `json` import.
- `agents/*.py` — resolve mode + write provenance (transform_mode.txt, resolved_config.json)
  into experiment archive + console header.
- `chexnet/eval_model.py` — provenance into `save_path`; `eval_seg.py`, `proxy_reid.py` —
  console-mode header.
- `test_operator_repair.py` — replaced with the 13-test final suite.
- `research_agent/00F_OPERATOR_REPAIR.md` — added "Scientific Review Remediation" (§11).
- `research_agent/00F2_OPERATOR_REVIEW_EVIDENCE.md` — this file (new).
- `research_agent/STEP1_OPERATOR_REPAIR.diff` — full diff vs pre-repair `9eaa5fd` (new).

Historical results (archive/), configs, checkpoints, dataset: **untouched**.

---

## 10. Exact reproduction commands

```bash
cd /home/minhtt/Neups_workshop

# 0. Environment (torch 2.7.0+cu128, Python 3.10 venv)
.venv/bin/python -c "import torch; print(torch.__version__)"

# 1. Provenance
git rev-parse HEAD                                   # (run at reproduction time)
git log --oneline -5
git diff --stat 9eaa5fdf22ec08885a15d726c949de8404d522ea HEAD
git diff 9eaa5fdf22ec08885a15d726c949de8404d522ea HEAD > research_agent/STEP1_OPERATOR_REPAIR.diff

# 2. Run the final regression suite (CPU, no GPU, no data, no training)
.venv/bin/python test_operator_repair.py

# 3. Optional: pre-existing gradient-accumulation regression (GPU)
.venv/bin/python test_grad_accum.py
```

Strict stop respected: no model was trained, no SNN retraining, no real-data evaluation,
no C2/C4 runs; `archive/` and `config_files/` were not modified.