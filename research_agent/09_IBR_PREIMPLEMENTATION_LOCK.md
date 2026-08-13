# STEP 6A — IBR Pre-Implementation Scientific + Artifact Lock

**Phase:** design / evidence assembly only. **NO TRAINING. NO TEST. NO NEW EXPERIMENT.**
**Method selected:** PRIMARY = **Identity-Bottleneck Resynthesis (IBR)**.
**Purpose:** resolve the four load-bearing issues (segmentation teacher, IBR graph, identifiable bottleneck, donor safety) plus development gates, re-score, implementation spec, compute estimate, and a final go/no-go.

Reference documents requested by this step (`08_PHASE2_POSITIVE_METHOD_DESIGN.md`, `08A_ORIGINAL_PAPER_CODE_RECONCILIATION.md`, `04_METHOD_SUCCESS_CRITERIA.md`, `04B_SUCCESS_CRITERIA_AMENDMENT.md`) are **not present in the repository** (checked at STEP 6A start, clean tree at `314efad`). This lock therefore stands alone and is internally consistent with the frozen STEP 3D / 3H / 4A / 4B / 4D evidence and the STEP 6A prompt itself.

---

## # 1. Segmentation checkpoint verification

**Restored / verified — exact artifact recovered. No replacement trained.**

| Attribute | Value |
| --------- | ----- |
| Exact path | `archive/train_seg_unet/best.pth` |
| Full SHA-256 | `2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0` |
| File size | 7,813,407 bytes (≈ 7.45 MB) |
| Expected provenance | STEP 3C segmentation checkpoint (digest begins `2dfdcf9b…`) — **matches exactly** |
| Recorded reference | `research_agent/03H_corrected_segmentation.json` `segmentation_sha256 = 2dfdcf9b…`; `03H_CORRECTED_BASELINE_UTILITY.md` line 119 `SHA-256 2dfdcf9b…` |
| Format | dict with keys `model`, `epoch`, `mean_dice`, `init_features`, `dice`, `iou` |
| Architecture | `UNetSeg(in_channels=1, out_channels=3, init_features=16)` (`networks/UNetSeg.py`) — 1,945,285 parameters |
| Epoch / metric | epoch 20, recorded `mean_dice = 0.9548` (matches VALIDATION reference 0.9550 below) |
| Loadability | `torch.load(weights_only=False, map_location='cpu')` OK |
| Strict state-dict load | `UNetSeg(1,3,16).load_state_dict(chk['model'], strict=True)` → missing=0, unexpected=0 |
| Forward | 1×1×256×256 → 1×3×256×256 sigmoid maps, range [0,1] — OK |

This is the exact frozen STEP 3C segmentation teacher. Verified hash, loadability, and architecture compatibility. **No substitution.**

---

## # 2. Corrected computational graph

The previous S1 description conflicted SELF and ANON reconstruction. The corrected graph:

```
(z_id, z_med) = E(x)                       # single shared encoder trunk, two heads
z_id           = 128-d identity vector     # identity-relevant variation
z_med          = spatial bottleneck map    # anatomical / pathology structure

SELF reconstruction:
    x_self = G(z_id, z_med)

ANONYMIZED (donor-swapped) reconstruction:
    x_anon = G(z_id_donor, z_med)
    where donor is independent of the source patient (see §5)
```

**Reconstruction loss acts ONLY on `x_self`:**

```
L_rec = || x_self - x ||_1
```

**Explicit rule:** `x_anon` is **never** directly forced to reproduce source pixels. `L_rec` is defined only over the SELF branch. The donor-swapped output is constrained by task/anatomy/privacy terms only (`L_path`, `L_anat`, `L_zmed_adv`). Rationale: if `L_rec` were applied to `x_anon`, the only way the model could minimize it while inserting a donor identity would be to keep identity in `z_med` (defeating the bottleneck) or to ignore the donor `z_id` entirely (making the swap vacuous). Restricting `L_rec` to the SELF branch is what makes `z_id` the identity channel and `z_med` the content channel, and what makes donor swapping an identity operation rather than a content operation.

---

## # 3. z_id identity objective

**Mechanism (not mutual information):** `z_id` is *organized to carry identity-relevant variation* through a pairwise verification objective. No claim of information-theoretic maximality is made.

For a pair `(x1, x2)` with identity labels `y_id ∈ {same, different}`:

```
z_id1 = E_id(x1)
z_id2 = E_id(x2)
```

**Pairwise contrastive/verification objective (identity organization on z_id):**

Use a learned similarity head `V(z_id1, z_id2) → s ∈ [0,1]` (patient-verification style, mirroring the existing `SiameseNetwork` head) trained with BCE:

```
L_zid_pair = - y * log V(z_id1, z_id2) - (1-y) * log(1 - V(z_id1, z_id2))
```

where `y=1` for same-patient pairs and `y=0` for different-patient pairs. This makes same-patient `z_id` similar and different-patient `z_id` separated, via the shared encoder `E_id` trunk. The verification head is the identity *organization* probe; it shares parameters with the encoder so that identity-relevant variation is pushed into `z_id`.

*(Alternative if a metric head is preferred: cosine margin / InfoNCE-style on normalized `z_id`, but the BCE verification head is chosen because it reuses the existing patient-verification formulation and is directly reportable as a verification AUC — consistent with the project's verification-metric convention.)*

---

## # 4. z_med identity-suppression objective

**Mechanism (not zero-entropy claim):** a small pairwise identity discriminator is trained to detect patient identity from `z_med`, and `E` is trained adversarially so that identity verification from `z_med` degrades toward chance.

```
H_med(z_med1, z_med2) → p̂(same|z_med1, z_med2)
```

**Training protocol:**

1. Train `H_med` normally to predict same/different patient from the `z_med` representations of a pair (BCE against the same pair labels as §3):
   ```
   L_Hmed = BCE(H_med(z_med1, z_med2), y)
   ```
2. Train `E` adversarially through **gradient reversal** (GRL) on the `z_med` path so that the encoder produces `z_med` from which `H_med` cannot recover identity:
   ```
   L_zmed_adv = BCE(H_med(GRL(z_med1), GRL(z_med2)), y)     # -λ scaling via GRL
   ```
   The gradient-reversal layer flips the sign of the gradient that flows into `E` for this term, making `E` maximize `H_med`'s error (confusion) while `H_med` itself is always trained normally.

**Defensible claim (locked):**

> pairwise patient identity is adversarially suppressed in `z_med`.

**Prohibited claim:** `I(z_med; patient) = 0` — no such assertion is made.

---

## # 5. Donor sampling protocol

**Exact donor selection rules:**

- **Donor patient ≠ source patient** (hard constraint).
- **Training:** donors come from the **TRAIN patient set only**.
- **Validation:** donors obey the **patient-disjoint protocol** — the validation donor patient set is disjoint from the source patient set within the validation split and never draws from train/test patients.
- **Development:** **no TEST donor pool** — test patients are never touched during development (mirrors the frozen `test_touched=false` discipline).
- **Determinism:** deterministic donor mapping given a fixed seed — donor index derived from a seeded RNG over the allowed donor-patient list; reproducible across runs.
- **Anti-leakage:** a donor is rejected if it would leak pathology through the swap; pathology/anatomy in `x_anon` comes from the **SOURCE image's `z_med`** (the donor contributes `z_id` only).

**Donor-pathology-leakage diagnostic (DONOR-L):** for each validation anon image, measure the class-conditional pathology agreement between (a) the source image's pathology label and (b) the donor image's pathology label as predicted/transferred. Concretely:

- Run the frozen classifier on `x_anon` and on `x_donor`; compute per-label whether the anon prediction correlates with the donor's true label (e.g., `corr(σ(cls(x_anon))_ℓ, y_donor,ℓ)`) across a donor-balanced sample.
- Also compute the pairwise image difference `||x_anon − x_self||_1` (must be non-negligible, i.e., the swap changes the image) and the difference `||x_anon − x_donor||_1` (must be substantial — anon must NOT simply be a copy of the donor).
- **Gate:** gross donor pathology transfer is defined as mean absolute donor-label correlation on anon predictions exceeding a predeclared tolerance (see §8 mechanism gate); if it does, the donor sampler / decomposition is defective and S1 does not pass.

**Identity-only intent:** donor `z_id` is the only donor contribution to `x_anon`; donor anatomy/pathology must not transfer because `z_med` comes from the source.

---

## # 6. S1 objective

**S1 stays minimal — no image-space identity adversary yet.**

| Branch | Term | Purpose |
| ------ | ---- | ------- |
| SELF | `L_rec = ‖x_self − x‖₁` | reconstruct the source, forcing the (z_id, z_med) pair to carry enough to rebuild the image |
| ANON | `L_path(x_anon, source pathology label)` | keep diagnostic pathology utility in the anonymized output |
| ANON | `L_anat(x_anon, segmentation-teacher(x_source))` | keep anatomical structure (lungs/heart) in the anonymized output |
| REPR | `L_zid_pair` (§3) | organize identity-relevant variation into `z_id` |
| REPR | `L_zmed_adv` (§4) | adversarially suppress pairwise patient identity in `z_med` |

**Full S1:**

```
L_S1 = λ_rec·L_rec + λ_path·L_path + λ_anat·L_anat + λ_zid·L_zid_pair + λ_adv·L_zmed_adv
```

**Explicit: NO `L_id` on `x_anon` in S1** — the donor identity is inserted through `z_id`, and the privacy objective on the output image is deferred to the (blocked) image-space identity adversary in a later stage.

**Conservative initial coefficients** (loss-scale normalization based, not tuned):

| Coefficient | Initial value | Basis |
| ----------- | ------------- | ----- |
| `λ_rec` | 1.0 | anchor scale (image-domain L1) |
| `λ_path` | 1.0 | BCE on 14 labels, same magnitude domain as verification BCE |
| `λ_anat` | 1.0 | BCE / Dice on 3 structures |
| `λ_zid` | 1.0 | pair BCE, matched scale |
| `λ_adv` | 1.0 | GRL λ=1.0 at start; GRL lambda is the only knob that may scale |

One conservative set, no invented multi-coefficient sweep. **Only TWO substantive follow-up variants are permitted** (locked): (a) rebalance `λ_path/λ_anat` jointly to recover utility if a development gate fails; (b) raise the GRL λ if the z_med probe stays above chance. No other coefficient variants.

---

## # 7. Validation references (frozen, reconstructed from committed evidence)

Official TEST utility numbers are **not** the direct screening reference. Reconstructed VALIDATION references:

| Quantity | Expected | Reconstructed exact | Source artifact | Source SHA-256 |
| -------- | -------- | ------------------- | --------------- | -------------- |
| Adaptive Re-ID reference mean (VAL) | ≈ 0.8382 | **0.8382** | `research_agent/03D_artifacts/seed_{0..9}/training_diagnostics.json` (`best_validation_auc`, n=10) | per-seed files; aggregated mean 0.8382 / SD(ddof=1) 0.0363 (recomputed in STEP 4D E1) |
| Classification mean AUC14 (VAL) | ≈ 0.7938 | **0.7938** | `research_agent/05A_artifacts/utility_by_band.json` → `bands.original.classification.mean_auc_14` = 0.793827 | `8cf42376e9c8c3d2c4292c0d7ab058a60e537c447f07b593f68b1c08b44f1c10` |
| Segmentation Dice (VAL) | ≈ 0.9550 | **0.9550** | `research_agent/05A_artifacts/utility_by_band.json` → `bands.original.segmentation.mean_dice` = 0.955019 | same as above |

Notes:
- The adaptive Re-ID validation reference is the per-seed best-validation-AUC mean from the STEP 3D attacker runs (same quantity the 4A/4B/4D mechanism diagnostics compared against), not the TEST 0.739.
- The classification/segmentation VALIDATION references come from `utility_by_band.json` (`original` band = unmodified corrected-baseline deformed images, VALIDATION n=10816 cases), consistent with the mechanism-diagnostics convention. (TEST utility: class 0.781227 / Dice 0.942717 from `03H_*`, recorded for context, NOT the screening reference.)

---

## # 8. Development gates — 1-seed S1 sanity

**Privacy gate (adaptive validation Re-ID):**
- `mean adaptive validation Re-ID ≤ 0.75`

**Utility gates (vs VALIDATION reference):**
- classification drop vs 0.7938 `≤ 0.03` → class AUC14 `≥ 0.7638`
- Dice drop vs 0.9550 `≤ 0.025` → Dice `≥ 0.9300`

**Mechanism gates (all must pass):**
- `z_med` pairwise identity probe materially weakened (probe on z_med clearly below the probe on z_id; target: probe error substantially above the reference verification error, i.e., suppression demonstrably active)
- `z_id` pairwise verifier materially above chance (AUC substantially > 0.5)
- donor swap produces non-identical images (`‖x_anon − x_self‖₁` non-negligible; anon ≠ donor copy)
- no gross pathology transfer from donor (DONOR-L diagnostic within tolerance)

---

## # 9. Development gates — 3-seed promotion to V2

**Point-estimate screening only — NOT confirmatory CI inference.**

- **Strong promotion:** `mean adaptive validation Re-ID ≤ 0.65`
- **Priority target:** `≤ 0.635`
- **Explicitly NOT required:** a 3-seed 95% CI upper bound `≤ 0.635`. No confirmatory CI gate at this stage.
- **Also required:**
  - no `NUMERICALLY_INVALID` runs
  - validation classification / Dice within their development gates (§8)
  - mechanism diagnostics pass (§8)
- **Decision rule:** if `mean > 0.689`, S1 does not have sufficient privacy leverage — **do not promote directly to confirmatory testing** (STOP and report; only the two permitted follow-up variants may be applied before re-testing).

Official TEST success criteria remain unchanged and are **not** used as development gates.

---

## # 10. Upstream checkpoint-selection re-score

**Zero-training verification performed** using the stored per-epoch validation histories already committed in `research_agent/03D_artifacts/seed_{0..9}/training_diagnostics.json` (`validation_auc_per_epoch`, `validation_loss_per_epoch`). No new attacker training. **No checkpoint files needed to be re-loaded** — the selection rule operates on recorded per-epoch validation curves.

Question quantified: *how much can the checkpoint-selection convention explain?*

| Selection convention | Validation-AUC at selected epoch, mean (n=10) |
| -------------------- | --------------------------------------------- |
| Reported / mechanism reference (best validation AUC) | 0.8382 |
| Upstream-style validation-loss selection (lowest validation loss) | 0.8360 |
| **Δ (convention)** | **+0.0022** |

Per-seed epoch agreement: 4/10 seeds select the same epoch under both rules; the remaining 6 differ by at most a few epochs with per-seed Δ ∈ {0.0000, 0.0012, 0.0017, 0.0031, 0.0073, 0.0076}.

**Interpretation:** checkpoint-selection convention explains a mean difference of only **~0.0022 AUC** — negligible relative to both the material-effect threshold (0.10) and the reference spread (SD 0.0363). Convention does **not** explain the observed privacy level. This is a zero-training diagnostic; it is not a privacy estimate.

**Deferred (explicitly not run now):** the 3-seed legacy-operator attacker arm is scientifically useful but secondary to the positive-method objective and must not delay IBR.

---

## # 11. Exact implementation specification (S1)

Prefer existing modules/backbones. Single shared encoder trunk with two heads.

| Component | Specification |
| --------- | ------------- |
| Encoder trunk `E` | U-Net-style encoder (brain-segmentation skeleton, same family as `UNet_PriCheXyNet`/`UNetSeg`), init_features=32, 256×256×1 input → bottleneck 16×16×512. Existing pattern reused. |
| `z_id` head | `AdaptiveAvgPool2d` on bottleneck → `Linear(512 → 128)` → identity vector dim **128**. |
| `z_med` | the **spatial bottleneck map** (16×16×512) — retained as the content channel; identity suppression applied via `H_med` on this map. |
| Decoder `G` | U-Net decoder (symmetric to trunk, skip connections), output 256×256×1, `tanh` head (matches pipeline image range). `z_id` injected at the bottleneck (`z_med + Linear_up(z_id)` broadcast) so the SELF branch reconstructs the source. |
| Pairwise `z_id` head `V` | verification head: `MLP(256 → 128 → 1)` on `[z_id1; z_id2]`, output sigmoid probability of same patient (mirrors existing `SiameseNetwork.fc_end` head style). Trained with BCE. |
| Pairwise `z_med` adversary `H_med` | `Conv1x1` stack → GAP → `Linear` → 1 logit on `[z_med1; z_med2]` (or pooled concat); trained normally; encoder path uses **GRL**. |
| Donor sampler | seeded index over TRAIN donor-patient list (deterministic per seed); validation donor set patient-disjoint; no TEST donors. Hard `donor != source`. |
| Frozen classifier | DenseNet-121 14-label, `networks/pretrained_classifier.pth`, SHA `8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663`, eval/no-grad — drives `L_path`. |
| Frozen segmentation teacher | `UNetSeg(1,3,16)` = `archive/train_seg_unet/best.pth`, SHA `2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0`, eval/no-grad — drives `L_anat` (3 structures). |

Optimizer: Adam lr=1e-4 for E/G/V; Adam lr=1e-4 for `H_med` (separate optimizer so GRL handles the sign flip cleanly). Batch size and memory below.

---

## # 12. Compute estimate (RTX 5070 Ti 16 GB)

Measured with a GPU forward+backward memory probe of the exact S1 graph (shared encoder, skip-connection decoder, z_id injection, `V`, `H_med`+GRL, plus frozen DenseNet-121 and UNetSeg forward passes on `x_self` and `x_anon`). No optimizer, no epochs — pure memory measurement.

| Batch size | Trainable params | Peak GPU alloc | Free during run |
| ---------- | ---------------- | -------------- | --------------- |
| **16** | 8.12 M (E+G+V+H_med) | **6.78 GB** | 8.85 GB |
| 8 | 8.12 M | 4.49 GB | 8.85 GB |

Frozen auxiliaries: DenseNet-121 ≈ 7.0 M params, UNetSeg ≈ 1.9 M params (eval only).

**Decision:** `bs = 16` is safely feasible (peak 6.78 GB ≈ 42% of 16 GB, ~8.9 GB headroom). **Use bs=16.** No gradient accumulation is used and none is silently introduced; if bs=16 ever becomes infeasible, stop and document rather than silently accumulate.

---

## # 13. Blocking issues

1. **Missing source documents** — `08_PHASE2_POSITIVE_METHOD_DESIGN.md`, `08A_ORIGINAL_PAPER_CODE_RECONCILIATION.md`, `04_METHOD_SUCCESS_CRITERIA.md`, `04B_SUCCESS_CRITERIA_AMENDMENT.md` are absent from the repo. This lock was written self-contained from the STEP 6A prompt and frozen committed evidence. **Not a code/training blocker**; flag for the owner to confirm the paper-side design file is synchronized with this lock.
2. **GRL implementation** — no GRL module exists in the current codebase; must be added (small, standard `λ`-scaled autograd Function). Not a blocker.
3. **Pathology-label availability for S1 training** — `L_path` requires source pathology labels on TRAIN pairs; the frozen classifier + NIH labels exist, so this is available. Confirm label coverage for the pair files. Not a blocker.
4. **Donor-patient list** — requires an explicit patient→identity map for TRAIN/VALIDATION splits with deterministic seeding; the existing pair files / NIH labels provide the identity structure. Implementation detail.
5. **No blocking scientific issue found.** The segmentation teacher is verified, the graph is corrected, the bottleneck is made identifiable via pairwise objectives, and the compute estimate is feasible at bs=16.

---

## # 14. Final implementation decision

- Segmentation teacher: **RESTORED and VERIFIED** (exact STEP 3C artifact, hash match `2dfdcf9b…`, loadable, strict-load OK).
- IBR graph: **corrected** — `L_rec` only on `x_self`; `x_anon` governed by task/anatomy/privacy terms only.
- Bottleneck: **identifiable** — `z_id` via pairwise verification BCE; `z_med` via GRL-adversarial pairwise identity suppression; claim limited to *adversarial suppression*, not `I(z_med;patient)=0`.
- Donor protocol: **defined** — train-only donors, patient-disjoint validation donors, no TEST donors, deterministic per seed, donor pathology-leakage diagnostic specified.
- S1 objective: **minimal and fixed** — five terms, one conservative coefficient set, exactly two permitted follow-up variants.
- Development gates: **locked** against reconstructed VALIDATION references (0.8382 / 0.7938 / 0.9550) with 1-seed sanity and 3-seed point-estimate promotion; no confirmatory CI gate; STOP rule at mean > 0.689.
- Re-score: **performed (zero training)** — checkpoint-selection convention explains only ~0.0022 AUC.
- Implementation spec and compute estimate: **complete**; bs=16 feasible on RTX 5070 Ti 16 GB.

**Decision: proceed to implement IBR S1 as specified in §11, with the S1 objective of §6 and the gates of §8/§9.**

---

IBR S1 IMPLEMENTATION: READY