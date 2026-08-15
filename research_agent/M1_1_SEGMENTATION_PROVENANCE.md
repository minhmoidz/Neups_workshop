# M1.1 — Segmentation Provenance Recovery Manifest

- **Task (M1.1 §6):** read-only recovery audit for `archive/train_seg_unet/best.pth`.
- **Determination:** **SEGMENTATION_STILL_BLOCKED** (with substantially recovered provenance).
- **No TEST data used, no segmentation run on TEST, no segmenter trained, no metric computed in this step.**

---

## 1. What was recovered (all read-only)

### 1.1 Checkpoint (authoritative bytes)

- Path: `archive/train_seg_unet/best.pth`
- SHA-256: `2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0`
- Self-recorded metadata (read from the checkpoint dict):
  - `epoch = 20`
  - `mean_dice = 0.9548383020407831`
  - `init_features = 16`
  - `dice = [0.9544278977826858, 0.9639166558161378, 0.9461703525235257]` (Left Lung, Right Lung, Heart)
  - `iou = [0.9202553412566582, 0.9360721794422716, 0.9025846011160562]`
  - 118 tensors; `encoder1.enc1conv1.weight [16,1,3,3]`, `conv.bias [3]` → UNet 1→3 channels, init_features 16.

### 1.2 Source recovered from git history (commit `9eaa5fd` + successors)

- `networks/UNetSeg.py` — **architecture**: UNetSeg(in=1, out=3, init_features=16 default **32** in code; checkpoint used 16), sigmoid output, brain-segmentation skeleton.
- `chexnet/seg_dataset.py` — **dataset class**: SegDataset(fold∈{train,val,test}), NIH official split from `chexnet/nih_labels.csv` fold column, restricted to rows with CheXmask masks.
- `utils/segmask.py` — **mask loader**: CheXmask RLE decode from `data/chexmask/ChestX-Ray8.csv`, 3 structures (Left Lung, Right Lung, Heart), 256×256 nearest.
- `train_seg.py` — **training script**: train on `fold==train` (subsample 0 = full), validate on `fold==val` subsample=1500; Adam lr=3e-4 default, batch 16 default, BCEWithLogits + soft-Dice (0.5/0.5); **checkpoint save rule: best mean val dice → `best.pth`** (`if mean_dice > best_dice`).
- `eval_seg.py` — **metric code**: Dice `(2·|G∩P|+1e-7)/(|G|+|P|+1e-7)`, IoU `(|G∩P|+1e-7)/(|G∪P|+1e-7)` at threshold 0.5; HD95 = two-sided 95th-pct Hausdorff (scipy EDT), pixel units @256; empty-mask → NaN handling.

### 1.3 Documentation chain (non-TEST, from git history)

- `HANDOVER/GOALS.md:87` — `archive/train_seg_unet/best.pth`, "val dice 0.955/0.964/0.946 trên 1500 ảnh" (matches checkpoint `dice` exactly, rounded).
- `RESEARCH_BRIEF.md:139-140` — T7: CheXmask RLE, U-Net feat=16, scripts `train_seg.py`/`eval_seg.py`, model at `archive/train_seg_unet/best.pth` val Dice 0.955/0.964/0.946.
- `research_agent/03H_CORRECTED_BASELINE_UTILITY.md` §7–8 — architecture, preprocessing (grayscale, 256 bilinear, `(x/255-0.5)/0.5`), split (`fold==test` for the 03H evaluation; n=25,596), training on **raw** images `fold==train`, checkpoint SHA `2dfdcf9b…`, metric definitions verbatim.
- Commit `7e762de` message — "Segmentation Dice 0.9427 / IoU 0.8955 / HD95 2.006" from a **TEST-fold** evaluation (03H).

### 1.4 Environment / data fingerprints

- `chexnet/nih_labels.csv` SHA-256 `80324996867e73546bd7a09025df4a4cc3243fc00663b753023ccd90a9b5f8b9` (split definition).
- `data/chexmask/ChestX-Ray8.csv` SHA-256 `48766ab0268235d63666bb2bacbd9f642b33fce7c1be40b9e1ecb381605545fa` (mask metadata).

---

## 2. Why STILL BLOCKED (exactly what is missing)

Per M1.1 §6 certification requires ALL of: training data provenance, validation data
provenance, no TEST contamination, checkpoint-selection semantics, preprocessing, model
architecture, Dice/IoU/HD95 implementation, checkpoint SHA. Most are established, but:

1. **The actual training log for the checkpoint run is NOT preserved.** The only segmenter
   log in the repo (`logs/train_seg.log`) is a **different, broken run** (25 epochs, best
   mean val dice ≈ 0.0604) that does NOT match the checkpoint (mean_dice 0.9548 @ epoch 20).
   Without the checkpoint run's own epoch-by-epoch log we cannot demonstrate the training
   trajectory that produced `best.pth`.
2. **Exact training configuration/CLI for the checkpoint run is not reconstructable.** The
   checkpoint records only `init_features=16` (which differs from the code default of 32),
   `epoch`, `mean_dice`, `dice`, `iou`. It does not record lr, batch size, seed, or total
   epochs. No config file, shell script, or commit records the exact `train_seg.py`
   invocation used for the checkpoint run. (Recoverable defaults: lr 3e-4, batch 16,
   seed 42, val subsample 1500, train subsample 0.)
3. **TRAIN/VAL patient lists are defined by `chexnet/nih_labels.csv` fold column but the
   exact train/val image counts actually used for the checkpoint run are not verifiable**
   (no matching log, no saved split snapshot for that run).

Historical TEST results (e.g., 03H fold==test Dice 0.9427) are explicitly NOT certification
per M1.1 §6 ("Do NOT treat historical TEST results as certification") and are NOT used here.

---

## 3. Required to certify (for a future M-step)

- (a) Re-commit `train_seg.py`, `chexnet/seg_dataset.py`, `utils/segmask.py`,
  `networks/UNetSeg.py`, `eval_seg.py` onto the working branch as the canonical evaluator
  source; **or** (b) re-run `train_seg.py` under the canonical protocol (fold==train /
  fold==val subsample 1500, raw images, init_features=16) with a preserved config + log +
  split snapshot, producing a new certified `best.pth` with SHA.
- Record lr/batch/seed/epochs and the exact CLI/config in the branch.
- Save the full training log + validation dice curve.
- Then re-certify the evaluator (new checkpoint SHA) before it is used in any utility gate.

## 4. Consequence for M1.1 protocol

- Segmentation gate: **NOT silently treated as PASS**. The M1.1 protocol records the
  segmentation criterion as REQUIRED-IF-CERTIFIED (frozen thresholds `Dice_C4 >= Dice_Bdev
  - 0.005` and no material HD95 degradation) and as `NOT APPLICABLE` while the evaluator is
  BLOCKED. This does not block the privacy/classification protocol.