# M1.2 — Development Evaluator Isolation + Threat-Model Lock Report

**Date**: 2026-08-15  
**Protocol Version**: `1.2.0`  
**Branch**: `research/method-restart`  
**Base Commit**: `e53b19208c7de2800b410a3554e73ee5796d4c0a` (M1.1)  
**Test Suite Status**: 40/40 checks **PASS** (T1–T15, M0.1, M1, M1.1, T30–T40)  
**TEST Benchmark Status**: **CLOSED / UNTOUCHED** (Zero test evaluations run)  

---

## 1. Executive Summary & Audit Findings

An independent code audit identified that upstream evaluator classes and scripts directly violate development isolation by reading or hard-coding the closed official TEST benchmark:

1. **P0-A — SNN Agent constructs TEST loader**:
   `agents/AgentSiameseNetwork.py` constructs `self.test_loader = utils.get_data_loader(phase='testing', ...)` unconditionally in `__init__()`, and its `run()` method executes `training_validation()` followed immediately by `testing_evaluation()`. Constructing or calling `AgentSiameseNetwork.run()` during development would breach test isolation.
2. **P0-B — Classification Evaluator hard-codes TEST fold**:
   `eval_classifier.py` calls `chexnet.eval_model.make_pred_multilabel()`, which unconditionally instantiates `CXRDataset(..., fold="test")`. Using this function directly during development touches the test split.
3. **P0-C — SNN Training vs. Attack Geometry Ambiguity**:
   Upstream `train_snn()` and `validate_snn()` evaluate on `anon(x1), anon(x2)`, whereas `test_snn()` evaluates under the realistic linkage scenario `anon(x1), real(x2)`. This distinction between attacker early-stopping selection and the scientific privacy metric must be explicitly disambiguated and locked before M2 development.

M1.2 resolves these issues without modifying historical upstream code by introducing dedicated, fail-closed TRAIN/VAL development modules in `research_agent/m2_dev/`.

---

## 2. Threat-Model & Evaluator Geometry Locks

### 2.1 Attacker Training Geometry (`anon_anon`)
- **Split**: TRAIN pairs only.
- **Inputs**: `anon(image1), anon(image2)` using the frozen legacy operator (`mu=0.01`, Gaussian kernel 9, sigma 2, border padding, align corners).
- **Generator State**: Frozen, in `eval()` mode, `requires_grad = False` (no generator updates during attacker training).
- **Network**: Fresh ImageNet-initialized Siamese Network (`SiameseNetwork`).
- **Loss**: `nn.BCEWithLogitsLoss`.

### 2.2 Attacker Checkpoint Selection Geometry (`anon_anon`)
- **Split**: VALIDATION pairs only.
- **Inputs**: `anon(image1), anon(image2)`.
- **Metric**: Lowest validation BCE loss.
- **Purpose**: Exclusively for attacker checkpoint selection and early stopping. This is **NOT** the scientific privacy metric.

### 2.3 Scientific Privacy Validation Geometry (`anon_real`)
- **Split**: VALIDATION pairs only (using the fixed validation pair file).
- **Inputs**: `anon(image1), real(image2)` (the second image is untouched by the generator).
- **Evaluator**: Frozen best attacker checkpoint selected from 2.2.
- **Metric**: ROC-AUC computed across all validation pairs (`AUC_Bdev_VAL` and `AUC_C4_VAL`).
- **Function**: `research_agent/m2_dev/eval_reid_val.py::evaluate_reid_val_mixed()`.

### 2.4 Classification Utility Evaluation Geometry (`VAL-only`)
- **Split**: `fold = "val"` only. Any request for `test`/`testing`/`final_test` fails-closed before dataset initialization.
- **Model**: Released frozen DenseNet-121 classifier (`networks/pretrained_classifier.pth`, SHA: `8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663`). Both B_dev and C4 use the identical frozen evaluator checkpoint (not arm critics).
- **Preprocessing**: 1-ch expanded to 3-ch, resized to `(224, 224)`, ImageNet-normalized.
- **Metric**: Mean ROC-AUC over 14 pathologies (`macro_AUC_Bdev_VAL`, `macro_AUC_C4_VAL`).
- **Function**: `research_agent/m2_dev/eval_classifier_val.py::evaluate_classification_val()`.

---

## 3. Development Infrastructure Architecture (`research_agent/m2_dev/`)

```
research_agent/m2_dev/
├── __init__.py
├── evaluator_common.py       # Shared legacy operator, preprocessing, checkpoint selection, order fingerprint
├── dev_attacker.py           # TRAIN/VAL-only DevAttacker with configurable seed & firewall
├── eval_reid_val.py          # Scientific anon/real validation privacy evaluator
└── eval_classifier_val.py    # VAL-only DenseNet-121 classifier utility evaluator
```

### Key Architectural Enforcements:
1. **Configurable Attacker Seeds**: `DevAttacker` takes `attacker_seed` and invokes `utils.seed_all(attacker_seed)` before model instantiation, DataLoader shuffling, and optimizer creation.
2. **Method-Neutral Checkpoint Selection**:
   Validation loss per epoch is broken into `ac_bce`, `privacy_term`, `feature_term`.
   - B_dev: `optimization_total = ac_bce + privacy_term`, `selection_total = ac_bce + privacy_term`
   - C4: `optimization_total = ac_bce + privacy_term + feature_term`, `selection_total = ac_bce + privacy_term`
   - Generator checkpoint selection for **both** arms selects minimum `selection_total` (tie-break earliest epoch), saved as `generator_best_method_neutral.pth`.
3. **Paired Data-Order Fingerprint**:
   `train_order_fingerprint()` computes a deterministic SHA256 of the epoch-0 DataLoader pair ordering using `torch.Generator().manual_seed(seed)`. B_dev and C4 must produce matching fingerprints before results are interpreted.
4. **TEST Firewall Integration**:
   `TestFirewall.check('dev')` is integrated directly into loader builders and dataset initializers. Any attempt to pass `test`, `testing`, or `final_test` immediately raises a `RuntimeError`.

---

## 4. Regression Test Suite Verification (T30–T40)

All 11 newly introduced tests passed on CPU using synthetic tensors, mock datasets, and static inspection:

| Test | Description | Verification Status |
| :--- | :--- | :--- |
| **T30** | `DevAttacker` creates NO `test_loader` and guarded loader builder rejects `testing` phase | **PASS** |
| **T31** | `evaluate_classification_val` rejects `fold='test'` before dataset construction | **PASS** |
| **T32** | Attacker TRAIN geometry passes `anon(x1), anon(x2)` (2 anonymize calls/batch) | **PASS** |
| **T33** | Attacker selection VAL geometry passes `anon(x1), anon(x2)` (2 anonymize calls/batch) | **PASS** |
| **T34** | Scientific VAL privacy geometry passes `anon(x1), real(x2)` (1 call/batch; x2 unchanged) | **PASS** |
| **T35** | Attacker seeds: seed 42 reproducible, seed 42 != seed 43 initialization stream | **PASS** |
| **T36** | Classification evaluator allows `fold='val'`, strictly rejects `test`/`testing`/`final_test` | **PASS** |
| **T37** | Both B_dev and C4 reference identical frozen classifier SHA (`8ad15b38...`) | **PASS** |
| **T38** | Method-neutral checkpoint rule selects on `selection_total` excluding feature loss | **PASS** |
| **T39** | Paired anonymizer data-order fingerprint is deterministic and seed-sensitive | **PASS** |
| **T40** | TEST firewall integrated; static AST/regex audit confirms dev runners contain no test builds | **PASS** |

### Complete Test Suite Execution:
```text
PASS     T1 step-per-batch==1  
PASS     T2 accum==summed (N=2,4)  
PASS     T2b accum1==full-batch  
PASS     T3 batch16+acc4 != batch64 (BN)  
PASS     T4 pristine deepcopy tracks current  
PASS     T4 main-style build-once is STALE  
PASS     T4 repaired refresh tracks current  
PASS     T4e repaired == pristine upstream  
PASS     T5 pos_weight=None == baseline  
PASS     T5 feature_loss_weight=0 == baseline  
PASS     T6 C4 detached source (gen-safe)  
PASS     T6b undetached source gets grad  
PASS     T7 verifier label 1.0=same  
PASS     T7b verifier loss is BCE  
PASS     T8 dev config disables C3  
PASS     T9 C2 zero-budget == baseline  
PASS     T10 budget mean == mu  
PASS     T10b zero channel => uniform  
PASS     T11 audit: legacy==upstream, legacy!=corrected  
PASS     T12 corrected mu=0 => identity  
PASS     T12 legacy mu=0 NOT identity  
PASS     T13 dev modes pass firewall  
PASS     T13 test mode raises unless allowed  
PASS     T14 provenance deterministic  
PASS     T14 sha256 stable  
PASS     T15 no frozen ckpt modified  
PASS     M0.1 all config hashes match checkpoints  
PASS     M1 B_dev/C4 configs paired & frozen  
PASS     M1.1 S1/S2 gates, seeds, cost, segmentation frozen  
PASS     T30 attacker dev constructs no TEST loader  
PASS     T31 classifier dev rejects TEST before dataset init  
PASS     T32 attacker TRAIN geometry anon/anon  
PASS     T33 attacker selection VAL geometry anon/anon  
PASS     T34 scientific VAL privacy geometry anon/real  
PASS     T35 configurable attacker seeds  
PASS     T36 classification VAL fold val/test  
PASS     T37 common classification checkpoint SHA  
PASS     T38 method-neutral checkpoint rule  
PASS     T39 paired order fingerprint deterministic  
PASS     T40 TEST firewall integrated  
============================================================
M0 SUITE: ALL PASS (40/40)
```

---

## 5. Protocol Lock Update

`research_agent/M1_C4_PROTOCOL_LOCK.json` updated to version `1.2.0` with explicit lock fields:
- `attacker_train_geometry`: `"anon_anon"`
- `attacker_checkpoint_validation_geometry`: `"anon_anon"`
- `scientific_privacy_validation_geometry`: `"anon_real"`
- `classification_development_fold`: `"val"`
- `development_test_loader_allowed`: `false`
- `development_classifier_test_fold_allowed`: `false`

All scientific promotion gates and thresholds from M1.1 remain strictly frozen.

---

## 6. M1.2 Verification Checklist

- [x] No TEST loader constructed in dev attacker
- [x] No TEST classification fold constructed
- [x] Attacker TRAIN = `anon/anon`
- [x] Attacker checkpoint VAL = `anon/anon`
- [x] Scientific VAL privacy = `anon/real`
- [x] Classifier utility = VAL only
- [x] Classifier evaluator checkpoint same for both arms
- [x] Attacker seed configurable (`utils.seed_all`)
- [x] Method-neutral anonymizer checkpoint selection implemented
- [x] Paired data-order fingerprint implemented
- [x] TEST firewall integrated and verified
- [x] Full test suite (40/40) PASS
- [x] Official TEST benchmark untouched
- [x] No scientific method training run
- [x] Protocol lock updated to v1.2.0
