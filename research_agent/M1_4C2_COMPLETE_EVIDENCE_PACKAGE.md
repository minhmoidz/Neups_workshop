# M1.4c.2 COMPLETE EVIDENCE PACKAGE
## Hostile Read-Only Audit — Classification Replay, Provenance, Execution Integrity, Promotion, and Certification

**Audit branch:** `audit/m2-final-certification`  
**Exact audit commit:** `512305fbb855f61079316ae9b934ba2b9a74e98b`  
**Canonical branch:** `research/method-restart`  
**Canonical commit:** `c6431310061c04e54dce82d30ae6e0ce24440562`  
**Decision:** **C) BLOCKED — SMALL CERTIFICATION/PROMOTION CLOSEOUT REQUIRED**

---

## 0. Audit state and evidence boundary

The exact audit state was verified before this report:

- `HEAD` is exactly `512305fbb855f61079316ae9b934ba2b9a74e98b`.
- The canonical commit is an ancestor of the audit commit.
- The working tree is clean.
- No source files were edited during the forensic review.
- No commits, pushes, merges, rebases, cherry-picks, resets, or branch changes were performed.
- No training was run.
- No full M2-S1 execution was launched.
- No S1 attacker training was run.
- No S2 execution was run.
- No TEST data was opened.
- `image_pairs_testing_5000.txt` was not opened.
- No `fold='test'`, testing loader, or `final_test` loader was constructed.

The reproducer descriptions below are static/source-level reproductions and were not executed. This preserves the requested read-only and no-training boundary.

For this report, `VALID` means either:

```python
check_run_validity(...) == (True, "VALID")
```

or a generated result containing:

```json
"run_status": "VALID"
```

`VALID` is therefore a software status, not proof that the scientific result is trustworthy.

---

# 1. Executive conclusion

The core B_dev/C4 scientific semantics are substantially consistent with the frozen method under the intended canonical orchestration path:

- the C4 online teacher is refreshed from the current classifier critic;
- source/reference features are detached;
- the legacy deformation operator is preserved;
- generator, verifier, and classifier optimizers are separated;
- critic updates use detached fake images;
- method-neutral checkpoint selection excludes the C4 feature term;
- earliest-epoch tie-breaking is implemented;
- the expected anonymizer and attacker hyperparameters are represented in the frozen configuration.

However, the audited branch is not yet safe for clean canonical promotion because certification and production-facing integrity boundaries remain fail-open. The most important defects are:

1. direct production APIs accept injected models/loaders without frozen provenance;
2. direct calls to `run_orchestration()` bypass CLI-only scientific argument enforcement;
3. short privacy and classification artifacts can pass replay when scalar metadata is self-consistent;
4. mutable lock metadata controls expected configuration hashes without immutable lock anchoring;
5. source validation checks ancestry rather than exact certified source identity;
6. runtime order hashes are recorded but not compared with frozen expected hashes;
7. numerical diagnostic errors can be swallowed while the manifest still reports PASS;
8. T215 can pass after Git failure or assertion failure;
9. TEST-firewall claims exceed the actual implementation;
10. the final forensic report overclaims guarantees and lacks the requested §26;
11. the claimed 216/216 inventory is not bound to the exact final test source.

The correct disposition is therefore:

```text
C) BLOCKED — SMALL CERTIFICATION/PROMOTION CLOSEOUT REQUIRED
```

This is not a finding that the method itself must be redesigned. It is a finding that the execution, replay, provenance, and promotion boundaries must be closed before canonical promotion or scientific M2 launch.

---

# 2. Consolidated finding table

| Finding | Reachable from official scientific CLI? | Can return VALID? | Can alter scientific result? | Severity | Blocking? |
|---|---|---:|---:|---|---:|
| F01 injected model/loader provenance | No through normal fixed wrappers; yes through direct API | Yes via direct validity/reporting surface | Yes | HIGH | Yes |
| F02 direct scientific API parameters | No; CLI rejects altered values | Yes through direct orchestration | Yes | HIGH | Yes |
| F03 short privacy NPZ | Not from unmodified producer; yes through direct/tampered validity input | Yes | Yes, evidence/artifact | HIGH | Yes |
| F04 short classification CSV | Not from unmodified producer; yes through direct/tampered validity input | Yes | Yes, evidence/artifact | HIGH | Yes |
| F05 malformed result objects | Not normally | No | No | MEDIUM | No |
| F06 unit-test contract bypass | No normal CLI path | Yes, literal development `VALID` | Not an intended real result | MEDIUM | No for launch; yes for evidence labelling |
| F07 incomplete numerical checks | Vulnerable code path is reached; isolated reproducer uses runner seam | Conditionally | Yes | MEDIUM-HIGH | Yes |
| F08 swallowed diagnostics | Yes, C4 path | Yes | Potentially/evidence | MEDIUM-HIGH | Yes |
| F09 TEST/eval_test overclaim | Generic seam yes; normal CLI uses `dev` | Not alone | Combined with loader seams | MEDIUM-HIGH | Yes unless claim narrowed |
| F10 ancestry-only source guard | Yes | Yes | Yes | HIGH | Yes |
| F11 T215 fail-open | No; certification test only | Not scientific VALID; can pass certification | Can alter promotion decision | HIGH | Yes |
| F12 CPU fallback | Yes with `--device cpu` | No; crashes | No | P1 | Yes |
| F13 resume defect | Not normal CLI; direct runner API yes | No normally | Potentially | P2 | No if restart policy is enforced |
| F14 order hashes not enforced | Yes | Yes | Yes | MEDIUM-HIGH | Yes |
| F15 custom-config provenance | Not normal orchestrator; direct runner API yes | Yes | Yes | MEDIUM | Yes for direct scientific API |
| F16 report overclaims/missing §26 | No runtime path | No directly | No direct metric change; false certification | HIGH | Yes for promotion |
| S17 `fold='validation'` alias | No normal CLI; direct evaluator yes | Usually crashes; injected-loader variant can pass | Potentially | MEDIUM | No as separate CLI blocker |
| S18 mutable lock authority | Yes if a clean descendant changes lock/config | Yes if downstream bundle is self-consistent | Yes | MEDIUM-HIGH | Yes |
| S19 unbound 216/216 inventory | No runtime path | No directly | No; can invalidate certification | HIGH | Yes |

---

# 3. Detailed evidence

## F01 — Injected model and loader provenance bypass

**Severity:** HIGH

### Exact locations

#### Classification evaluator

File:

```text
research_agent/m2_dev/eval_classifier_val.py
```

Function:

```text
evaluate_classification_val()
```

Ranges:

- `122-154`: optional injected model and classifier-SHA logic;
- `179-203`: optional injected dataloader and weak length check.

Relevant behavior:

```python
if model is None:
    model = load_frozen_classifier(device)
```

An injected `model` is accepted without proving that it came from the frozen classifier checkpoint.

```python
if config.get('dataloader') is not None:
    dataloader = config['dataloader']
    n_images = len(dataloader.dataset)
```

An injected dataloader is accepted when its reported length is 10,816 for `fold == 'val'`.

#### Privacy evaluator

File:

```text
research_agent/m2_dev/eval_reid_val.py
```

Functions:

```text
evaluate_reid_val()
evaluate_reid_val_mixed()
```

Ranges:

- `25-79`: evaluates whatever loader is supplied;
- `82-147`: accepts `validation_loader` and checks only size 2,000 in non-unit mode.

#### Anonymizer runner

File:

```text
research_agent/m2_dev/anonymizer_runner.py
```

Function:

```text
M2AnonymizerRunner.__init__()
```

Ranges:

- `151-185`: accepts injected classifier and verifier models;
- `203-211`: accepts injected training/validation loaders and an optional sampler.

The constructor documents these seams as being for testing but does not enforce `unit_test_mode=True`.

### Executable call paths

Direct classification API:

```text
evaluate_classification_val(
    config={
        "dataloader": attacker_controlled_loader,
        "unit_test_mode": False
    },
    model=attacker_controlled_model,
    fold="val",
    generator_checkpoint=<valid generator path>
)
```

Direct privacy API:

```text
evaluate_reid_val(
    config={},
    attacker_checkpoint=<valid attacker path>,
    generator_checkpoint=<valid generator path>,
    validation_loader=attacker_controlled_loader,
    unit_test_mode=False
)
```

The loader only needs to report 2,000 samples.

Direct runner API:

```text
M2AnonymizerRunner(
    arm="B_dev",
    config=<valid-looking config>,
    ac_model=attacker_controlled_classifier,
    verification_model=attacker_controlled_verifier,
    training_loader=attacker_controlled_train_loader,
    validation_loader=attacker_controlled_val_loader,
    unit_test_mode=False
)
```

Normal official paths are:

```text
run_orchestration()
→ evaluate_classification_arm()
→ evaluate_classification_val()
```

and:

```text
run_orchestration()
→ evaluate_privacy_arm()
→ evaluate_reid_val()
```

The normal wrappers supply fixed paths and do not normally inject arbitrary models or loaders. The seam nevertheless remains reachable through direct production APIs.

### Minimum adversarial input

A classification loader with:

```text
len(loader.dataset) == 10816
```

but rows drawn from an arbitrary or TEST-derived cohort, together with a model that emits attacker-controlled predictions.

A privacy loader with:

```text
len(loader.dataset) == 2000
```

but arbitrary pair rows or ordering.

### Observed behavior

The evaluator returns predictions and metrics based on injected objects. The on-disk frozen classifier SHA can still pass because the evaluator checks the file, not the origin of the injected in-memory model.

The privacy evaluator checks the count but not:

- validation pair-file SHA;
- semantic pair-row hash;
- patient cohort;
- pair ordering;
- loader source path.

The runner can emit normal-looking manifests without recording proof that injected objects came from frozen artifacts.

### Expected behavior

Scientific mode should:

- construct the frozen model internally;
- construct loaders from frozen files internally;
- verify exact pair/image fingerprints;
- reject injected models/loaders unless explicit unit-test mode is active.

### Applicability

- Real scientific CLI: **not through the normal fixed orchestrator path**.
- Direct production Python API: **yes**.
- Unit-test-only path: **also yes**, but the unguarded seam is available outside unit mode.
- Documentation only: **no**.

### Outcome

- Can return `VALID`: **yes, when injected outputs are passed through public validity/orchestration surfaces**.
- Can produce scientific-looking artifacts: **yes**.
- Can silently alter scientific result: **yes**.
- Can silently drift provenance: **yes**.
- Exception/crash only: **no**.

### Smallest correct fix

Reject injected models, loaders, and samplers unless:

```python
unit_test_mode is True
```

In non-unit scientific mode, construct and verify all objects internally against frozen paths, SHAs, row hashes, and cohort fingerprints.

### Regression test

```text
test_scientific_apis_reject_injected_models_and_loaders()
```

The test should assert that:

- `evaluate_classification_val(..., model=FakeModel, unit_test_mode=False)` raises;
- an injected classification dataloader raises;
- `evaluate_reid_val(..., validation_loader=FakeLoader(2000), unit_test_mode=False)` raises;
- `M2AnonymizerRunner(..., ac_model=FakeModel, training_loader=FakeLoader, unit_test_mode=False)` raises.

Before the fix, these calls are accepted.

---

## F02 — Direct scientific API parameter bypass

**Severity:** HIGH

### Exact locations

File:

```text
research_agent/m2_dev/run_m2_s1.py
```

CLI-only enforcement:

```text
parse_args(): 65-107
```

The CLI rejects:

- non-`all` arm;
- anonymizer epochs other than 250;
- attacker epochs other than 100;
- patience other than 5;
- anonymizer seed other than 42;
- attacker seed other than 42.

Missing reusable-API enforcement:

```text
run_orchestration(): 636-823
```

The function consumes caller-provided values and does not repeat the frozen-value validation.

### Minimum adversarial configuration

```python
args = SimpleNamespace(
    scientific_m2_s1=True,
    arm="all",
    max_epochs=1,
    attacker_epochs=100,
    attacker_patience=5,
    seed=7,
    attacker_seed=42,
    device="cuda",
)
```

The deviations are:

```text
max_epochs = 1
seed = 7
```

### Executable call path

```text
run_orchestration(args, unit_test_mode=False)
→ run_anonymizer_arm(..., max_epochs=1, seed=7)
→ M2AnonymizerRunner.run(max_epochs=1)
→ check_run_validity(..., expected_epochs=args.max_epochs)
```

The validity check therefore expects the altered epoch count rather than the frozen 250.

### Observed behavior

A direct caller can execute a one-epoch, seed-7 scientific-looking run while retaining:

```python
scientific_m2_s1=True
```

The downstream summary path can continue to compute scientific gates because it sees the scientific flag, not the fact that the CLI-only validation was bypassed.

### Expected behavior

All scientific constraints must be enforced at the reusable orchestration boundary regardless of whether the caller arrived through CLI parsing.

### Applicability

- Real scientific CLI: **no; CLI rejects the configuration**.
- Direct production Python API: **yes**.
- Unit-test-only path: **no**.
- Documentation only: **no**.

### Outcome

- Can return `VALID`: **yes**.
- Can produce scientific-looking artifacts: **yes**.
- Can alter scientific result: **yes**.
- Can silently drift provenance: **yes**.
- Exception/crash only: **no**.

### Smallest correct fix

Create one frozen scientific-argument validator and call it from:

- `parse_args()`;
- `run_orchestration()`;
- every public scientific entrypoint.

At minimum enforce:

```text
arm == all
max_epochs == 250
attacker_epochs == 100
attacker_patience == 5
seed == 42
attacker_seed == 42
```

### Regression test

```text
test_run_orchestration_rejects_noncanonical_scientific_args()
```

Call `run_orchestration()` directly with the adversarial namespace and assert `ValueError` or `RuntimeError`.

Before the fix, the call proceeds past the missing validation.

---

## F03 — Short privacy NPZ can pass replay

**Severity:** HIGH

### Exact locations

File:

```text
research_agent/m2_dev/run_m2_s1.py
```

Producer:

```text
evaluate_privacy_arm(): 283-344
```

The producer writes the NPZ and reports `n_pairs`.

Validator:

```text
check_run_validity(): 393-467
```

The declared count is checked at `445-446`, but the replay section at `459-465` does not compare raw array lengths with the declared count.

### Minimum concrete adversarial object

A privacy file containing only four rows:

```python
y_true  = np.array([0, 1, 0, 1])
y_score = np.array([0.1, 0.9, 0.2, 0.8])
```

This has:

```text
ROC-AUC = 1.0
```

Use metadata:

```python
p_res = {
    "roc_auc": 1.0,
    "n_pairs": 2000,
    "generator_checkpoint_sha256": <matching generator SHA>,
    "attacker_checkpoint_sha256": <matching attacker SHA>,
    "predictions_file": <four-row NPZ>,
    "predictions_file_sha256": <actual four-row file SHA>,
}
```

Surround this with otherwise-valid manifests and classification results. In non-unit mode, provide the required valid configuration/classifier SHA fields.

### Executable call path

```text
run_orchestration()
→ evaluate_privacy_arm()
→ np.savez_compressed(...)
→ check_run_validity()
```

The normal producer writes 2,000 rows, but the validator itself is weak when given a shorter self-consistent artifact.

### Observed behavior

The replay computes AUC over four rows, obtains `1.0`, compares it to `p_res['roc_auc'] == 1.0`, and finds no mismatch.

There is no exact check that:

```text
len(y_true) == len(y_score) == 2000
```

### Expected behavior

The artifact must be INVALID because:

```text
4 != 2000
```

and because raw array lengths must match each other and the frozen pair contract.

### Applicability

- Real scientific CLI: **not through an unmodified producer; reachable through direct validity or artifact/metadata tampering**.
- Direct production Python API: **yes**.
- Unit-test-only path: **no; the missing count contract also affects non-unit replay**.
- Documentation only: **no**.

### Outcome

- Can return `VALID`: **yes**.
- Can produce scientific-looking artifacts: **yes**.
- Can silently alter scientific result/evidence: **yes**.
- Can silently drift provenance: **yes**.
- Exception/crash only: **no**.

### Smallest correct fix

Require all of:

```python
y_true.ndim == 1
y_score.ndim == 1
len(y_true) == len(y_score) == 2000
```

Also validate finite values, binary labels, expected pair ordering, and raw artifact provenance.

### Regression test

```text
test_privacy_replay_rejects_short_npz()
```

Create the four-row NPZ above, set metadata `n_pairs=2000`, and assert:

```python
check_run_validity(...) == (False, <reason>)
```

Before the fix, the replay reaches the scalar-AUC comparison and accepts the artifact.

---

## F04 — Short classification CSV can pass replay

**Severity:** HIGH

### Exact locations

File:

```text
research_agent/m2_dev/run_m2_s1.py
```

Function:

```text
check_run_validity()
```

Ranges:

- `470-480`: checks scalar `c_res['n_images']`;
- `482-496`: file/hash checks;
- `498-572`: CSV replay.

There is no check that:

```python
len(pred_df_re) == c_res['n_images'] == 10816
```

There is also no exact check for:

- `Image Index` set;
- duplicate indices;
- missing indices;
- binary ground truth;
- probability domain;
- unexpected extra columns;
- exact image ordering.

### Minimum concrete adversarial CSV

Create 64 rows. For every row:

```python
row["Image Index"] = "duplicate.png"
```

For every pathology `p`:

```python
row[p] = i % 2
row["prob_" + p] = 0.1 if i % 2 == 0 else 0.9
```

Each pathology then has 32 zeros and 32 ones, with:

```text
AUC = 1.0
```

Create an AUC CSV with exactly the 14 canonical pathology rows and `auc=1.0` for each.

Use result metadata:

```python
c_res = {
    "n_images": 10816,
    "n_classes_valid": 14,
    "macro_auc": 1.0,
    "generator_checkpoint_sha256": <matching generator SHA>,
    "predictions_file": <64-row CSV>,
    "predictions_file_sha256": <actual SHA>,
    "aucs_file": <14-row AUC CSV>,
    "aucs_file_sha256": <actual SHA>,
    "auc_df": <14 rows, every AUC 1.0>,
}
```

### Executable call path

```text
run_orchestration()
→ evaluate_classification_arm()
→ evaluate_classification_val()
→ writes CSV/AUC CSV
→ check_run_validity()
```

### Observed behavior

The validator:

- finds all 14 ground-truth columns;
- finds all 14 probability columns;
- computes 14 finite AUCs;
- checks the 14-row AUC CSV;
- matches per-pathology AUCs;
- matches macro AUC.

It never checks that 64 rows represent the claimed 10,816-image cohort.

### Expected behavior

The artifact must be INVALID because:

```text
64 != 10816
```

and because its image-index set is not the frozen VAL image-index set.

### Applicability

- Real scientific CLI: **not through an unmodified producer; reachable through direct validity or artifact/metadata tampering**.
- Direct production Python API: **yes**.
- Unit-test-only path: **no**.
- Documentation only: **no**.

### Outcome

- Can return `VALID`: **yes**.
- Can produce scientific-looking artifacts: **yes**.
- Can alter scientific result/evidence: **yes**.
- Can silently drift provenance: **yes**.
- Exception/crash only: **no**.

### Smallest correct fix

Require:

```python
len(pred_df_re) == 10816
```

and require the exact expected unique `Image Index` set/order, binary labels, finite probabilities, and exact required schema.

### Regression test

```text
test_classification_replay_rejects_short_csv()
```

Use the 64-row fixture above and assert INVALID.

Before the fix, the validator accepts the self-consistent CSV.

---

## F05 — Malformed result objects raise instead of returning structured INVALID

**Severity:** MEDIUM

### Exact location

File:

```text
research_agent/m2_dev/run_m2_s1.py
```

Function:

```text
check_run_validity()
```

Ranges:

- `437-446`: dereferences `p_res.get(...)`;
- `469-480`: dereferences `c_res.get(...)`.

These accesses occur before the later replay exception handlers.

### Minimum reproducer

```python
check_run_validity(
    b_dev_manifest=<valid>,
    c4_manifest=<valid>,
    b_att_manifest=<valid>,
    c4_att_manifest=<valid>,
    b_priv=None,
    c4_priv=<valid>,
    b_class=<valid>,
    c4_class=<valid>,
)
```

or:

```python
b_class = None
```

### Observed behavior

`None.get(...)` raises `AttributeError` rather than returning a structured INVALID result.

### Expected behavior

Every malformed or absent result object should fail closed with a structured result such as:

```python
(False, "Missing or malformed privacy result")
```

### Applicability and outcome

- Real scientific CLI: **not normal; orchestrator errors usually propagate earlier**.
- Direct production Python API: **yes**.
- Unit-test-only path: **no**.
- Documentation only: **no**.
- Can return `VALID`: **no**.
- Scientific-looking artifacts: **no**.
- Silent scientific alteration: **no**.
- Failure mode: **exception/crash**.

### Smallest correct fix

Require dictionary types before any `.get()` access and return `(False, reason)`.

### Regression test

```text
test_check_run_validity_none_results_return_invalid()
```

Pass `None` result objects and assert no exception plus a structured INVALID result.

---

## F06 — `unit_test_mode` bypasses scientific contracts

**Severity:** MEDIUM assurance issue

### Exact location

File:

```text
research_agent/m2_dev/run_m2_s1.py
```

Function:

```text
check_run_validity()
```

Ranges:

- `418-419`: config SHA skipped;
- `445-446`: privacy pair count skipped;
- `477-480`: classifier SHA and image count skipped;
- `450-456`: raw privacy file optional;
- `484-496`: raw classification/AUC files optional.

T104 intentionally validates a no-raw-file bundle in unit mode.

### Minimum reproducer

```python
check_run_validity(
    <valid-looking manifests>,
    b_priv={
        "roc_auc": 1.0,
        "generator_checkpoint_sha256": ...,
        "attacker_checkpoint_sha256": ...,
    },
    c_class={
        "macro_auc": 1.0,
        "n_classes_valid": 14,
        "generator_checkpoint_sha256": ...,
    },
    unit_test_mode=True,
)
```

### Observed behavior

The function can return:

```python
(True, "VALID")
```

without requiring:

- frozen configuration hashes;
- 2,000 privacy pairs;
- 10,816 classification images;
- classifier checkpoint SHA;
- raw prediction artifacts.

`run_orchestration()` does label the overall verdict as development-only when unit mode is active. The public validity function still returns the literal scientific-looking status `VALID`.

### Expected behavior

Unit results should use a distinct status such as:

```text
DEVELOPMENT_VALID
```

or be returned only by a private test helper that cannot be used for scientific reporting.

### Applicability and outcome

- Real scientific CLI: **no; no CLI unit-mode flag is exposed**.
- Direct production Python API: **yes**.
- Unit-test-only path: **yes, intentionally but weakly labelled**.
- Documentation only: **no**.
- Can return `VALID`: **yes, literal public API result**.
- Scientific-looking artifacts: **yes, synthetic summaries/manifests**.
- Alter a real scientific result: **not through the intended unit path**.
- Silent provenance drift: **yes if misused by a caller**.

### Smallest correct fix

Separate unit validity from scientific validity and prevent unit mode from being used by scientific reporting.

### Regression test

```text
test_unit_mode_never_returns_scientific_valid_status()
```

Use the existing T104-style no-raw bundle and assert that the result is not `(True, "VALID")`.

---

## F07 — Numerical fail-closed behavior is incomplete

**Severity:** MEDIUM-HIGH

### Exact locations

#### Anonymizer runner

File:

```text
research_agent/m2_dev/anonymizer_runner.py
```

`train_epoch()`:

- `323-327`: generator backward, gradient capture, optimizer step;
- `329-353`: critic backward, gradient capture, optimizer steps.

`run()`:

- `503-527`: checks scalar metrics and selected diagnostics only;
- no comprehensive per-gradient or per-parameter finite checks.

#### Attacker

File:

```text
research_agent/m2_dev/dev_attacker.py
```

Functions:

```text
DevAttacker.train_epoch()
DevAttacker.validate_selection()
```

Ranges:

- `123-152`: averaged train loss and post-epoch parameter checks;
- `154-178`: averaged validation loss after the validation loop.

The implementation does not check every batch loss tensor or every gradient tensor.

### Minimum adversarial input

A runner-level test fixture can add an unused trainable parameter to the generator:

```python
runner.generator.unused = torch.nn.Parameter(torch.tensor(1.0))
runner.optimizer_g.add_param_group(
    {"params": [runner.generator.unused]}
)
```

A controlled optimizer wrapper can set the parameter to NaN after an otherwise finite step:

```python
runner.generator.unused.data.fill_(float("nan"))
```

The current `run()` path does not inspect this parameter.

### Observed behavior

Scalar checks can remain finite while a parameter or gradient is non-finite. The manifest can still reach:

```text
numerical_validity = PASS
```

provided the checked scalar metrics remain finite.

This deterministic reproducer uses a direct runner/test seam. The audit did not claim that standard canonical CUDA operations naturally produce this exact isolated state.

### Expected behavior

Any non-finite gradient, parameter, or relevant optimizer state must abort the run and prevent scientific validity.

### Applicability and outcome

- Real scientific CLI: **the vulnerable code path is reached**.
- Direct production API: **yes**.
- Unit-test-only: **the smallest deterministic reproducer uses a controlled test seam**.
- Documentation only: **no**.
- Can return `VALID`: **conditionally**.
- Scientific-looking artifacts: **yes**.
- Alter scientific result: **yes**.
- Silent numerical drift: **yes**.

### Smallest correct fix

After every backward and optimizer step, check:

```python
for gradient in gradients:
    require torch.isfinite(gradient).all()

for parameter in owned_parameters:
    require torch.isfinite(parameter).all()
```

Apply equivalent checks to attacker training and relevant optimizer state.

### Regression test

```text
test_runner_rejects_nonfinite_parameter_after_optimizer_step()
```

Force the unused parameter to NaN after a finite step and assert `FloatingPointError` or a failed numerical manifest.

---

## F08 — Diagnostic failures are swallowed

**Severity:** MEDIUM-HIGH

### Exact location

File:

```text
research_agent/m2_dev/anonymizer_runner.py
```

`train_epoch()`:

- `298-320`: gradient diagnostic;
- `320-321`: catches all exceptions and stores an error dictionary.

`run()`:

- `503-509`: only numeric diagnostic fields enter the finite check;
- `577-579`: PASS is based on `nan_inf_detected` and completed epochs.

The diagnostic error dictionary is:

```python
{"epoch": epoch, "error": str(_diag_e)}
```

It contains no non-finite numeric value.

### Executable path

```text
run_orchestration()
→ run_anonymizer_arm("C4")
→ M2AnonymizerRunner.run()
→ train_epoch()
→ torch.autograd.grad(...)
→ exception
→ except Exception
→ continue training
→ numerical_validity == PASS
```

### Minimum adversarial input

During a C4 diagnostic epoch, force the diagnostic call to raise a controlled `RuntimeError` while keeping training losses finite. A regression test can monkeypatch `torch.autograd.grad` only for the diagnostic block.

### Observed behavior

The exception becomes:

```python
gradient_norm_diagnostics[epoch] = {
    "epoch": epoch,
    "error": "..."
}
```

Then:

- scalar finite checks remain true;
- `nan_inf_detected` remains false;
- training continues;
- the manifest can state `numerical_validity: PASS`;
- validity can accept the manifest.

### Expected behavior

A failed required diagnostic must abort the run or mark it INVALID.

### Applicability and outcome

- Real scientific CLI: **yes, C4 diagnostics are enabled by default**.
- Direct production API: **yes**.
- Unit-only: **the deterministic exception injection is test-controlled, but the catch is production code**.
- Documentation only: **no**.
- Can return `VALID`: **yes**.
- Scientific-looking artifacts: **yes**.
- Alter scientific result/evidence: **potentially; definitely invalidates the claimed diagnostic evidence**.
- Exception/crash only: **no; the exception is suppressed**.

### Smallest correct fix

Do not treat diagnostic exceptions as non-fatal. Abort or set:

```python
numerical_validity = "FAIL"
```

and make validity reject any diagnostic error record.

### Regression test

```text
test_gradient_diagnostic_exception_invalidates_manifest()
```

Force the diagnostic call to raise and assert that the runner raises or writes a failed numerical manifest.

---

## F09 — TEST firewall and `eval_test` overclaim

**Severity:** MEDIUM-HIGH

### Exact locations

Generic firewall:

```text
research_agent/test_firewall.py
```

- `17-21`: recognizes only `test`, `testing`, and `final_test`;
- `24-35`: unrestricted `allow=True` bypass;
- `55-58`: `provenance_record()` defaults `allow=True`.

Development fold/phase checks:

```text
research_agent/m2_dev/evaluator_common.py:45-47,342-359
```

The forbidden set omits `eval_test`.

Test coverage:

```text
research_agent/m0_tests/test_m12_dev_evaluators.py:320-342
```

T40 checks only selected source literals, does not instrument runtime file access, and does not inspect `run_m2_s1.py` or `anonymizer_runner.py`.

Hard-coded status:

```text
research_agent/m2_dev/run_m2_s1.py:803
```

sets:

```python
"test_touched": False
```

### Minimum direct reproducer

```python
TestFirewall(allow=False).check("eval_test")
```

returns successfully because `is_test_request("eval_test")` is false.

Likewise:

```python
assert_dev_phase("eval_test")
```

does not reject it because the forbidden set omits the token.

Important nuance: `assert_dev_fold("eval_test")` rejects it as an unknown fold. The problem is that the generic firewall and phase checks do not establish the report's repository-wide guarantee.

### Observed behavior

The implementation does not prove that all of the following are intercepted across all modules:

```text
test
testing
eval_test
```

The generic firewall checks mode strings only. It does not inspect arbitrary image paths, loader contents, pair files, or runtime file access.

`provenance_record(mode="eval_test", allow=False)` can also pass the generic firewall.

### Expected behavior

Either:

- block every test alias and equivalent token;
- enforce path/content provenance at loader construction;
- or narrow the report to the actual canonical M2 wrapper guarantee.

`allow=True` should require explicit owner-controlled authorization rather than a public boolean.

### Applicability and outcome

- Real scientific CLI: **normal CLI passes only `dev`; the repository-wide claim is not established**.
- Direct production API: **yes**.
- Unit-only: **T40 itself is static/test-only, but the firewall seam is production code**.
- Documentation: **the overclaim is documentation-level, while the missing token is executable**.
- Can return `VALID` alone: **no**.
- Scientific-looking provenance: **yes**.
- Alter scientific result: **yes when combined with injected loaders**.
- Silent provenance drift: **yes**.

### Smallest correct fix

Add `eval_test` and equivalent aliases to the closed set, remove unrestricted bypasses, bind loaders to approved files/cohorts, and narrow unsupported report claims.

### Regression test

```text
test_eval_test_is_blocked()
```

Assert both:

```python
with pytest.raises(RuntimeError):
    TestFirewall(allow=False).check("eval_test")
```

and:

```python
with pytest.raises(RuntimeError):
    assert_dev_phase("eval_test")
```

Before the fix, generic checks can pass.

---

## F10 — Ancestry-only source guard

**Severity:** HIGH

### Exact location

File:

```text
research_agent/m2_dev/run_m2_s1.py
```

Function:

```text
check_git_source_guard()
```

Ranges:

- `600-621`.

It checks:

```text
git diff --quiet
git diff --cached --quiet
git merge-base --is-ancestor 851c3f1... HEAD
```

It does not check:

- exact HEAD;
- exact branch;
- `git status --porcelain`;
- untracked source files;
- exact canonical SHA;
- exact audit source SHA.

### Minimum adversarial input

A clean checkout at any descendant of:

```text
851c3f1a6912255c97345a7f53ed138e7ae7981d
```

passes the ancestry predicate even when it is not exact canonical HEAD `c643131...`.

An untracked importable source file also does not affect the two `git diff` checks.

### Observed behavior

A descendant can pass the source guard while containing source changes after the certified ancestor. This proves only:

```text
HEAD descends from a certified ancestor
```

not:

```text
HEAD is the exact certified source
```

### Expected behavior

Scientific execution must enforce an immutable source identity, such as exact HEAD, branch/ref, lock hash, and source manifest.

### Applicability and outcome

- Real scientific CLI: **yes**.
- Direct production API: **yes**.
- Unit-only: **no**.
- Documentation: **also overclaimed in §20**.
- Can return `VALID`: **yes**.
- Scientific-looking artifacts: **yes**.
- Alter scientific result: **yes through descendant source changes**.
- Silent provenance drift: **yes**.

### Smallest correct fix

Require exact certified source identity before execution and define an explicit policy for untracked runtime source. Reject importable untracked source or execute from a clean isolated tree.

### Regression test

```text
test_source_guard_rejects_wrong_head_and_branch()
```

Mock clean tracked/index state and successful ancestry while returning a noncanonical HEAD or branch. Assert the guard raises.

Before the fix, it returns true.

---

## F11 — T215 promotion-coverage test is fail-open

**Severity:** HIGH certification blocker

### Exact location

File:

```text
research_agent/m0_tests/test_m14c2_closeout.py
```

Function:

```text
test_t215_promotion_fileset_no_forbidden_baggage()
```

Ranges:

- `540-618`;
- central dynamic check `600-616`.

Specific defects:

```python
canonical_sha = data.get('canonical_base_commit', ...)
```

The test trusts the fileset's declared canonical SHA rather than asserting the required canonical SHA.

It performs the Git/diff check inside:

```python
try:
    ...
except Exception:
    pass
```

It converts raw lists to sets, erasing duplicate entries, and checks only:

```text
changed_files - all_classified
```

### Minimum adversarial inputs

Any of the following can cause a false pass:

1. mock `subprocess.run()` to return nonzero status;
2. provide an invalid canonical SHA;
3. return empty diff output;
4. insert duplicate paths into include/retain lists;
5. add stale/extra classified paths.

### Observed behavior

Git failures, invalid SHAs, empty diffs, and assertion failures inside the broad exception handler are swallowed. Set conversion prevents duplicate detection.

The current 25/13 partition happens to be structurally exact, but the test is not a reliable proof of that fact.

### Expected behavior

T215 must fail on any Git failure, wrong canonical SHA, empty/malformed diff, missing classification, stale/extra path, duplicate path, or overlap.

### Applicability and outcome

- Real scientific CLI: **no**.
- Direct production API: **no**.
- Certification test path: **yes**.
- Documentation only: **no**.
- Scientific `VALID`: **no**.
- Scientific-looking artifacts: **no**.
- Alter scientific result: **no**.
- Silently approve invalid promotion: **yes**.
- Exception/crash only: **no**.

### Smallest correct fix

Fail closed, assert the exact canonical SHA, preserve lists before duplicate checks, compare all changed/classified/stale sets, and remove the broad exception handler.

### Regression tests

```text
test_t215_git_failure_fails_closed()
test_t215_rejects_wrong_canonical_sha()
test_t215_rejects_duplicate_raw_paths()
test_t215_rejects_stale_classified_paths()
```

Before the fix, T215 can pass under these invalid conditions.

---

## F12 — CPU scientific fallback is broken

**Severity:** P1 launch blocker

### Exact locations

CPU fallback selection:

```text
research_agent/m2_dev/anonymizer_runner.py:91-93
```

CUDA-only ACLoss construction:

```text
research_agent/m0_port/ACLoss.py:48-49
```

The loss module unconditionally calls `.cuda()`.

### Executable call path

```text
python run_m2_s1.py \
    --scientific-m2-s1 \
    --arm all \
    --device cpu
```

then:

```text
parse_args()
→ run_orchestration()
→ run_anonymizer_arm()
→ M2AnonymizerRunner.__init__()
→ ACLoss.__init__()
→ .cuda()
```

### Observed behavior

On a CPU-only environment, construction fails before training. On a CUDA-capable environment with `--device cpu`, the loss object is still moved to CUDA while other tensors may be on CPU, creating device inconsistency.

### Expected behavior

Either support CPU consistently or reject CPU scientific execution before model/loss construction with a clear deterministic error.

### Applicability and outcome

- Real scientific CLI: **yes**.
- Direct production API: **yes**.
- Unit-only: **no**.
- Documentation only: **no**.
- Can return `VALID`: **no**.
- Scientific-looking artifacts: **only partial output possible**.
- Alter scientific result: **no; it crashes first**.
- Silent provenance drift: **no**.
- Exception/crash only: **yes**.

### Smallest correct fix

The least ambiguous fix is to reject unsupported CPU scientific mode during preflight. If CPU support is intended, replace unconditional `.cuda()` with `.to(device)` throughout ACLoss and certify the full CPU path.

### Regression test

```text
test_cpu_scientific_mode_fails_with_explicit_device_error_or_runs_end_to_end()
```

Before the fix, failure occurs inside `.cuda()` rather than at a controlled device-policy boundary.

---

## F13 — Resume state defect

**Severity:** P2

### Exact locations

File:

```text
research_agent/m2_dev/anonymizer_runner.py
```

Functions:

```text
save_resumable_checkpoint()
load_resumable_checkpoint()
run()
```

Ranges:

- `432-457`: saves sampler RNG state but not `epoch_indices`;
- `461-489`: restores sampler RNG but not `epoch_indices`;
- `491-502`: resumes and then requests an epoch order hash.

The sampler state is in:

```text
research_agent/m2_dev/evaluator_common.py:486-528
```

`get_epoch_order_hash()` raises if the requested epoch has not been recorded.

### Minimum reproducer

Save at epoch 0, load into a fresh runner with:

```python
epoch_indices == []
```

After load:

```text
start_epoch == 1
```

The first post-resume iteration records an order at index zero, then the runner asks for epoch one.

### Observed behavior

The resume API can crash or produce invalid order telemetry. The protocol already states that scientific resume is uncertified.

### Expected behavior

Either restore complete sampler order state or reject resume explicitly and require restart from epoch 0.

### Applicability and outcome

- Real scientific CLI: **not currently wired as a normal scientific resume path**.
- Direct production API: **yes**.
- Unit-only: **no**.
- Documentation only: **no**.
- Can return `VALID`: **no normally**.
- Scientific-looking artifacts: **partial/ambiguous artifacts possible**.
- Alter scientific result: **potentially if resume were accepted**.
- Silent provenance drift: **potentially**.
- Primary failure: **exception/crash**.

### Smallest correct fix

Because the protocol explicitly disallows scientific resume, reject resume in scientific mode with a clear error. Full state restoration is required only if operational resume is intended.

### Regression test

```text
test_scientific_resume_is_rejected_or_restores_epoch_indices()
```

Assert explicit rejection or successful epoch-1 order hashing after full state restoration.

**Blocking status:** not independently blocking a clean restart-from-epoch-0 launch if the prohibition is enforced.

---

## F14 — Train-order hashes are recorded but not enforced

**Severity:** MEDIUM-HIGH

### Exact locations

Frozen expected hashes:

```text
research_agent/M2_S1_EXECUTION_LOCK.json:47-53
```

The lock contains epoch-0/epoch-1 hashes and expected cross-arm equality.

Runtime logging:

```text
research_agent/m2_dev/anonymizer_runner.py:501-517
```

The runner computes and records `order_sha256` but never compares it with the lock.

### Minimum adversarial input

A sampler whose:

```python
get_epoch_order_hash(epoch)
```

returns an incorrect but well-formed SHA string.

Equivalent real-world condition: a descendant source change alters sampler ordering while still passing the ancestry-only source guard.

### Observed behavior

The incorrect hash is recorded as telemetry. No exception or INVALID result follows. The run can continue to manifest generation and validity checking.

### Expected behavior

Runtime hashes must be compared to immutable epoch values, expected cross-arm equality, and required presence for every audited epoch.

### Applicability and outcome

- Real scientific CLI: **yes**.
- Direct production API: **yes**.
- Unit-only: **no**.
- Documentation only: **no**.
- Can return `VALID`: **yes**.
- Scientific-looking artifacts: **yes**.
- Alter scientific result: **yes, through changed training order**.
- Silent provenance drift: **yes**.

### Smallest correct fix

Load immutable expected hashes and fail on any missing or mismatched runtime hash.

### Regression test

```text
test_runtime_order_hash_mismatch_invalidates_run()
```

Return a wrong hash from a controlled sampler and assert runner failure or failed validity.

Before the fix, the wrong hash is merely logged.

---

## F15 — Custom-config provenance defect

**Severity:** MEDIUM, HIGH for direct scientific API use

### Exact locations

File:

```text
research_agent/m2_dev/anonymizer_runner.py
```

Constructor ranges:

- `105-116`: arbitrary dict config receives a canonical default `config_path`.

Manifest generation ranges:

- `551-570`: manifest hashes `self.config_path`.

### Minimum reproducer

```python
M2AnonymizerRunner(
    arm="B_dev",
    config={
        "mu": 0.99,
        "batch_size": 16,
        "image_size": 256,
        "learning_rate": 1e-4,
        "max_epochs": 250,
    },
    config_path=None,
    unit_test_mode=False,
)
```

Since no path is supplied, `self.config_path` defaults to the canonical B_dev config path even though the in-memory configuration differs.

### Observed behavior

The manifest can report the canonical config path and SHA while the actual in-memory config uses a different `mu`, feature weight, or other setting.

`check_run_validity()` checks the manifest file hash, not equality between the manifest hash and the actual in-memory configuration.

### Expected behavior

Reject arbitrary dict configs in scientific mode or hash the exact serialized in-memory config and compare it with immutable frozen content.

### Applicability and outcome

- Real scientific CLI: **not through the normal wrapper, which supplies an explicit path**.
- Direct production API: **yes**.
- Unit-only: **no**.
- Documentation only: **no**.
- Can return `VALID`: **yes**.
- Scientific-looking artifacts: **yes**.
- Alter scientific result: **yes**.
- Silent provenance drift: **yes**.
- Exception/crash only: **no**.

### Smallest correct fix

When `config` is a dict and `config_path` is absent, reject in non-unit mode or serialize/hash the actual dict and compare every frozen field.

### Regression test

```text
test_custom_dict_config_cannot_claim_canonical_config_sha()
```

Pass a dict with `mu=0.99` and no `config_path`; assert construction or run fails in non-unit mode.

Before the fix, the runner assigns the canonical path and can claim the canonical SHA.

---

## F16 — Forensic report overclaims and missing §26

**Severity:** HIGH certification/documentation blocker

### Exact file

```text
research_agent/M1_4C_FINAL_FORENSIC_CERTIFICATION.md
```

### Exact unsupported claims

Classification contract, `198-204`:

The report claims all AUCs lie strictly in `(0.0, 1.0)`. The implementation checks finiteness and one-class failure, not the open interval.

Numerical claims, `225-230`:

The report says attacker loss tensors are checked at every iteration. The implementation checks aggregate epoch-level values in:

```text
research_agent/m2_dev/dev_attacker.py:123-178
```

Source claims, `234-239`:

The report claims `git status --porcelain` cleanliness and exact canonical HEAD verification. The source guard only checks diffs plus ancestry from `851c3f1...`.

Firewall claims, `265-269`:

The report claims `test`, `testing`, `eval_test`, and test-split paths are intercepted across all modules. The implementation does not establish this.

Final verdict, `273-285`:

The report declares:

```text
FULL PASS
100% hardened and certified
```

despite the unsupported claims above.

Missing section:

The file ends at §25. There is no §26 section/content.

### Executable path

There is no runtime call path. The report is consumed by human auditors, promotion reviewers, and certification workflows.

### Observed behavior

The report presents unsupported guarantees as completed certification and omits an expected section.

### Expected behavior

The report must state only implemented guarantees, distinguish synthetic/unit-only evidence from scientific execution, correct source/firewall/numerical claims, and add or explicitly account for §26.

### Applicability and outcome

- Real scientific CLI: **no direct runtime effect**.
- Direct production API: **no**.
- Unit-only: **no**.
- Report/documentation: **yes**.
- Can return `VALID`: **no directly**.
- Can produce scientific-looking artifacts: **not directly**.
- Can alter scientific result: **no direct metric change**.
- Can silently drift provenance: **yes, in the certification record**.
- Can cause false promotion/signoff: **yes**.

### Smallest correct fix

Correct the report, add §26, remove unsupported claims, and bind claims to exact source/test evidence.

### Regression test

```text
test_forensic_report_has_complete_sections_and_supported_claims()
```

Assert that:

- §26 exists;
- unsupported `eval_test`, exact-HEAD, and per-iteration claims are removed or qualified;
- `FULL PASS` is not present while blocking findings remain;
- synthetic/unit-only evidence is labelled as such.

Before the fix, the current report fails these assertions.

---

# 4. Supplemental findings

## S17 — `fold='validation'` alias inconsistency

**Severity:** MEDIUM

### Exact locations

Allowed fold list:

```text
research_agent/m2_dev/evaluator_common.py:45-47
```

It allows both `val` and `validation`.

Fold checks:

```text
research_agent/m2_dev/evaluator_common.py:351-359
research_agent/m2_dev/eval_classifier_val.py:136-200
```

The 10,816 check is conditional on `fold == 'val'`.

Dataset:

```text
chexnet/cxr_dataset.py:9-29
```

The dataset documents only `train`, `val`, and `test` and filters exact equality.

### Behavior

With `fold='validation'`:

- the alias whitelist accepts it;
- the dataset searches for literal `validation`;
- the NIH CSV uses `val`;
- the 10,816 check is skipped;
- the dataset may be empty and fail later during AUC calculation.

With an injected loader, the alias can bypass the 10,816 check and return metrics.

### Smallest fix

Require exactly `fold='val'`, or normalize `validation` to `val` before dataset creation and apply all checks afterward.

### Regression test

```text
test_classifier_rejects_or_normalizes_validation_alias()
```

Before the fix, the alias reaches dataset construction and bypasses the normal size check.

---

## S18 — Mutable execution-lock provenance

**Severity:** MEDIUM-HIGH

### Exact locations

File:

```text
research_agent/m2_dev/evaluator_common.py
```

Function:

```text
verify_frozen_scientific_configs()
```

Ranges:

- `224-237`: expected SHAs are read from mutable lock `artifact_provenance`;
- `239-258`: current files are compared against those values.

The lock is:

```text
research_agent/M2_S1_EXECUTION_LOCK.json
```

but no immutable lock SHA/signature is verified.

### Executable path

```text
run_orchestration()
→ verify_environment_and_hashes()
→ assert_m2_scientific_mode_ready()
→ verify_scientific_dependencies()
→ verify_frozen_scientific_configs()
```

### Observed behavior

The verifier proves:

```text
current config SHA == SHA declared by current lock
```

It does not prove:

```text
current lock == immutable certified lock
```

A clean descendant can therefore modify lock/config authority without being rejected by exact-source validation.

### Expected behavior

The lock itself must be anchored to an immutable SHA or signature before its expected hashes are trusted.

### Smallest fix

Verify the exact lock blob/hash before reading `artifact_provenance` and bind it to exact certified source.

### Regression test

```text
test_modified_execution_lock_is_rejected_before_config_hash_use()
```

Supply a temporary lock with altered artifact SHA values and assert preflight rejects it. Before the fix, the lock can redefine expected hashes.

---

## S19 — Final-source 216/216 inventory is unproven

**Severity:** HIGH certification blocker

### Exact locations

Inventory:

```text
research_agent/M1_4C2_TEST_INVENTORY.json:1-6,378-384
```

It records counts and timings but not:

- source commit;
- source tree SHA;
- test-file hashes;
- runtime identity;
- exact command.

Closeout claim:

```text
research_agent/M1_4C2_FINAL_CLOSEOUT.md:139-145
```

Git history shows the inventory was committed before the target commit changed `test_m14c2_closeout.py`.

### Observed behavior

The inventory can truthfully describe a prior 216/216 run while failing to prove that the final T201-T216 source was executed.

### Expected behavior

The inventory must bind every result to exact source commit, exact test-file/tree hashes, command, environment, and terminal status.

### Smallest fix

Rerun the final source and generate a new inventory containing exact source/environment hashes.

### Regression test

```text
test_inventory_binds_to_current_source_tree()
```

Compare recorded source and test-file hashes with the current target. The current inventory would fail.

---

# 5. Additional evidence-only issues

## Git-LFS checkpoint reporting

At exact commit `512305f`, checkpoint paths are 133-byte Git-LFS pointer blobs. Pointer metadata records the hydrated sizes:

```text
generator:    31,112,853 bytes
classifier:   28,504,657 bytes
verification: 95,407,079 bytes
```

The hydrated worktree contains real binaries whose SHA256 values match the frozen values.

`M1_4C2_FINAL_CLOSEOUT.md:162-166` reports different sizes:

```text
31,116,923
29,322,467
95,202,467
```

This is a provenance/documentation defect, not a SHA identity failure.

## Promotion references

The include/retain partition is structurally exact:

```text
38 changed
25 include
13 retain
0 overlap
0 unclassified
```

However, promoted/current reports reference retained historical artifacts, including:

- `M1_4C2_TEST_INVENTORY.json`;
- `audit_operator_equivalence.py`;
- `upstream_10run_reproduction_results.json`.

These references must be explicitly classified as audit-history references or removed/repaired for a clean promotion.

## Whitespace

`git diff --check c643131..512305f` reports trailing-whitespace violations in added reports and scripts. This is low-severity promotion hygiene, not a scientific-method defect.

## Historical operator record

The existing canonical `research_agent/operator_audit_results.json` contains measured values inconsistent with its own conclusion. It is historical/non-runtime and should not be used as clean current evidence without reconciliation.

---

# 6. Minimum required fix set

Only genuinely blocking fixes are included. No method redesign is required.

## 6.1 Close scientific dependency-injection seams

- Reject injected models, loaders, samplers, and arbitrary config dictionaries in non-unit mode.
- Permit them only under explicitly non-scientific unit-test mode.
- Bind normal evaluators to frozen checkpoints, pair files, image-index sets, row hashes, and cohort fingerprints.

Covers: F01, F15, S17.

## 6.2 Centralize frozen scientific argument enforcement

Validate at `run_orchestration()` and every public scientific entrypoint:

```text
arm == all
max_epochs == 250
attacker_epochs == 100
attacker_patience == 5
seed == 42
attacker_seed == 42
```

Do not rely on `parse_args()` as the only enforcement boundary.

Covers: F02.

## 6.3 Make replay structurally fail closed

Privacy replay must require:

```text
exactly 2,000 rows
matching one-dimensional arrays
binary labels
finite values
frozen pair/order provenance
```

Classification replay must require:

```text
exactly 10,816 rows
exact unique Image Index set/order
binary labels
finite probabilities
exact required schema
```

Malformed result objects must return structured INVALID rather than raising. Unit-mode results must not be reported as scientific `VALID`.

Covers: F03, F04, F05, F06.

## 6.4 Make numerical failures genuinely fail closed

- Check gradients and owned parameters after every update.
- Reject non-finite optimizer state where relevant.
- Do not swallow required diagnostic exceptions.
- Prevent any diagnostic error from producing a PASS manifest.

Covers: F07, F08.

## 6.5 Enforce exact source and protocol authority

- Require exact certified source commit/branch or an equivalent immutable source manifest.
- Verify the execution-lock hash before trusting artifact-provenance fields.
- Reject importable untracked runtime source.
- Enforce the existing restart-from-epoch-0 policy for scientific mode.

Covers: F10, S18, and operationally contains F13.

## 6.6 Enforce train-order provenance

- Compare runtime epoch hashes with immutable lock values.
- Enforce required cross-arm equality.
- Treat missing or mismatched hashes as INVALID.

Covers: F14.

## 6.7 Close and accurately describe the TEST firewall

- Block `eval_test` and equivalent aliases.
- Remove unrestricted public boolean bypasses or require explicit owner authorization.
- Bind loader/path provenance instead of relying solely on mode strings.
- Remove or narrow unsupported repository-wide firewall claims.

Covers: F09 and S17.

## 6.8 Make device policy fail closed

Either:

- implement device-correct CPU support using `.to(device)`; or
- reject CPU scientific execution before model/loss construction.

Do not advertise a CPU fallback that fails inside `.cuda()`.

Covers: F12.

## 6.9 Repair promotion and certification evidence

- Make T215 fail closed.
- Assert the exact canonical SHA.
- Reject duplicate, stale, extra, missing, or overlapping fileset paths.
- Fail on Git errors and empty/malformed diffs.
- Rerun the complete final-source suite.
- Bind the inventory to exact source/environment hashes.
- Correct forensic-report overclaims.
- Add the missing §26.
- Correct LFS/size statements.
- Resolve or explicitly classify retained-artifact references.

Covers: F11, F16, S19, and the evidence-only issues.

---

# 7. Final disposition

The audited branch should not yet be promoted to canonical and should not yet authorize a fresh scientific M2-S1 launch.

The method implementation is not the primary blocker. The blockers are the integrity boundary, replay completeness, exact-source binding, numerical fail-closed behavior, promotion-test reliability, and certification evidence accuracy.

Final decision:

```text
C) BLOCKED — SMALL CERTIFICATION/PROMOTION CLOSEOUT REQUIRED
```

No code fix, method change, commit, push, training, M2 launch, or TEST access is authorized by this evidence package.
