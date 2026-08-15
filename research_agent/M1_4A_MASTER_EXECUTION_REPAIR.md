# M1.4a — MASTER EXECUTION REPAIR REPORT

**Status**: ALL CHECKS PASS (26/26 M1.4a Gates PASS; 86/86 M0–M1.4 Regression PASS; Firewall Self-Check PASS)  
**Protocol Version**: M1.4a (Master Execution Repair for M2-S1 Launch)  
**Branch**: `research/method-restart`  
**Execution Lock**: `research_agent/M2_S1_EXECUTION_LOCK.json` (v1.4.1)  
**Official TEST Status**: STRICTLY CLOSED (0 Test Access; verified by T112 + firewall self-check)

---

## A. Executive Summary

M1.4a executes the final repair of the master execution harness immediately prior to launching the ~32 GPU-hour M2-S1 paired scientific run (B_dev vs C4 at mu=0.01, legacy operator, batch 16/accum 1, Adam lr 1e-4, max_epochs 250, seeds 42). A new integration suite (`test_m14a_execution_harness.py`, gates **T87–T112**) was added to certify the end-to-end orchestration path that the M1.4 regression suite did not cover: `run_m2_s1.run_orchestration`, the adaptive attacker contracts, the privacy/classification evaluator paths, the scientific CLI fail-closed checks, and the full synthetic master smoke.

Initial execution of the M1.4a suite surfaced **six distinct root-cause defects** in the harness and its reference tests. All six were repaired, the full suite re-run, and **26/26 M1.4a gates now PASS**. Combined with the re-verified 86/86 M0–M1.4 regression tests and the firewall self-check, the repository is certified launch-ready.

Key accomplishments in M1.4a:
- **Gradient-diagnostic non-interference** (T94/T95): the C4 feature loss is no longer silently disabled when gradient diagnostics are turned off.
- **True one-step gradient parity** (T97–T99) against an independent reference after correcting a double-sigmoid error in the reference classifier term.
- **Explained the historical ~2e-4 parameter-parity gap** as CUDA bitwise nondeterminism (two identical runs differ ~2e-4), confirmed by deterministic CPU re-verification at <1e-7.
- **Master runner E2E smoke** (T110–T112) now completes end-to-end with synthetic loaders after repairing `unit_test_mode` forwarding in `run_orchestration`.
- **Scientific CLI fail-closed** (T100–T104) and **manifest-based handoffs** (T105–T107) certified.

---

## B. M1.4a Gate Status

| # | Gate | Status |
| :-- | :--- | :--- |
| T87 | DevAttacker.run returns structured history dict | PASS |
| T88 | DevAttacker saves exact best validation-loss checkpoint | PASS |
| T89 | attacker manifest SHA matches actual checkpoint | PASS |
| T90 | master attacker call uses run(), not nonexistent train() | PASS |
| T91 | master reporting path has pandas available | PASS |
| T92 | privacy arrays stored in NPZ, not float-cast in JSON | PASS |
| T93 | M2 summary JSON serializes end-to-end | PASS |
| T94 | gradient diagnostics do not alter generator scientific update | PASS |
| T95 | C4 gradient diagnostic returns finite base/feature norms | PASS |
| T96 | non-vacuous classification test verifies 14 finite AUCs | PASS |
| T97 | true generator gradient parity against independent reference | PASS |
| T98 | true verifier gradient parity | PASS |
| T99 | true classifier gradient parity | PASS |
| T100 | scientific CLI rejects max_epochs != 250 | PASS |
| T101 | scientific CLI rejects anonymizer seed != 42 | PASS |
| T102 | scientific CLI rejects attacker seed != 42 | PASS |
| T103 | scientific CLI rejects attacker patience != 5 | PASS |
| T104 | run_valid is computed, not hardcoded | PASS |
| T105 | generator manifest path+SHA handoff enforced | PASS |
| T106 | attacker manifest path+SHA handoff enforced | PASS |
| T107 | privacy evaluator output checkpoint SHAs verified | PASS |
| T108 | privacy evaluation requires exactly 2000 VAL pairs in scientific mode | PASS |
| T109 | classification scientific VAL requires 10816 images | PASS |
| T110 | full synthetic master orchestration completes end-to-end | PASS |
| T111 | master orchestration creates report + summary + prediction artifacts | PASS |
| T112 | master orchestration never constructs TEST loader | PASS |

**Result: 26/26 PASS.**

---

## C. Root-Cause Defects Found and Repaired

### C.1 — `eval_reid_val.py`: missing `import os`; image_size not threaded
- `import os` was absent, raising `NameError` inside the scientific 2,000-pair path.
- `evaluate_reid_val` did not accept/pass `image_size`, so unit-test mode (64 px) collided with the frozen flow-field components built at 256 px.
- Unit-test mode only used the synthetic loader when the configured `image_path` was missing; it now always uses the synthetic loader when no explicit loader is supplied.

### C.2 — `dev_attacker.py`: `load_frozen_anonymizer`/`DevAttacker` image_size not threaded
- `load_frozen_anonymizer(config=None, device=None, checkpoint_path=None, image_size=None)` and `DevAttacker.__init__(..., image_size=None)` now forward `image_size` into `make_flow_field_components(device, image_size=...)`, fixing the 256-vs-64 shape mismatch in T87/T88/T89.

### C.3 — `anonymizer_runner.py`: diagnostic flag disabling the scientific C4 feature loss
- The gradient-diagnostic branch was gated on `self.arm == 'C4'` alone, so `gradient_diagnostics_enabled=False` (used to obtain the scientific update) silently dropped the C4 feature term.
- New constructor param `gradient_diagnostics_enabled=True`, stored as `... and arm == 'C4'`; the diagnostic gate now reads `self.gradient_diagnostics_enabled and self.arm == 'C4'`. T94 proves enabling diagnostics changes generator parameters by < 1e-7.

### C.4 — `eval_classifier_val.py`: image_size not threaded; synthetic branch condition
- `evaluate_classification_val(..., image_size=None)`; flow-field components use the threaded size (default 256); the synthetic branch now keys on `unit_test_mode` alone with default image_size 64.

### C.5 — `run_m2_s1.py`: unit-test mode not fully threaded
- `run_anonymizer_arm` forces `cfg['image_size'] = 64` in unit-test mode.
- `DevAttacker`, `evaluate_reid_val`, and `evaluate_classification_val` receive `image_size=64 if unit_test_mode else None`.

### C.6 — `run_m2_s1.py` `run_orchestration`: `unit_test_mode` NOT forwarded (new, discovered during M1.4a)
- `run_orchestration` accepted `unit_test_mode` but never passed it to `run_anonymizer_arm` or `train_s1_attacker_arm`. As a result the T110–T112 smoke silently built **real image loaders** over the NIH dataset (observed as a hang reading PNGs via `pil_loader`).
- Fixed by forwarding `unit_test_mode=unit_test_mode` at both anonymizer-arm call sites and both attacker-arm call sites. The smoke now completes in ~5 s on CUDA with synthetic loaders.

### C.7 — `test_m14_final_hardening.py`: reference double-sigmoid
- `independent_upstream_reference_one_step` applied a second `torch.sigmoid` to the classifier head, whose forward pass already terminates in `Sigmoid` (head = `Sequential(Linear(1024,14), Sigmoid())`), diverging from true upstream `criterion_ac(outputs_ac, labels)` with `nn.BCELoss`.
- Step 6 now uses `loss_ac = crit_ac(ac_probs_c, labels)`, restoring exact classifier/verifier/generator one-step parity (T97–T99).

### C.8 — `test_m14a_execution_harness.py` T94: baseline comparison flaw + CUDA nondeterminism
- The original T94 loaded **post-training** weights into the diagnostic runner and its `arm='C4_nodiag'` hack disabled the feature loss entirely.
- Rewritten: CPU device (deterministic), deep-copies INITIAL weights before training, runner2 uses `gradient_diagnostics_enabled=False`, asserts both feature terms > 0 and generator-parameter diff < 1e-7.

---

## D. CUDA Bitwise Nondeterminism (Explanation of the Historical ~2e-4 Parity Gap)

M1.4 documented a residual generator parameter gap of ~2e-4 between the runner and the upstream reference. Investigation during M1.4a established the cause:

- **CUDA kernels are not bitwise deterministic across launches.** Two *identical* runs of the C4 runner on the same CUDA device produce generator parameters differing by ~2e-4 (measurement noise floor), not a scientific discrepancy.
- The same comparison on **CPU with `torch.set_num_threads(8)` is deterministic**; the independent-reference one-step parity there closes to < 1e-7 (T97–T99).
- One tiny CPU train epoch takes ~0.8 s, so the deterministic CPU verification is fast and reproducible.

This explains why M1.4's CUDA-based parameter-parity checks could never reach machine precision: the variance is intrinsic to GPU nondeterminism, not the harness.

---

## E. Gradient-Diagnostic Non-Interference (T94/T95)

- **T94** proves `gradient_diagnostics_enabled=True` vs `False` changes generator parameters by < 1e-7 after one epoch on identical data, while both feature terms remain > 0 — diagnostics are observationally transparent to the scientific update.
- **T95** proves the C4 diagnostic returns finite base/feature norms on CUDA.

---

## F. True One-Step Gradient Parity (T97–T99)

| Component | Check | Result |
| :--- | :--- | :--- |
| Generator gradient | runner vs independent reference | PARITY |
| Verifier critic gradient | runner vs independent reference | PARITY |
| Classifier critic gradient | runner vs independent reference | PARITY |

Parameter-parity tolerance on deterministic CPU is < 1e-7 (see Section D).

---

## G. Scientific CLI Fail-Closed (T100–T104)

The scientific launcher rejects any configuration not matching the frozen M2-S1 protocol:
- `max_epochs != 250` → rejected.
- anonymizer `seed != 42` → rejected.
- attacker `seed != 42` → rejected.
- attacker patience `!= 5` → rejected.
- `run_validity` is **computed** from manifest/eval evidence (T104), never hardcoded.

The preferred launch form is `--scientific-m2-s1`.

---

## H. Manifest-Based Handoffs (T105–T107)

- T105: generator handoff requires the `checkpoint_manifest.json` selected path + matching SHA256; mismatch raises `RuntimeError`.
- T106: attacker manifest path + SHA256 verified before privacy evaluation.
- T107: privacy evaluator output references the verified checkpoint SHAs.

No implicit/historical generator fallback exists anywhere on the evaluation path (consistent with M1.4 P0-2).

---

## I. Attacker & Evaluator Contracts (T87–T93, T108–T109)

- `DevAttacker.run()` returns a structured history dict and saves the exact best validation-BCE checkpoint (T87–T88); the attacker manifest SHA matches the saved file (T89).
- Privacy NPZ outputs store raw arrays (no float-casting) and the JSON summary serializes cleanly (T92–T93).
- Scientific privacy VAL must contain exactly **2,000** pairs (T108).
- Scientific classification VAL must contain exactly **10,816** images and yield 14 finite disease AUCs (T109, T96).

---

## J. Master Runner End-to-End Smoke (T110–T112)

- T110: full synthetic `run_orchestration` (B_dev + C4 anonymizers, both S1 attackers, privacy VAL, classification VAL) completes without exception; summary `run_status == VALID` and verdict `C4 S1: DO NOT PROMOTE` (expected for synthetic data). Runtime ~5 s on CUDA.
- T111: `M2_S1_summary.json`, `M2_S1_C4_RESULT.md`, and both `privacy_val_predictions.npz` artifacts are created.
- T112: the orchestration never constructs a TEST loader (firewall enforced).

---

## K. Full Regression Status

| Suite | Result |
| :--- | :--- |
| M0–M1.4 regression (`run_all.py`, T1–T86 + M0.1/M1/M1.1) | **86/86 PASS** |
| M1.4a gates (T87–T112) | **26/26 PASS** |
| TEST firewall self-check (`test_t13_t14_firewall.py`) | **PASS** |

---

## L. Execution Lock Update (v1.4.1)

`research_agent/M2_S1_EXECUTION_LOCK.json` was advanced from 1.4.0 to 1.4.1 with the M1.4a certification fields:
- `master_runner_e2e_smoke`: PASS (T110–T112)
- `attacker_privacy_contracts`: PASS (T87–T93, T105–T108)
- `gradient_diagnostic_noninterference`: PASS (T94–T95)
- `true_gradient_parity`: PASS (T97–T99)
- `scientific_cli_frozen`: true (T100–T104)
- `run_validity_computed`: true (T104)
- `manifest_handoffs`: true (T105–T107)
- `test_firewall`: CLOSED (T112 + self-check)

**No scientific gate, config hash, frozen weight, or dataset hash was modified.** All frozen scientific choices remain exactly as locked in v1.4.0.

---

## M. Remaining Risks & Launch Conditions

- **Launch gate**: all M1.4a gates PASS + 86/86 regression PASS + firewall PASS + lock v1.4.1 → cleared.
- **GPU**: RTX 5070 Ti must be free (preflight peak VRAM ~8.8 GB/arm) before launch; if a foreign job occupies the device the launch is blocked, not interrupted.
- **Runtime**: full M2-S1 is a ~32 GPU-hour run; launched under tmux session `m2_s1_c4` with `--scientific-m2-s1` (fail-closed unless max_epochs==250, attacker_epochs==100, attacker_patience==5, seed==42, attacker_seed==42).
- **S2 must not be run**; S1 produces the promote/do-not-promote verdict only.

---

## N. Pre-Launch Checklist

- [x] M1.4a suite 26/26 PASS
- [x] M0–M1.4 regression 86/86 PASS
- [x] Firewall self-check PASS
- [x] Execution lock v1.4.1 committed
- [x] Local HEAD == origin/research/method-restart (no force push)
- [ ] GPU free (nvidia-smi) → then launch tmux `m2_s1_c4`
- [ ] Monitor first epoch for finite losses before unattended operation

---

## O. Final Verdict

**M1.4A PASS** — the master execution harness is repaired, certified, and cleared for the M2-S1 scientific launch.
