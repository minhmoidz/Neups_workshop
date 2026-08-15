# Protocol Authority & Document Hierarchy

**Date**: 2026-08-15  
**Branch**: `audit/m2-final-certification`  
**Certified Ancestor**: `851c3f1a6912255c97345a7f53ed138e7ae7981d`

---

## 1. Authoritative Current Lock

The **sole authoritative scientific and execution lock** for M2-S1 execution is:

```
research_agent/M2_S1_EXECUTION_LOCK.json (v1.4.2+)
```

All experimental scripts, assertions, evaluators, and preflight routines must validate against the parameters, hashes, and invariants declared in `M2_S1_EXECUTION_LOCK.json`.

---

## 2. Superseded Historical Documents

The following historical documents and protocol locks are **superseded** and retained strictly for provenance and audit history:

| Document | Superseded By | Reason |
|---|---|---|
| `research_agent/M1_C4_PROTOCOL_LOCK.json` | `M2_S1_EXECUTION_LOCK.json` | Pre-dated F1–F13 execution hardening, operator fix, and hash audits |
| `research_agent/M1_4_TRUE_PARITY_RESULTS.json` | `M1_4C_FINAL_PARITY_CERTIFICATION.json` | Replaced by truly independent pristine upstream reference parity |
| `research_agent/M1_C4_DEVELOPMENT_PROTOCOL_LOCK.md` | `M2_S1_EXECUTION_LOCK.json` | Early development draft superseded by M1.4b/M1.4c locks |
| Earlier M1 / M0 audit documents | `M2_S1_EXECUTION_LOCK.json` | Historical evolution records |

> [!WARNING]
> In any conflict between a historical document and `M2_S1_EXECUTION_LOCK.json` / certified code under `research_agent/m2_dev/`, the **current execution lock and certified code take absolute precedence**.

---

## 3. Explicit Method and Provenance Statements

1. **B_dev Control Definition**: `B_dev` is a **matched development control** ($\mu=0.01$, legacy operator, batch size 16, accumulation steps 1, seed 42), **not** an exact batch-64 upstream retraining (because UNet contains BatchNorm layers).
2. **C4 Online Teacher**: In C4, the feature representation is extracted from a deepcopy of the **active, evolving auxiliary classifier critic** at each forward step. It is **not** a frozen, static CheXNet teacher.
3. **Optimizers**:
   - Generator: `Adam(lr=1e-4)`
   - Verifier Critic: `Adam(lr=1e-4)`
   - Auxiliary Classifier Critic: `SGD(lr=1e-4, momentum=0.9, weight_decay=1e-4)`
4. **Checkpoint Selection**: Both arms use **method-neutral selection** based on minimum validation $(L_{AC\_BCE} + L_{priv})$ across epochs. The feature loss term is **strictly excluded**. Ties break to the earliest epoch.
5. **Resume Policy**: Scientific resume is **uncertified** because data sampler `epoch_indices` are not persisted across process restarts. In scientific mode, any interrupted run must restart from epoch 0.
