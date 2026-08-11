"""Training diagnostics persistence (STEP 2B Part 2).

A machine-readable JSON file is persisted per attacker restart. The file used to
determine run validity (training_diagnostics.json) deliberately contains NO test
metric: test information is written to separate files (run_state.json,
test_metrics.json) so test numbers cannot leak into the validity stage.

The classification does not read this file to learn test AUC; it reads exactly the
training/validation fields below.
"""

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, 'rb') as f:
        return sha256_bytes(f.read())


def sha256_str(text: str) -> str:
    return sha256_bytes(text.encode('utf-8'))


def json_serializable(obj: Any) -> Any:
    """Best-effort conversion of common numpy/torch scalars to JSON-compatible values."""
    import numpy as np
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch_float):
        return obj.item()
    return obj


def torch_float(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def build_training_diagnostics(
    *,
    attacker_seed: int,
    transform_mode: str,
    mu: float,
    stochastic_lambda: float,
    generator_checkpoint_path: str,
    generator_checkpoint_hash: str,
    pair_train_path: str,
    pair_validation_path: str,
    pair_train_hash: str,
    pair_validation_hash: str,
    epochs_completed: int,
    termination_reason: str,
    training_loss_per_epoch: List[float],
    validation_loss_per_epoch: List[float],
    validation_auc_per_epoch: List[float],
    validation_accuracy_per_epoch: List[float],
    best_validation_loss: float,
    best_validation_loss_epoch: int,
    best_validation_auc: float,
    best_validation_auc_epoch: int,
    any_nan_inf: bool,
    checkpoint_exists: bool,
    checkpoint_loadable: bool,
    weights_changed_from_initialization: bool,
    run_start_timestamp: str,
    run_end_timestamp: str,
    protocol_documents: Optional[Dict[str, str]] = None,
    frozen_artifacts: Optional[Dict[str, str]] = None,
    initial_parameter_hash: str = '',
    final_parameter_hash: str = '',
) -> Dict[str, Any]:
    """Assemble the canonical training-diagnostics record (no test fields).

    :param protocol_documents: {path: sha256} of the frozen authoritative protocol
        documents (R-12: 01_ADAPTIVE_REID_PROTOCOL.md, 01B_PROTOCOL_AMENDMENT.md).
    :param frozen_artifacts: {path: sha256} of other frozen inputs (e.g. Top-k list).
    :param initial_parameter_hash: sha256 of the trainable parameters at the start of
        training (R-2: weight-update state). Empty string when not applicable (stub).
    :param final_parameter_hash: sha256 of the trainable parameters after training.
    """
    return {
        'attacker_seed': attacker_seed,
        'transform_mode': str(transform_mode),
        'mu': float(mu),
        'stochastic_lambda': float(stochastic_lambda),
        'generator_checkpoint_path': str(generator_checkpoint_path),
        'generator_checkpoint_hash': str(generator_checkpoint_hash),
        'pair_train_path': str(pair_train_path),
        'pair_validation_path': str(pair_validation_path),
        'pair_train_hash': str(pair_train_hash),
        'pair_validation_hash': str(pair_validation_hash),
        'epochs_completed': int(epochs_completed),
        'termination_reason': str(termination_reason),
        'training_loss_per_epoch': [float(x) for x in training_loss_per_epoch],
        'validation_loss_per_epoch': [float(x) for x in validation_loss_per_epoch],
        'validation_auc_per_epoch': [float(x) for x in validation_auc_per_epoch],
        'validation_accuracy_per_epoch': [float(x) for x in validation_accuracy_per_epoch],
        'best_validation_loss': float(best_validation_loss),
        'best_validation_loss_epoch': int(best_validation_loss_epoch),
        'best_validation_auc': float(best_validation_auc),
        'best_validation_auc_epoch': int(best_validation_auc_epoch),
        'any_nan_inf': bool(any_nan_inf),
        'checkpoint_exists': bool(checkpoint_exists),
        'checkpoint_loadable': bool(checkpoint_loadable),
        'weights_changed_from_initialization': bool(weights_changed_from_initialization),
        'initial_parameter_hash': str(initial_parameter_hash),
        'final_parameter_hash': str(final_parameter_hash),
        'protocol_documents': {str(p): str(h) for p, h in (protocol_documents or {}).items()},
        'frozen_artifacts': {str(p): str(h) for p, h in (frozen_artifacts or {}).items()},
        'run_start_timestamp': str(run_start_timestamp),
        'run_end_timestamp': str(run_end_timestamp),
    }


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


# --- Validity stage reads ONLY training_diagnostics.json ------------------------
VALIDITY_FILENAME = 'training_diagnostics.json'
RUNSTATE_FILENAME = 'run_state.json'
TESTMETRICS_FILENAME = 'test_metrics.json'

# Fields that a training-diagnostics record MUST carry. Kept in sync with the
# classifier's required keys so the persisted real-run file can be consumed by
# classify_run_health without any transformation.
REQUIRED_DIAGNOSTICS_FIELDS = frozenset({
    'attacker_seed', 'transform_mode', 'mu', 'stochastic_lambda',
    'generator_checkpoint_path', 'generator_checkpoint_hash',
    'pair_train_path', 'pair_validation_path',
    'pair_train_hash', 'pair_validation_hash',
    'epochs_completed', 'termination_reason',
    'training_loss_per_epoch', 'validation_loss_per_epoch',
    'validation_auc_per_epoch', 'validation_accuracy_per_epoch',
    'best_validation_loss', 'best_validation_loss_epoch',
    'best_validation_auc', 'best_validation_auc_epoch',
    'any_nan_inf', 'checkpoint_exists', 'checkpoint_loadable',
    'weights_changed_from_initialization',
    'run_start_timestamp', 'run_end_timestamp',
})

# Fields that a training-diagnostics record MUST NOT carry (test-derived data).
FORBIDDEN_DIAGNOSTICS_FIELDS = frozenset({
    'test_auc', 'test_predictions', 'test_labels', 'test_metrics',
})


def validate_training_diagnostics_schema(record: Any) -> List[str]:
    """Return a list of schema violations (empty list == valid).

    Checks that every required field is present and that no test-derived field leaked
    into the training-diagnostics record (R-8 / R-11).
    """
    violations = []
    if not isinstance(record, dict):
        return ['training diagnostics record must be a dict, got %s' % type(record).__name__]
    missing = sorted(REQUIRED_DIAGNOSTICS_FIELDS - set(record))
    for field in missing:
        violations.append('missing required field %r' % field)
    leaked = sorted(FORBIDDEN_DIAGNOSTICS_FIELDS & set(record))
    for field in leaked:
        violations.append('forbidden test-derived field %r must not be present' % field)
    return violations


def assert_valid_diagnostics_schema(record: Any) -> None:
    violations = validate_training_diagnostics_schema(record)
    if violations:
        raise ValueError('training diagnostics schema violated: %s' % '; '.join(violations))


def is_finite_series(values) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


def persist_real_training_diagnostics(path: str, record: Dict[str, Any]) -> None:
    """Write a REAL training-diagnostics record after schema validation (no stubs).

    Real-run persistence must never emit a record that violates the schema or mixes in
    test-derived fields. Stub/synthetic records must be written through the runner's
    stub builder instead, so they can never be confused with real diagnostics (R-11).
    """
    assert_valid_diagnostics_schema(record)
    write_json(path, record)