"""Arm provenance record (STEP 2B Part 12).

Every arm receives a provenance JSON. It must be impossible to confuse legacy vs
corrected modes or different generator checkpoints: all of

    transform_mode, generator_checkpoint_path, generator_checkpoint_hash, mu, seeds...

are recorded explicitly.
"""

from typing import Any, Dict, List, Optional


def build_arm_provenance(
    *,
    arm_id: str,
    git_commit: str,
    transform_mode: str,
    generator_checkpoint_path: str,
    generator_checkpoint_hash: str,
    mu: float,
    stochastic_lambda: float,
    attacker_architecture: str,
    attacker_hyperparameters: Dict[str, Any],
    attacker_seeds_attempted: List[int],
    pair_train_path: str,
    pair_validation_path: str,
    pair_test_path: str,
    pair_train_hash: str,
    pair_validation_hash: str,
    pair_test_hash: str,
    representative_attacker_seed: Optional[int],
    representative_selection_criterion: str,
    run_states: Dict[int, str],
    near_chance_flags: Dict[int, bool],
    run_start_timestamp: str,
    run_end_timestamp: str,
    schedule_name: str,
    protocol_documents: Optional[Dict[str, str]] = None,
    frozen_artifacts: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the canonical per-arm provenance record.

    :param protocol_documents: {path: sha256} of the authoritative frozen protocol
        documents (R-12: 01_ADAPTIVE_REID_PROTOCOL.md, 01B_PROTOCOL_AMENDMENT.md).
    :param frozen_artifacts: {path: sha256} of other frozen inputs, e.g. the Top-k
        gallery/probe list (R-7).
    """
    record = {
        'arm_id': arm_id,
        'git_commit': git_commit,
        'transform_mode': str(transform_mode),
        'generator_checkpoint_path': str(generator_checkpoint_path),
        'generator_checkpoint_hash': str(generator_checkpoint_hash),
        'mu': float(mu),
        'stochastic_lambda': float(stochastic_lambda),
        'attacker_architecture': str(attacker_architecture),
        'attacker_hyperparameters': dict(attacker_hyperparameters),
        'attacker_seeds_attempted': [int(s) for s in attacker_seeds_attempted],
        'pair_train_path': str(pair_train_path),
        'pair_validation_path': str(pair_validation_path),
        'pair_test_path': str(pair_test_path),
        'pair_train_hash': str(pair_train_hash),
        'pair_validation_hash': str(pair_validation_hash),
        'pair_test_hash': str(pair_test_hash),
        'representative_attacker_seed': representative_attacker_seed,
        'representative_selection_criterion': str(representative_selection_criterion),
        'run_states': {str(k): v for k, v in run_states.items()},
        'near_chance_flags': {str(k): bool(v) for k, v in near_chance_flags.items()},
        'run_start_timestamp': str(run_start_timestamp),
        'run_end_timestamp': str(run_end_timestamp),
        'schedule_name': str(schedule_name),
        'protocol_documents': {str(p): str(h) for p, h in (protocol_documents or {}).items()},
        'frozen_artifacts': {str(p): str(h) for p, h in (frozen_artifacts or {}).items()},
    }
    if extra:
        record.update(extra)
    return record