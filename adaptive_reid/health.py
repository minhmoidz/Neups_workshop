"""Run-health classification (STEP 2B Part 3).

classify_run_health(training_diagnostics) -> (state, near_chance)

Only objective execution failures can produce NUMERICALLY_INVALID. Performance level
must NEVER trigger exclusion. A completed run that meets the near-chance criteria stays
VALID and remains in every final estimand.

CRITICAL API DESIGN: the signature takes only a training-diagnostics dict. It has NO
parameter to receive test AUC, test predictions, or test labels, so it is impossible to
pass test-derived information into run validity.
"""

from . import constants as C


def _chain_errors(record, required_keys):
    """Return a list of Objective execution-failure descriptions, or None."""
    errors = []
    if record is None:
        return ['missing training diagnostics record']
    for key in required_keys:
        if key not in record:
            errors.append('missing required field %r' % key)
    return errors or None


def classify_run_health(training_diagnostics):
    """Classify a single restart.

    :param training_diagnostics: dict as produced by
        adaptive_reid.diagnostics.build_training_diagnostics (NO test fields).
    :return: (state, near_chance)
        state: NUMERICALLY_INVALID | VALID
        near_chance: bool, meaningful only when state == VALID.
    """
    required_keys = [
        'attacker_seed', 'transform_mode', 'mu', 'stochastic_lambda',
        'generator_checkpoint_path', 'generator_checkpoint_hash',
        'pair_train_path', 'pair_validation_path',
        'epochs_completed', 'termination_reason',
        'training_loss_per_epoch', 'validation_loss_per_epoch',
        'validation_auc_per_epoch', 'validation_accuracy_per_epoch',
        'best_validation_loss', 'best_validation_loss_epoch',
        'best_validation_auc', 'best_validation_auc_epoch',
        'any_nan_inf', 'checkpoint_exists', 'checkpoint_loadable',
        'weights_changed_from_initialization',
        'run_start_timestamp', 'run_end_timestamp',
    ]
    errors = _chain_errors(training_diagnostics, required_keys)
    if errors is not None:
        return C.NUMERICALLY_INVALID, False

    # --- Objective execution failures only -------------------------------
    if training_diagnostics['any_nan_inf']:
        return C.NUMERICALLY_INVALID, False
    if not training_diagnostics['checkpoint_exists']:
        return C.NUMERICALLY_INVALID, False
    if not training_diagnostics['checkpoint_loadable']:
        return C.NUMERICALLY_INVALID, False
    if not training_diagnostics['weights_changed_from_initialization']:
        return C.NUMERICALLY_INVALID, False
    if training_diagnostics['termination_reason'] == C.TERMINATION_INFRASTRUCTURE:
        return C.NUMERICALLY_INVALID, False
    if training_diagnostics['termination_reason'] not in (
            C.TERMINATION_EARLY_STOPPING, C.TERMINATION_EPOCH_CAP):
        return C.NUMERICALLY_INVALID, False  # illegal termination path
    if int(training_diagnostics['epochs_completed']) < 1:
        return C.NUMERICALLY_INVALID, False

    # A completed, numerically valid run is VALID regardless of performance.
    state = C.VALID
    best_loss = float(training_diagnostics['best_validation_loss'])
    best_auc = float(training_diagnostics['best_validation_auc'])

    near_chance = (
        best_loss >= C.NEAR_CHANCE_VAL_LOSS
        and best_auc <= C.NEAR_CHANCE_VAL_AUC
    )
    return state, near_chance