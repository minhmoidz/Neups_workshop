"""Staged pipeline orchestration (STEP 2B Part 6).

Critical: no test-derived representative selection.

    STAGE A: train + validate all required attacker restarts (persist training
             diagnostics, checkpoint, and validation curves).
    STAGE B: classify numerical validity + compute near-chance flags.
    STAGE C: select representative attacker using validation statistics ONLY.
    STAGE D: persist representative-attacker identity in provenance.
    STAGE E: evaluate frozen test pairs once per completed attacker.

Only the FINAL per-attacker test evaluation happens in Stage E, after the
representative is frozen in Stage D. Legacy scripts are untouched.
"""

from typing import Any, Callable, Dict, List, Optional

from . import constants as C
from . import selection as sel
from . import summary as summ


class ArmPipeline:
    def __init__(self, *, train_validate_and_persist: Callable[[int], Dict[str, Any]],
                 evaluate_test: Optional[Callable[[int], Dict[str, Any]]] = None,
                 select_representative_from_validation: Optional[Callable[[List[Dict]], int]] = None):
        """Configure the pipeline with injectable workers (pure, unit-testable).

        :param train_validate_and_persist: callable(seed) -> dict that trains and
            validates ONE restart and persists its diagnostics; must return a dict with
            at least 'attacker_seed', 'diagnostics', 'validation_record'.
        :param evaluate_test: callable(seed) -> test_metrics dict (invoked only in
            Stage E). May be None when the caller only wants stages A-D.
        :param select_representative_from_validation: callable(valid_validation_records)
            -> seed. Defaults to adaptive_reid.selection.select_representative.
        """
        self.train_validate_and_persist = train_validate_and_persist
        self.evaluate_test = evaluate_test
        self.select_representative = (select_representative_from_validation
                                      or sel.select_representative)

    # -- STAGE A ---------------------------------------------------------------
    def stage_a_train_all(self, attempts) -> List[Dict]:
        """Train/validate every attempted restart; return the persisted records."""
        out = []
        for attempt in attempts:
            info = self.train_validate_and_persist(attempt['attacker_seed'])
            attempt['validation_record'] = info['validation_record']
            attempt['diagnostics'] = info['diagnostics']
            attempt['state'] = info['state']
            attempt['near_chance'] = info['near_chance']
            out.append(attempt)
        return out

    # -- STAGE B ---------------------------------------------------------------
    def stage_b_classify(self, attempts) -> List[Dict]:
        """Classify each completed restart (state + near_chance), in place."""
        for attempt in attempts:
            if 'state' not in attempt or 'near_chance' not in attempt:
                raise ValueError('attempt %s not classified' % attempt['attacker_seed'])
        return attempts

    # -- STAGE C ---------------------------------------------------------------
    def stage_c_select_representative(self, attempts) -> int:
        valid = [a for a in attempts if a['state'] == C.VALID]
        records = [a['validation_record'] for a in valid]
        return int(self.select_representative(records))

    # -- STAGE D ---------------------------------------------------------------
    def stage_d_persist_representative(self, attempts, representative_seed,
                                       mark: Dict[int, bool]) -> None:
        """Record the representative identity (validation-only). No test metrics here."""
        for a in attempts:
            a['is_representative'] = (a['attacker_seed'] == representative_seed)
            mark[a['attacker_seed']] = a['is_representative']

    # -- STAGE E ---------------------------------------------------------------
    def stage_e_evaluate_test(self, attempts) -> List[Dict]:
        """Evaluate frozen test pairs once for each completed attacker (Stage E only)."""
        if self.evaluate_test is None:
            raise RuntimeError('evaluate_test worker not configured (Stage E disabled)')
        for a in attempts:
            if a['state'] != C.VALID:
                a['test_metrics'] = None
                continue
            a['test_metrics'] = self.evaluate_test(a['attacker_seed'])
        return attempts

    # -- Full run --------------------------------------------------------------
    def run(self, attempts) -> Dict:
        self.stage_a_train_all(attempts)
        self.stage_b_classify(attempts)
        representative_seed = self.stage_c_select_representative(attempts)
        mark: Dict[int, bool] = {}
        self.stage_d_persist_representative(attempts, representative_seed, mark)
        # representative selection is frozen BEFORE any test evaluation
        self.stage_e_evaluate_test(attempts)
        summary = summ.summarize_arm(attempts)
        summary['representative_attacker_seed'] = representative_seed
        return summary