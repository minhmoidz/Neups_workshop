"""Restart orchestration driver (STEP 2B Part 5).

Two modes:

Screening:
    target 3 numerically valid completed runs
    initial seeds 0,1,2 ; max 2 replacements ; max 5 total attempts

Confirmatory:
    target 10 numerically valid completed runs
    initial seeds 0..9 ; max 5 infrastructure replacements ; max 15 total attempts

Replacement occurs ONLY when state == NUMERICALLY_INVALID. A completed
VALID + near_chance=True is NEVER replaced. Replacement seeds are the next integers in
ascending order after the initial seed set; the driver never lets a caller pick a
"convenient" seed after seeing outcomes. Every attempted run is recorded.
"""

from . import constants as C


class _Schedule:
    def __init__(self, target, initial_seeds, max_replacements, max_attempts, name):
        if len(initial_seeds) != target:
            raise ValueError('%s: initial seed count must equal target (%d != %d)' %
                             (name, len(initial_seeds), target))
        self.target = target
        self.initial_seeds = tuple(int(s) for s in initial_seeds)
        self.max_replacements = max_replacements
        self.max_attempts = max_attempts
        self.name = name

    def replacement_seed_generator(self):
        """Yield replacement seeds sequentially after the initial seeds, ascending."""
        s = self.initial_seeds[-1] + 1
        while True:
            yield s
            s += 1

    @property
    def attempt_cap(self):
        return self.max_attempts


class ScreeningSchedule(_Schedule):
    def __init__(self):
        super().__init__(C.SCREENING_TARGET, C.SCREENING_INITIAL_SEEDS,
                         C.SCREENING_MAX_REPLACEMENTS, C.SCREENING_MAX_ATTEMPTS, 'screening')


class ConfirmatorySchedule(_Schedule):
    def __init__(self):
        super().__init__(C.CONFIRMATORY_TARGET, C.CONFIRMATORY_INITIAL_SEEDS,
                         C.CONFIRMATORY_MAX_REPLACEMENTS, C.CONFIRMATORY_MAX_ATTEMPTS, 'confirmatory')


def run_schedule(schedule, train_and_report):
    """Orchestrate the restarts.

    :param schedule: a Schedule instance.
    :param train_and_report: callable seed -> attempts log entry (has run the restart
        and produced a record with at least 'attacker_seed' and 'state').
    :return: list of attempt records, in execution order. Includes every attempted run.
    """
    attempts = []
    replacements_used = 0
    repl_gen = schedule.replacement_seed_generator()

    def n_valid():
        return sum(1 for a in attempts if a['state'] == C.VALID)

    def _next_seed():
        """Deterministic: run all initial seeds first, then replacement seeds ascending."""
        done = [a['attacker_seed'] for a in attempts]
        for s in schedule.initial_seeds:
            if s not in done:
                return s
        # all initial seeds attempted -> feed replacements in ascending order
        return next(repl_gen)

    while len(attempts) < schedule.attempt_cap and n_valid() < schedule.target:
        seed = _next_seed()
        record = train_and_report(seed)
        record.setdefault('attacker_seed', seed)
        attempts.append(record)
        if record.get('state') == C.NUMERICALLY_INVALID:
            # only invalid runs consume a replacement slot
            if seed not in schedule.initial_seeds:
                replacements_used += 1
            if replacements_used >= schedule.max_replacements and n_valid() < schedule.target:
                # do not exceed max replacements; stop (cap will end the loop)
                break

    if n_valid() < schedule.target:
        raise RuntimeError(
            '%s: could not collect %d valid runs in %d attempts (got %d)' %
            (schedule.name, schedule.target, schedule.attempt_cap, n_valid()))
    return attempts