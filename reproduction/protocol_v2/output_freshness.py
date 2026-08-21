"""Output freshness guard (G0.2 §6).

Never deletes anything and never cleans directories automatically — it only
decides whether a destination is safe to write NEW scientific output into,
and raises if not.
"""
import os

# Filenames this guard treats as "scientific result" artifacts — presence of
# any of these in a target directory means the directory is not a fresh
# destination, even if other files are absent.
_STALE_RESULT_NAMES = {
    'checkpoint_manifest.json',
    'attacker_manifest.json',
    'attacker_manifest_pilot.json',
    'train_log.jsonl',
    'epoch_metrics.csv',
    'pilot_result.json',
}


class OutputNotFreshError(RuntimeError):
    pass


def assert_fresh_output_dir(path: str, allow_empty_existing: bool = True):
    """Raise OutputNotFreshError unless `path` is safe to write fresh results into.

    Safe means: does not exist yet, OR exists and is empty, OR exists and
    contains no recognized stale result filename. Never deletes or modifies
    anything — this is a check, not a cleanup tool.
    """
    if not os.path.exists(path):
        return  # nonexistent destination is always fresh
    if not os.path.isdir(path):
        raise OutputNotFreshError('Destination exists and is not a directory: %s' % path)

    entries = os.listdir(path)
    if not entries:
        if allow_empty_existing:
            return
        raise OutputNotFreshError('Destination exists but must be nonexistent (not merely empty): %s' % path)

    stale = sorted(set(entries) & _STALE_RESULT_NAMES)
    if stale:
        raise OutputNotFreshError(
            'Destination %s already contains scientific result file(s) %s — '
            'refusing to append/overwrite. Choose a new destination.' % (path, stale))
    # Directory has other, non-result content (e.g. a .gitkeep) — treat as fresh.


def assert_no_append(existing_log_path: str):
    """Explicit guard against the 'open(path, "a")' append-to-result pattern.

    Callers of a v2 runner should invoke this before opening any log file in
    append mode, to fail closed if that file already has content.
    """
    if os.path.exists(existing_log_path) and os.path.getsize(existing_log_path) > 0:
        raise OutputNotFreshError(
            'Refusing to append to existing non-empty result log: %s' % existing_log_path)
