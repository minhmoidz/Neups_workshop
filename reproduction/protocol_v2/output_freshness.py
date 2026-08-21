"""Output freshness guard (G0.2 §6).

Never deletes anything and never cleans directories automatically — it only
decides whether a destination is safe to write NEW scientific output into,
and raises if not.

Fix 3 (G0.2A.2): the original filename-allowlist check
(`_STALE_RESULT_NAMES`) was not genuinely fail-closed — an existing
directory containing an UNRECOGNIZED file (a stray text file, a nested
directory, an old-but-differently-named checkpoint) was silently treated as
"fresh" as long as it didn't happen to match one of a handful of known
result filenames. In scientific mode the guard must reject every non-empty
existing directory regardless of what it contains, not just directories
containing a recognized name.
"""
import os


class OutputNotFreshError(RuntimeError):
    pass


def assert_fresh_output_dir(path: str, allow_empty_existing: bool = True, scientific_mode: bool = True):
    """Raise OutputNotFreshError unless `path` is safe to write fresh results into.

    scientific_mode=True (default): safe means ONLY — does not exist yet, OR
    exists and is genuinely empty (if allow_empty_existing). ANY existing,
    non-empty directory is rejected outright, regardless of its contents'
    filenames (unknown text file, old checkpoint-like file, nested
    directory, or a recognized result file — all rejected alike). Never
    deletes or modifies anything — this is a check, not a cleanup tool.

    scientific_mode=False is not used by any scientific caller; it exists
    only so non-scientific tooling could in principle request the older,
    permissive allowlist behavior — no such caller currently exists in this
    codebase, and none should be added without a separate, explicit review.
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

    if scientific_mode:
        raise OutputNotFreshError(
            'Destination %s already exists and is non-empty (contains %s) — '
            'refusing to write into any pre-existing non-empty directory in scientific mode. '
            'Choose a new destination.' % (path, sorted(entries)))

    # Non-scientific fallback path (unused by any current caller): only reject
    # directories containing one of a small set of known result filenames.
    _stale_result_names = {
        'checkpoint_manifest.json', 'attacker_manifest.json', 'attacker_manifest_pilot.json',
        'train_log.jsonl', 'epoch_metrics.csv', 'pilot_result.json',
    }
    stale = sorted(set(entries) & _stale_result_names)
    if stale:
        raise OutputNotFreshError(
            'Destination %s already contains scientific result file(s) %s — '
            'refusing to append/overwrite. Choose a new destination.' % (path, stale))


def assert_no_append(existing_log_path: str):
    """Explicit guard against the 'open(path, "a")' append-to-result pattern.

    Callers of a v2 runner should invoke this before opening any log file in
    append mode, to fail closed if that file already has content.
    """
    if os.path.exists(existing_log_path) and os.path.getsize(existing_log_path) > 0:
        raise OutputNotFreshError(
            'Refusing to append to existing non-empty result log: %s' % existing_log_path)
