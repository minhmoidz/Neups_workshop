"""M0.1 — non-vacuous checkpoint-provenance hash regression test.

For every dev config, verify each configured checkpoint SHA256 against:

A) the SHA256 of the materialized checkpoint bytes when the working tree contains
   real bytes (the strong check); OR
B) the `oid sha256:` value in the Git LFS pointer when the working tree contains
   an LFS pointer rather than materialized bytes.

The test FAILS when:
- the config value differs from the verified checkpoint/LFS SHA, or
- a checkpoint path in the config does not exist (neither bytes nor pointer),
  so it can never become vacuously PASS merely because a large checkpoint is
  unavailable.
"""
import sys
import os
import json
import hashlib
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, HERE)
from m0_common import run_all  # noqa: E402

CONFIGS = [
    os.path.join(ROOT, 'config_files', 'config_dev_restored_baseline.json'),
    os.path.join(ROOT, 'config_files', 'config_dev_c4.json'),
    os.path.join(ROOT, 'config_files', 'config_dev_c2c4.json'),
]

# configured-checkpoint-key -> checkpoint filename
CHECKPOINT_KEYS = {
    'generator_checkpoint_sha256': 'generator_checkpoint_path',
    'classifier_checkpoint_sha256': 'classifier_checkpoint_path',
    'verification_model_sha256': 'verification_model_path',
}


def _file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _lfs_pointer_oid(path):
    """Read `oid sha256:` from a Git LFS pointer file, or None if not a pointer."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('oid sha256:'):
                    return line.split(':', 1)[1].strip()
    except (UnicodeDecodeError, OSError):
        return None
    return None


def _git_lfs_oid_from_index(rel_path):
    """Fall back to the oid recorded in the git index LFS pointer if present."""
    try:
        sha = subprocess.run(
            ['git', 'ls-files', '-s', rel_path], capture_output=True, text=True,
            cwd=ROOT).stdout.split()
        if not sha:
            return None
        blob = subprocess.run(
            ['git', 'cat-file', '-p', sha[0]], capture_output=True, text=True,
            cwd=ROOT).stdout
        for line in blob.splitlines():
            if line.strip().startswith('oid sha256:'):
                return line.strip().split(':', 1)[1].strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def _verify_checkpoint(cfg_path, sha_key, path_key):
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg_sha = cfg.get(sha_key)
    rel = cfg.get(path_key)
    if not cfg_sha or not rel:
        raise RuntimeError('%s missing key %s/%s' % (cfg_path, sha_key, path_key))
    abs_path = os.path.join(ROOT, rel)
    if not os.path.exists(abs_path):
        # neither materialized bytes nor a pointer file present -> FAIL (non-vacuous)
        raise RuntimeError('%s checkpoint missing: %s' % (rel, abs_path))
    # strong check A: materialized bytes
    if not _lfs_pointer_oid(abs_path):
        actual = _file_sha256(abs_path)
    else:
        # pointer file: oid from the file itself, else from git index
        actual = _lfs_pointer_oid(abs_path) or _git_lfs_oid_from_index(rel)
    if not actual:
        raise RuntimeError('%s: no materialized bytes and no LFS oid resolvable' % rel)
    if actual != cfg_sha:
        raise RuntimeError(
            '%s: config %s=%s != checkpoint %s (actual=%s)' % (rel, sha_key, cfg_sha, rel, actual))
    return True


def _provenance_check():
    for cfg_path in CONFIGS:
        for sha_key, path_key in CHECKPOINT_KEYS.items():
            _verify_checkpoint(cfg_path, sha_key, path_key)
    return True


def t_m01_all_configs_match_checkpoints():
    return _provenance_check()


if __name__ == '__main__':
    ok = run_all([
        ('M0.1 all config hashes match checkpoints', t_m01_all_configs_match_checkpoints),
    ])
    sys.exit(0 if ok else 1)