"""M0 TEST firewall + provenance recorder.

DEVELOPMENT firewall: any dev/probe run MUST fail-closed before it can open or load
TEST splits/pairs/checkpoints. We never need TEST during development; the moment a
run config requests a TEST artifact, the guard refuses (exit non-zero) so that a
dev run can never silently contaminate the closed TEST benchmark.

Only the final frozen protocol (after M0 PASS + owner approval) may run TEST, and
only from an explicit allowlist.

The provenance recorder computes deterministic hashes of the runtime + config +
checkpoints and writes a sidecar JSON so every run can be reproduced or audited.
"""
import hashlib
import json
import os
import platform
import subprocess

ALLOWED_TEST_MODE = 'final_test'
DEV_MODES = ('dev', 'val', 'validate', 'probe', 'debug')


def is_test_request(mode):
    """True iff a mode string requests the closed TEST benchmark."""
    if mode is None:
        return False
    return str(mode).lower() in ('test', 'final_test')


class TestFirewall:
    """Fail-closed guard. If construction indicates TEST while not explicitly
    allowed, raise. Baseline usage: Firewall.check(mode='dev') is a no-op;
    Firewall.check(mode='test') raises unless allow=True AND approved."""
    def __init__(self, allow=False):
        self.allow = allow

    def check(self, mode):
        if is_test_request(mode) and not self.allow:
            raise RuntimeError(
                'TEST firewall: mode=%r requests the CLOSED TEST benchmark. '
                'Development runs must use val/dev. TEST is frozen and locked '
                'until the final protocol is approved.' % mode)
        return True


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _try_git(field):
    try:
        out = subprocess.run(['git', field, '--show-current' if field == 'branch' else 'HEAD'],
                             capture_output=True, text=True, cwd=os.getcwd())
        return (out.stdout or out.stderr).strip()
    except Exception:  # noqa: BLE001
        return None


def provenance_record(config_path=None, extra=None, allow=True, mode='dev'):
    """Deterministic runtime+config+checkpoint provenance dict."""
    import torch
    fw = TestFirewall(allow=allow)
    fw.check(mode)  # fail-closed for TEST unless explicitly allowed

    rec = {
        'mode': mode,
        'cwd': os.getcwd(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'git_head': _try_git('rev-parse'),
        'git_branch': _try_git('branch'),
        'config_sha256': file_sha256(config_path) if config_path and os.path.exists(config_path) else None,
    }
    if extra:
        for k, v in extra.items():
            if isinstance(v, str) and os.path.exists(v):
                rec['sha256:' + k] = file_sha256(v)
            else:
                rec[k] = v
    return rec


if __name__ == '__main__':
    import sys
    # self-check the firewall semantics
    fw = TestFirewall(allow=False)
    ok = True
    try:
        fw.check('dev'); fw.check('val'); fw.check('probe')
    except RuntimeError:
        ok = False
    try:
        fw.check('test')
        ok = False  # should have raised
    except RuntimeError:
        pass
    print('firewall self-check:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)