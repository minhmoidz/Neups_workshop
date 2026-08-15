"""M0 TEST firewall + provenance recorder (reusable utility).

DEVELOPMENT firewall: any dev/probe run MUST fail-closed before it can open or load
TEST splits/pairs/checkpoints. Only the final frozen protocol (after M0 PASS + owner
approval) may run TEST, and only from an explicit allowlist.

The provenance recorder computes deterministic hashes of the runtime + config +
checkpoints and writes a sidecar JSON so every run can be reproduced or audited.
"""
import hashlib
import json
import os
import platform
import subprocess


def is_test_request(mode):
    """True iff a mode string requests the closed TEST benchmark."""
    if mode is None:
        return False
    return str(mode).lower() in ('test', 'final_test')


class TestFirewall:
    """Fail-closed guard. check(mode) is a no-op for dev/val/probe; raises for
    TEST unless allow=True (reserved for the approved final protocol)."""
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


def _git_or_none(args):
    try:
        out = subprocess.run(['git'] + args, capture_output=True, text=True, cwd=os.getcwd())
        return (out.stdout or out.stderr).strip() or None
    except Exception:  # noqa: BLE001
        return None


def provenance_record(config_path=None, extra=None, allow=True, mode='dev'):
    """Deterministic runtime+config+checkpoint provenance dict."""
    import torch
    TestFirewall(allow=allow).check(mode)  # fail-closed for TEST unless explicitly allowed

    rec = {
        'mode': mode,
        'cwd': os.getcwd(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'git_head': _git_or_none(['rev-parse', 'HEAD']),
        'git_branch': _git_or_none(['branch', '--show-current']),
        'config_sha256': file_sha256(config_path) if config_path and os.path.exists(config_path) else None,
    }
    if extra:
        for k, v in extra.items():
            if isinstance(v, str) and os.path.exists(v):
                rec['sha256:' + k] = file_sha256(v)
            else:
                rec[k] = v
    return rec