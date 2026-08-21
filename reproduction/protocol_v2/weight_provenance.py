"""Pretrained-weight provenance recording (G0.2 §6).

Never downloads anything. Only records identity/hash information for a
weight file the caller already has locally, and fails closed in scientific
mode if that identity cannot be established.

Addresses reproduction/reports/G0_1_PROTOCOL_REPAIR_SPEC_2026-08-21.md §10
mult 2: `models.resnet50(pretrained=True)` does not pin an exact weights
enum or file hash.
"""
import hashlib
import os


class WeightProvenanceError(RuntimeError):
    pass


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def record_weight_provenance(weight_file_path: str, weight_enum: str, torchvision_version: str,
                              architecture_identifier: str, scientific_mode: bool = True) -> dict:
    """Build a provenance record for an explicit local weight file.

    Never fetches a weight from the network. `weight_file_path` must already
    exist locally. In scientific_mode (default), every identity field is
    mandatory and the file's SHA256 is computed and required to be present;
    non-scientific/test callers may pass scientific_mode=False to build a
    partial record for tooling purposes only.
    """
    if scientific_mode:
        missing = [name for name, val in (
            ('weight_enum', weight_enum),
            ('torchvision_version', torchvision_version),
            ('architecture_identifier', architecture_identifier),
        ) if not val or not str(val).strip()]
        if missing:
            raise WeightProvenanceError(
                'Scientific weight provenance missing required field(s): %s' % missing)
        if not weight_file_path or not os.path.exists(weight_file_path):
            raise WeightProvenanceError(
                'Scientific weight provenance requires an existing local weight file, got: %r' % weight_file_path)

    record = {
        'weight_enum': weight_enum,
        'torchvision_version': torchvision_version,
        'architecture_identifier': architecture_identifier,
        'weight_file_path': os.path.abspath(weight_file_path) if weight_file_path else None,
        'weight_file_sha256': file_sha256(weight_file_path) if weight_file_path and os.path.exists(weight_file_path) else None,
    }
    if scientific_mode and record['weight_file_sha256'] is None:
        raise WeightProvenanceError('Scientific weight provenance requires a computable file SHA256')
    return record
