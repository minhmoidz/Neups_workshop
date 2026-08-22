"""P0 bridge runner — authorization-gated entry point.

Revision P0_2_1 (external source-review closeout):
- hardened approval validation (non-empty approver, tz-aware ISO-8601
  timestamp, output root bound to the locked protocol path, repo-confinement,
  no symlink escape, unknown-field rejection);
- ACTUAL artifact byte-hash verification with dependency-injected repository
  roots (synthetic-testable; never touches real artifacts in this task);
- dirty tracked worktree and runner-commit mismatch rejection;
- ImageNet weight artifact gate: refuses execution while its provenance is
  UNRESOLVED in the locked protocol;
- scientific imports remain strictly lazy; --execute still terminates with
  SCIENTIFIC_LOOP_NOT_IMPLEMENTED after every gate passes.

Importing this module has NO side effects. No bypass flags exist.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROTOCOL_PATH = os.path.join(HERE, "protocol_v1.json")

ACTIVE_PROCESS_PATTERN = "run_hardened_verifier"
EXPECTED_SCHEMA = "P0_PROTOCOL_V1_1"

REQUIRED_APPROVAL_FIELDS = (
    "authorization_status",
    "approved_protocol_sha256",
    "approved_runner_commit",
    "approved_screen_or_full_stage",
    "approved_seed_list",
    "approved_generator_roles_and_hashes",
    "approved_pair_hashes",
    "approved_SEOI",
    "approved_output_root",
    "approved_by",
    "approval_timestamp",
    "active_process_clearance",
)

FORBIDDEN_FLAGS = {
    "--force", "--skip-hash", "--ignore-mismatch", "--replace-seed",
    "--flip-score",
}


def canonical_protocol_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def load_protocol():
    with open(PROTOCOL_PATH, "rb") as fh:
        raw = fh.read()
    protocol = json.loads(raw.decode("utf-8"))
    return protocol, hashlib.sha256(
        canonical_protocol_bytes(protocol)).hexdigest()


def runner_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def worktree_is_clean_tracked():
    """True iff no tracked file differs from the current HEAD."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                         cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return False
    return out.stdout.strip() == ""


def active_training_process_running():
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if ACTIVE_PROCESS_PATTERN in line and "grep" not in line:
            return True, line.strip()
    return False, None


# ---------------------------------------------------------------------------
# Artifact byte verification (dependency-injected roots; synthetic-testable)
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_artifact_sha(repo_root, relative_path, expected_sha256):
    """Verify actual bytes of one artifact under an injected repository root.

    Raises PermissionError on absence or hash mismatch. Never downloads.
    """
    real = os.path.realpath(os.path.join(repo_root, relative_path))
    if not real.startswith(os.path.realpath(repo_root) + os.sep):
        raise PermissionError("artifact escapes repository: %r" % relative_path)
    if not os.path.isfile(real) or os.path.islink(real):
        raise PermissionError("artifact missing or is a symlink: %s"
                              % relative_path)
    actual = sha256_file(real)
    if actual != expected_sha256:
        raise PermissionError(
            "artifact hash mismatch for %s: %s != %s"
            % (relative_path, actual, expected_sha256))
    return True


def verify_all_artifacts(protocol, repo_root):
    """Byte-verify generators, pair files, and the ImageNet weight artifact."""
    for arm_spec in protocol["arms"].values():
        verify_artifact_sha(repo_root, arm_spec["generator_path"],
                            arm_spec["generator_sha256"])
    for kind, spec in protocol.get("pair_files", {}).items():
        verify_artifact_sha(repo_root, spec["path"], spec["sha256"])
    weights = protocol.get("imagenet_weight_artifact", {})
    if weights.get("status") != "LOCKED":
        raise PermissionError(
            "ImageNet weight artifact is %r: refusing execution until a "
            "human-approved local artifact (identifier + SHA-256) is locked "
            "in the protocol" % weights.get("status"))
    verify_artifact_sha(repo_root, weights["local_path"],
                        weights["sha256"])


# ---------------------------------------------------------------------------
# Approval-gate helpers
# ---------------------------------------------------------------------------

def _validate_approver(approved_by):
    if not isinstance(approved_by, str):
        raise PermissionError("approved_by must be a string")
    normalized = approved_by.strip()
    if not normalized:
        raise PermissionError("approved_by must be non-empty")
    return normalized


def _validate_timestamp(approval_timestamp):
    if not isinstance(approval_timestamp, str):
        raise PermissionError("approval_timestamp must be a string")
    try:
        parsed = datetime.datetime.fromisoformat(approval_timestamp)
    except ValueError:
        raise PermissionError(
            "approval_timestamp is not valid ISO-8601: %r"
            % approval_timestamp)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PermissionError(
            "approval_timestamp must be timezone-aware")
    return parsed


def _validate_output_root(protocol, approved_output_root):
    locked_root = protocol["output_contract"]["root"]
    if approved_output_root != locked_root:
        raise PermissionError(
            "approved_output_root %r does not equal the protocol-locked "
            "output root %r" % (approved_output_root, locked_root))
    resolved = os.path.realpath(os.path.join(REPO_ROOT, approved_output_root))
    if not resolved.startswith(os.path.realpath(REPO_ROOT) + os.sep):
        raise PermissionError("output root escapes the repository")
    parts = approved_output_root.replace("\\", "/").split("/")
    if any(p == ".." for p in parts) or approved_output_root.startswith("/"):
        raise PermissionError("output root contains traversal or is absolute")
    probe = os.path.join(REPO_ROOT, locked_root)
    ancestor = probe
    while ancestor != os.path.dirname(ancestor):
        if os.path.islink(ancestor):
            raise PermissionError("symlink escape in output root: %s" % ancestor)
        ancestor = os.path.dirname(ancestor)
    return resolved


def load_and_check_approval(approval_path):
    """All gates BEFORE any scientific import. Raises on any failure."""
    if not approval_path or not os.path.exists(approval_path):
        raise PermissionError("execution refused: missing execution manifest")
    if os.path.islink(approval_path):
        raise PermissionError("execution manifest must not be a symlink")
    with open(approval_path, "rb") as fh:
        approval = json.loads(fh.read().decode("utf-8"))

    known = set(REQUIRED_APPROVAL_FIELDS)
    unknown = sorted(set(approval.keys()) - known)
    if unknown:
        raise PermissionError("unknown approval fields: %s" % unknown)
    missing = sorted(known - set(approval.keys()))
    if missing:
        raise PermissionError("approval fields missing: %s" % missing)

    if approval["authorization_status"] != "APPROVED":
        raise PermissionError(
            "execution refused: authorization_status=%r"
            % approval["authorization_status"])
    _validate_approver(approval["approved_by"])
    _validate_timestamp(approval["approval_timestamp"])

    protocol, sha = load_protocol()
    if approval["approved_protocol_sha256"] != sha:
        raise PermissionError("protocol-hash mismatch")
    rc = runner_commit()
    if not rc or approval["approved_runner_commit"] != rc:
        raise PermissionError("runner-commit mismatch")

    stage = approval["approved_screen_or_full_stage"]
    if stage not in ("screen", "full"):
        raise PermissionError("stage mismatch: %r" % stage)
    expected_seeds = protocol["seeds"][stage]
    if sorted(approval["approved_seed_list"]) != sorted(expected_seeds):
        raise PermissionError("seed-list mismatch vs locked protocol")
    locked_roles = {a: s["generator_sha256"] for a, s in protocol["arms"].items()}
    if dict(approval["approved_generator_roles_and_hashes"]) != locked_roles:
        raise PermissionError("generator role/checkpoint hash changed")
    locked_pairs = {k: v["sha256"] for k, v in protocol["pair_files"].items()}
    if dict(approval["approved_pair_hashes"]) != locked_pairs:
        raise PermissionError("pair hashes changed")
    if float(approval["approved_SEOI"]) != float(
            protocol["statistics"]["seoi"]):
        raise PermissionError("SEOI changed without new ratification")
    if approval["active_process_clearance"] is not True:
        raise PermissionError("active_process_clearance is not true")
    _validate_output_root(protocol, approval["approved_output_root"])

    running, cmdline = active_training_process_running()
    if running:
        raise PermissionError(
            "refusing to start: an active %s process exists (%s)"
            % (ACTIVE_PROCESS_PATTERN, cmdline))

    # Environment and byte-level gates LAST: they require a clean tracked
    # worktree and the actual governed artifacts.
    if not worktree_is_clean_tracked():
        raise PermissionError(
            "dirty tracked worktree: tracked files differ from the approved "
            "runner commit")
    verify_all_artifacts(protocol, REPO_ROOT)
    return protocol, sha, stage, approval


# ---------------------------------------------------------------------------
# Validate-only mode
# ---------------------------------------------------------------------------

def _check_seed_derivation(protocol):
    sys.path.insert(0, HERE)
    from seed_contract import derive_seed, DOMAINS
    for arm_spec in protocol["arms"].values():
        int(arm_spec["generator_sha256"], 16)
    for seed in set(protocol["seeds"]["screen"]) | set(protocol["seeds"]["full"]):
        vals = {derive_seed(seed, d) for d in DOMAINS}
        if len(vals) != len(DOMAINS):
            raise ValueError("domain collision at seed %d" % seed)
    return True


def _check_identities(protocol):
    full = protocol["seeds"]["full"]
    screen = protocol["seeds"]["screen"]
    if sorted(screen) != sorted(set(screen)) or len(screen) != 5:
        raise ValueError("screen seed list invalid")
    if full != list(range(42, 68)):
        raise ValueError("full seed list must be integers 42..67")
    if not set(screen).issubset(set(full)):
        raise ValueError("screen seeds must be a subset of full seeds")
    return True


def _check_path_syntax(protocol):
    rel = re.compile(r"^(?!/)(?!.*\.\.)[\w./\-]+$")
    paths = [spec.get("generator_path", spec.get("path"))
             for spec in protocol["arms"].values()]
    paths += [spec["path"] for spec in protocol["pair_files"].values()]
    weights = protocol.get("imagenet_weight_artifact") or {}
    if weights.get("status") == "LOCKED":
        paths.append(weights["local_path"])
    for candidate in paths:
        if not isinstance(candidate, str) or not rel.match(candidate):
            raise ValueError("unsafe path syntax: %r" % (candidate,))
    return True


def validate_protocol_only():
    protocol, sha = load_protocol()
    if protocol.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("wrong protocol schema: %r"
                         % protocol.get("schema_version"))
    _check_seed_derivation(protocol)
    _check_identities(protocol)
    _check_path_syntax(protocol)
    superseded = protocol.get("supersedes_protocol_sha256")
    if not isinstance(superseded, str) or len(superseded) != 64:
        raise ValueError("supersedes_protocol_sha256 missing/invalid")
    return {"protocol_sha256": sha,
            "authorization_status": protocol["authorization_status"],
            "cuda_touched": False,
            "real_data_opened": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="run_p0_bridge")
    parser.add_argument("--validate-protocol-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-manifest", type=str, default=None)
    args = parser.parse_args(argv)

    unknown = set(argv) & FORBIDDEN_FLAGS
    if unknown:
        raise SystemExit("prohibited flags present: %s" % sorted(unknown))

    if args.validate_protocol_only and not args.execute:
        print(json.dumps(validate_protocol_only(), indent=1))
        return 0

    if args.execute:
        load_and_check_approval(args.execution_manifest)
        # Every authorization, provenance and byte-artifact gate has passed.
        # The scientific loop remains unimplemented by design.
        raise SystemExit("SCIENTIFIC_LOOP_NOT_IMPLEMENTED")

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
