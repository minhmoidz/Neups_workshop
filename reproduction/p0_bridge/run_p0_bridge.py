"""P0 bridge runner — authorization-gated entry point.

Importing this module has NO side effects and imports no scientific model.

Modes:
  --validate-protocol-only : CPU-only protocol/seed/identity validation.
      Never imports medical models, never opens pair files or images, never
      loads checkpoints, never touches CUDA, never creates result directories.
  --execute                : FAILS CLOSED unless an external human-approved
      execution manifest exists and every provenance gate passes. All
      scientific imports happen lazily AFTER the gates. The scientific training
      loop itself is NOT_EXECUTED_REQUIRES_EXTERNAL_SOURCE_REVIEW in this task.

There are deliberately NO bypass flags (--force / --skip-hash /
--ignore-mismatch / --replace-seed / --flip-score).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(HERE, "protocol_v1.json")

ACTIVE_PROCESS_PATTERN = "run_hardened_verifier"

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
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.abspath(os.path.join(HERE, "..", "..")),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def active_training_process_running():
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if ACTIVE_PROCESS_PATTERN in line and "grep" not in line:
            return True, line.strip()
    return False, None


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
    for candidate in paths:
        if not isinstance(candidate, str) or not rel.match(candidate):
            raise ValueError("unsafe path syntax: %r" % (candidate,))
    return True


def validate_protocol_only():
    protocol, sha = load_protocol()
    if protocol.get("schema_version") != "P0_PROTOCOL_V1":
        raise ValueError("wrong protocol schema")
    if protocol.get("authorization_status") != "NOT_AUTHORIZED":
        # validation mode reports but does not execute anything either way
        pass
    _check_seed_derivation(protocol)
    _check_identities(protocol)
    _check_path_syntax(protocol)
    return {"protocol_sha256": sha,
            "authorization_status": protocol["authorization_status"],
            "cuda_touched": False,
            "real_data_opened": False}


def load_and_check_approval(approval_path):
    """All gates BEFORE any scientific import. Raises on any failure."""
    if not approval_path or not os.path.exists(approval_path):
        raise PermissionError("execution refused: missing execution manifest")
    with open(approval_path, "rb") as fh:
        approval = json.loads(fh.read().decode("utf-8"))

    for field in REQUIRED_APPROVAL_FIELDS:
        if field not in approval:
            raise PermissionError("execution manifest missing field: %s" % field)
    if approval["authorization_status"] != "APPROVED":
        raise PermissionError(
            "execution refused: authorization_status=%r"
            % approval["authorization_status"])

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

    running, cmdline = active_training_process_running()
    if running:
        raise PermissionError(
            "refusing to start: an active %s process exists (%s)"
            % (ACTIVE_PROCESS_PATTERN, cmdline))

    out_root = approval["approved_output_root"]
    if os.path.exists(out_root) and os.listdir(out_root):
        raise PermissionError("output collision: non-empty output root")
    return protocol, sha, stage, approval


def main(argv=None):
    parser = argparse.ArgumentParser(prog="run_p0_bridge")
    parser.add_argument("--validate-protocol-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execution-manifest", type=str, default=None)
    args = parser.parse_args(argv)

    unknown = set(argv or []) & FORBIDDEN_FLAGS
    if unknown:
        raise SystemExit("prohibited flags present: %s" % sorted(unknown))

    if args.validate_protocol_only and not args.execute:
        print(json.dumps(validate_protocol_only(), indent=1))
        return 0

    if args.execute:
        # Every gate runs BEFORE any scientific import happens anywhere.
        protocol, sha, stage, approval = load_and_check_approval(
            args.execution_manifest)
        # NOT_EXECUTED_REQUIRES_EXTERNAL_SOURCE_REVIEW: the scientific loop is
        # intentionally absent until a separately reviewed implementation task.
        raise SystemExit(
            "scientific execution path is NOT_EXECUTED_REQUIRES_EXTERNAL_"
            "SOURCE_REVIEW; authorization gates passed but no reviewed "
            "training implementation exists in this task")

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
