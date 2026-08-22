"""Immutable per-run manifests and deterministic aggregate manifest.

One immutable run_manifest.json per <arm>/<seed>/ directory, written atomically
(temp file + flush + fsync + rename). No shared append-only JSONL.
"""
import hashlib
import json
import os
import tempfile


class ManifestError(RuntimeError):
    pass


def canonical_json_bytes(obj):
    """Deterministic JSON serialization used for hashing and writing."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def write_run_manifest_atomic(run_dir, manifest):
    """Atomically create <run_dir>/run_manifest.json; refuse pre-existing."""
    final_path = os.path.join(run_dir, "run_manifest.json")
    if os.path.exists(final_path):
        raise ManifestError("pre-existing run manifest: %s" % final_path)
    os.makedirs(run_dir, exist_ok=True)
    payload = canonical_json_bytes(manifest)
    fd, tmp_path = tempfile.mkstemp(dir=run_dir, prefix=".run_manifest.",
                                    suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp_path, final_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return final_path


def load_run_manifest(run_dir):
    path = os.path.join(run_dir, "run_manifest.json")
    if not os.path.exists(path):
        raise ManifestError("missing run manifest: %s" % path)
    with open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


REQUIRED_IDENTITY_FIELDS = (
    "protocol_schema", "protocol_sha256", "runner_commit", "arm",
    "master_seed", "derived_seeds", "generator_role", "generator_sha256",
    "train_pair_sha256", "val_pair_sha256",
    "initial_attacker_state_hash", "epoch_order_hashes",
    "best_epoch", "stop_epoch", "score_direction", "predictions_sha256",
    "environment_provenance", "status", "started_utc", "finished_utc",
)


def _validate_identity(manifest):
    for field in REQUIRED_IDENTITY_FIELDS:
        if field not in manifest:
            raise ManifestError("run manifest missing field: %s" % field)


def aggregate_manifests(runs_root, arms, seeds, expected_protocol_sha256,
                        stage):
    """Deterministically aggregate and validate all per-run manifests.

    stage: 'screen' (10 identities) or 'full' (52 identities).
    Rejects missing/duplicate/unexpected identities, stale protocol hashes,
    paired order-hash disagreement, paired initial-attacker-hash disagreement.
    """
    if stage == "screen":
        expected_seeds = sorted(seeds)[:5]
    elif stage == "full":
        expected_seeds = sorted(seeds)
    else:
        raise ManifestError("unknown stage: %r" % stage)

    records = []
    seen = set()
    for arm in arms:
        for seed in expected_seeds:
            identity = (str(arm), int(seed))
            if identity in seen:
                raise ManifestError("duplicate identity: %r" % (identity,))
            seen.add(identity)
            run_dir = os.path.join(runs_root, str(arm), str(seed))
            m = load_run_manifest(run_dir)
            _validate_identity(m)
            if m["protocol_sha256"] != expected_protocol_sha256:
                raise ManifestError("stale protocol hash in %r" % (identity,))
            records.append((identity, m))

    # extra identities on disk?
    for arm in arms:
        arm_dir = os.path.join(runs_root, str(arm))
        if not os.path.isdir(arm_dir):
            raise ManifestError("missing arm directory: %s" % arm_dir)
        for entry in os.listdir(arm_dir):
            if (str(arm), int(entry)) not in seen:
                raise ManifestError("unexpected identity: %r" % ((arm, entry),))

    by_seed = {}
    for (arm, seed), m in records:
        by_seed.setdefault(seed, {})[arm] = m

    for seed, per_arm in sorted(by_seed.items()):
        if len(per_arm) != len(arms):
            continue  # missing handled above only for full grid; keep explicit
        refs = list(per_arm.values())
        order_ref = refs[0]["epoch_order_hashes"]
        init_ref = refs[0]["initial_attacker_state_hash"]
        for m in refs[1:]:
            if m["epoch_order_hashes"] != order_ref:
                raise ManifestError(
                    "paired order-hash mismatch at seed %d" % seed)
            if m["initial_attacker_state_hash"] != init_ref:
                raise ManifestError(
                    "paired initial-attacker-hash mismatch at seed %d" % seed)

    aggregate = {
        "stage": stage,
        "expected_protocol_sha256": expected_protocol_sha256,
        "identities": [
            {"arm": arm, "seed": seed,
             "manifest_sha256": hashlib.sha256(
                 canonical_json_bytes(m)).hexdigest()}
            for (arm, seed), m in sorted(records, key=lambda kv: (kv[0][1], kv[0][0]))
        ],
        "count": len(records),
    }
    return aggregate


def write_aggregate_manifest_atomic(runs_root, aggregate):
    final_path = os.path.join(runs_root, "aggregate_manifest.json")
    if os.path.exists(final_path):
        raise ManifestError("pre-existing aggregate manifest: %s" % final_path)
    payload = canonical_json_bytes(aggregate)
    fd, tmp_path = tempfile.mkstemp(dir=runs_root, prefix=".aggregate.",
                                    suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp_path, final_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return final_path
