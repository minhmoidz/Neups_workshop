"""Immutable per-run manifests and protocol-aware deterministic aggregation.

Revision P0_2_1 (external source-review closeout):
- exclusive fresh-run-directory claim BEFORE any output is written;
- atomic writes with fsync of file AND containing directory;
- aggregator derives arms/seeds from the LOCKED PROTOCOL itself (never from
  caller-supplied lists) and validates every manifest field against it;
- exact identity counts: screen = 2 arms x 5 seeds = 10; full = 52;
- directory/manifest identity binding; stale/dirty/incomplete rejection.
"""
import hashlib
import json
import os
import shutil
import tempfile

RUN_MANIFEST_NAME = "run_manifest.json"
AGGREGATE_NAME = "aggregate_manifest.json"
PREDICTIONS_NAME = "predictions.parquet"
ATTACKER_CKPT_NAME = "attacker_best.pth"

RUN_MANIFEST_SCHEMA = "P0_RUN_MANIFEST_V1_1"
AGGREGATE_SCHEMA = "P0_AGGREGATE_MANIFEST_V1_1"


class ManifestError(RuntimeError):
    pass


def canonical_json_bytes(obj):
    """Deterministic JSON serialization used for hashing and writing."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # directory fsync unsupported on this platform


def claim_run_directory(runs_root, arm, seed):
    """Exclusively create a FRESH <runs_root>/<arm>/<seed> directory.

    Rejects: existing directory of any kind (including empty), symlinks,
    pre-existing predictions/checkpoints/manifests/partial temp files.
    Returns the created run directory path. The caller owns it thereafter.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ManifestError("seed must be a plain int")
    arm_dir = os.path.join(runs_root, str(arm))
    run_dir = os.path.join(arm_dir, str(seed))
    if os.path.islink(run_dir) or os.path.exists(run_dir):
        raise ManifestError(
            "fresh-output violation: run directory already exists: %s" % run_dir)
    os.makedirs(arm_dir, exist_ok=True)
    try:
        os.mkdir(run_dir)          # exclusive creation
    except FileExistsError:
        raise ManifestError(
            "fresh-output violation: concurrent claim of %s" % run_dir)
    # defensive re-scan: nothing may pre-exist inside a claimed dir
    entries = set(os.listdir(run_dir))
    forbidden = {PREDICTIONS_NAME, ATTACKER_CKPT_NAME, RUN_MANIFEST_NAME,
                 AGGREGATE_NAME}
    dirty = entries & forbidden
    if dirty or any(e.endswith(".tmp") or e.startswith(".tmp") for e in entries):
        shutil.rmtree(run_dir, ignore_errors=True)
        raise ManifestError(
            "fresh-output violation inside new run dir: %s" % sorted(dirty))
    return run_dir


def write_bytes_atomic(directory, final_name, payload):
    """Atomically create <directory>/<final_name>; refuse pre-existing."""
    final_path = os.path.join(directory, final_name)
    if os.path.lexists(final_path):
        raise ManifestError("pre-existing output: %s" % final_path)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_",
                                    suffix=".atomic")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp_path, final_path)
        _fsync_dir(directory)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    return final_path


def write_run_manifest_atomic(run_dir, manifest):
    os.makedirs(run_dir, exist_ok=True)
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise ManifestError(
            "run manifest schema must be %s" % RUN_MANIFEST_SCHEMA)
    payload = canonical_json_bytes(manifest)
    manifest["manifest_payload_sha256"] = hashlib.sha256(payload).hexdigest()
    payload = canonical_json_bytes(manifest)
    return write_bytes_atomic(run_dir, RUN_MANIFEST_NAME, payload)


def load_run_manifest(run_dir):
    path = os.path.join(run_dir, RUN_MANIFEST_NAME)
    if not os.path.exists(path):
        raise ManifestError("missing run manifest: %s" % path)
    with open(path, "rb") as fh:
        m = json.loads(fh.read().decode("utf-8"))
    recorded = m.pop("manifest_payload_sha256", None)
    if recorded != hashlib.sha256(canonical_json_bytes(m)).hexdigest():
        raise ManifestError("run manifest payload hash mismatch in %s" % path)
    m["manifest_payload_sha256"] = recorded
    return m


REQUIRED_IDENTITY_FIELDS = (
    "schema_version", "protocol_schema", "protocol_sha256", "runner_commit",
    "arm", "master_seed", "derived_seeds", "generator_role",
    "generator_sha256", "train_pair_sha256", "val_pair_sha256",
    "initial_attacker_state_hash", "epoch_order_hashes",
    "attacker_best_sha256", "best_epoch", "stop_epoch", "score_direction",
    "predictions_sha256", "environment_provenance", "status",
    "started_utc", "finished_utc",
)


def aggregate_manifests(runs_root, protocol, protocol_sha256, runner_commit,
                        stage):
    """Protocol-aware, fail-closed deterministic aggregation.

    Arms and seeds are derived FROM THE LOCKED PROTOCOL — caller lists are not
    accepted. screen = exactly 2 arms x 5 seeds = 10 identities;
    full = exactly 2 arms x 26 seeds = 52.
    """
    if stage == "screen":
        expected_seeds = sorted(protocol["seeds"]["screen"])
    elif stage == "full":
        expected_seeds = sorted(protocol["seeds"]["full"])
    else:
        raise ManifestError("unknown stage: %r" % stage)
    arms = sorted(protocol["arms"].keys())
    expected_count = len(arms) * len(expected_seeds)

    records = []
    seen = set()
    for arm in arms:
        arm_spec = protocol["arms"][arm]
        for seed in expected_seeds:
            identity = (str(arm), int(seed))
            if identity in seen:
                raise ManifestError("duplicate internal identity: %r"
                                    % (identity,))
            seen.add(identity)
            run_dir = os.path.join(runs_root, str(arm), str(seed))
            m = load_run_manifest(run_dir)

            for field in REQUIRED_IDENTITY_FIELDS:
                if field not in m:
                    raise ManifestError("run manifest missing field %s at %r"
                                        % (field, identity))
            # directory/manifest identity binding
            if m["arm"] != arm:
                raise ManifestError(
                    "identity mismatch: directory arm %r vs manifest arm %r "
                    "at %r" % (arm, m["arm"], identity))
            if m["master_seed"] != int(seed):
                raise ManifestError(
                    "identity mismatch: directory seed %d vs manifest seed %r "
                    "at %r" % (seed, m["master_seed"], identity))
            # protocol binding
            if m["schema_version"] != RUN_MANIFEST_SCHEMA:
                raise ManifestError("wrong run-manifest schema at %r"  %(identity,))
            if m["protocol_schema"] != protocol["schema_version"]:
                raise ManifestError("stale protocol schema at %r"  %(identity,))
            if m["protocol_sha256"] != protocol_sha256:
                raise ManifestError("stale protocol hash at %r"  %(identity,))
            if m["runner_commit"] != runner_commit:
                raise ManifestError("runner-commit mismatch at %r"  %(identity,))
            if m["generator_role"] != arm or \
                    m["generator_sha256"] != arm_spec["generator_sha256"]:
                raise ManifestError("generator role/hash mismatch at %r"
                                     %(identity,))
            if m["train_pair_sha256"] != \
                    protocol["pair_files"]["train"]["sha256"]:
                raise ManifestError("TRAIN pair hash mismatch at %r"  %(identity,))
            if m["val_pair_sha256"] != \
                    protocol["pair_files"]["val"]["sha256"]:
                raise ManifestError("VAL pair hash mismatch at %r"  %(identity,))
            expected_bundle = {
                d: __import__("seed_contract").derive_seed(int(seed), d)
                for d in protocol["attacker_protocol"]["domains"]}
            if {k: v for k, v in m["derived_seeds"].items()
                    if k in expected_bundle} != expected_bundle:
                raise ManifestError("derived-seed bundle mismatch at %r"
                                     %(identity,))
            for hname in ("initial_attacker_state_hash", "predictions_sha256",
                          "attacker_best_sha256"):
                h = m[hname]
                if not isinstance(h, str) or len(h) != 64:
                    raise ManifestError("invalid %s at %r" % (hname, identity))
            if m["score_direction"] != \
                    protocol["attacker_protocol"]["score_direction"]:
                raise ManifestError("score direction mismatch at %r"  %(identity,))
            if not (0 <= m["best_epoch"] <= m["stop_epoch"]
                    <= protocol["attacker_protocol"]["max_epochs"] - 1):
                raise ManifestError("invalid best/stop epochs at %r"  %(identity,))
            if m["status"] != "COMPLETE":
                raise ManifestError("incomplete status at %r"  %(identity,))
            if not m.get("started_utc") or not m.get("finished_utc"):
                raise ManifestError("missing timestamps at %r"  %(identity,))
            if not isinstance(m.get("environment_provenance"), dict) or \
                    not m["environment_provenance"]:
                raise ManifestError("missing environment provenance at %r"
                                     %(identity,))
            records.append((identity, m))

    # extra identities on disk?
    for arm in arms:
        arm_dir = os.path.join(runs_root, str(arm))
        if not os.path.isdir(arm_dir):
            raise ManifestError("missing arm directory: %s" % arm_dir)
        for entry in os.listdir(arm_dir):
            try:
                entry_seed = int(entry)
            except ValueError:
                raise ManifestError("unexpected non-seed entry %r under %s"
                                    % (entry, arm_dir))
            if (str(arm), entry_seed) not in seen:
                raise ManifestError("unexpected identity: %r"
                                    % ((arm, entry_seed),))

    if len(records) != expected_count:
        raise ManifestError("identity count mismatch: got %d expected %d"
                            % (len(records), expected_count))

    by_seed = {}
    for (arm, seed), m in records:
        by_seed.setdefault(seed, {})[arm] = m

    for seed, per_arm in sorted(by_seed.items()):
        refs = list(per_arm.values())
        order_ref = refs[0]["epoch_order_hashes"]
        init_ref = refs[0]["initial_attacker_state_hash"]
        for m in refs[1:]:
            if m["epoch_order_hashes"] != order_ref:
                raise ManifestError("paired order-hash mismatch at seed %d"
                                    % seed)
            if m["initial_attacker_state_hash"] != init_ref:
                raise ManifestError(
                    "paired initial-attacker-hash mismatch at seed %d" % seed)

    aggregate = {
        "schema_version": AGGREGATE_SCHEMA,
        "stage": stage,
        "expected_protocol_sha256": protocol_sha256,
        "expected_runner_commit": runner_commit,
        "identities": [
            {"arm": arm, "seed": seed,
             "manifest_payload_sha256": m["manifest_payload_sha256"]}
            for (arm, seed), m in
            sorted(records, key=lambda kv: (kv[0][1], kv[0][0]))
        ],
        "count": len(records),
    }
    return aggregate


def write_aggregate_manifest_atomic(runs_root, aggregate):
    payload = canonical_json_bytes(aggregate)
    return write_bytes_atomic(runs_root, AGGREGATE_NAME, payload)
