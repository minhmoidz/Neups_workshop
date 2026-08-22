"""Immutable per-run manifests and protocol-aware deterministic aggregation.

Revision P0_2_2 (manifest-artifact integrity hotfix):
- manifests are BOUND TO ACTUAL OUTPUT BYTES: predictions.parquet and
  attacker_best.pth must exist as real regular files, their recomputed
  SHA-256 must match the manifest, and outputs must not be modified after
  manifest creation (mtime ordering);
- the full paired-order contract is validated: exact locked domain-key set in
  derived_seeds; non-empty epoch_order_hashes with exact epoch keys
  0..stop_epoch; lowercase-hex64 values; hashes RECOMPUTED from the locked
  seed + epoch + sampler schema + auditable TRAIN pair count;
- the runs-root output-tree identity set is closed: undeclared arm/seed
  directories and non-directory/symlink entries fail closed;
- exclusive fresh-run-directory claim and atomic fsync'd writes retained.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

RUN_MANIFEST_NAME = "run_manifest.json"
AGGREGATE_NAME = "aggregate_manifest.json"
PREDICTIONS_NAME = "predictions.csv"
ATTACKER_CKPT_NAME = "attacker_best.pth"
REQUIRED_OUTPUT_FILES = frozenset(
    {PREDICTIONS_NAME, ATTACKER_CKPT_NAME, RUN_MANIFEST_NAME})

RUN_MANIFEST_SCHEMA = "P0_RUN_MANIFEST_V1_2"
AGGREGATE_SCHEMA = "P0_AGGREGATE_MANIFEST_V1_2"

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


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
        pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require_hex64(value, what, identity):
    if not isinstance(value, str) or not HEX64_RE.match(value):
        raise ManifestError(
            "%s must be lowercase 64-char hex at %r, got %r"
            % (what, identity, value))
    return value


def claim_run_directory(runs_root, arm, seed):
    """Exclusively create a FRESH <runs_root>/<arm>/<seed> directory.

    Rejects: existing directory of any kind (including empty), symlinks,
    pre-existing predictions/checkpoints/manifests/partial temp files.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ManifestError("seed must be a non-negative plain int")
    arm_dir = os.path.join(runs_root, str(arm))
    run_dir = os.path.join(arm_dir, str(seed))
    if os.path.islink(run_dir) or os.path.exists(run_dir):
        raise ManifestError(
            "fresh-output violation: run directory already exists: %s" % run_dir)
    os.makedirs(arm_dir, exist_ok=True)
    try:
        os.mkdir(run_dir)
    except FileExistsError:
        raise ManifestError(
            "fresh-output violation: concurrent claim of %s" % run_dir)
    entries = set(os.listdir(run_dir))
    dirty = entries & REQUIRED_OUTPUT_FILES
    if dirty or any(e.endswith(".tmp") or e.startswith(".tmp") or
                    e.startswith(".atomic") for e in entries):
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


def _validate_output_tree_and_bind_bytes(run_dir, manifest, identity):
    """Require exact scientific output files bound to actual bytes."""
    if os.path.islink(run_dir) or not os.path.isdir(run_dir):
        raise ManifestError("run directory missing/symlink/non-dir at %r"
                            % (identity,))
    entries = set(os.listdir(run_dir))
    missing = sorted(REQUIRED_OUTPUT_FILES - entries)
    if missing:
        # CE1 regression: aggregate without predictions/checkpoint bytes
        raise ManifestError(
            "required output files missing at %r: %s" % (identity, missing))
    extra = sorted(entries - REQUIRED_OUTPUT_FILES)
    partial = [e for e in extra
               if e.endswith(".tmp") or e.startswith(".tmp")
               or e.startswith(".atomic")]
    if partial:
        raise ManifestError("partial/temp output files present at %r: %s"
                            % (identity, sorted(partial)))
    if extra:
        raise ManifestError("undeclared output entries present at %r: %s"
                            % (identity, sorted(extra)))
    for fname in (PREDICTIONS_NAME, ATTACKER_CKPT_NAME):
        fpath = os.path.join(run_dir, fname)
        if os.path.islink(fpath) or not os.path.isfile(fpath):
            raise ManifestError("%s is a symlink or not a regular file at %r"
                                % (fname, identity))

    # byte binding: recompute from ACTUAL bytes
    pred_sha = sha256_file(os.path.join(run_dir, PREDICTIONS_NAME))
    attk_sha = sha256_file(os.path.join(run_dir, ATTACKER_CKPT_NAME))
    _require_hex64(manifest["predictions_sha256"], "predictions_sha256",
                   identity)
    _require_hex64(manifest["attacker_best_sha256"], "attacker_best_sha256",
                   identity)
    if pred_sha != manifest["predictions_sha256"]:
        raise ManifestError(
            "predictions bytes do not match predictions_sha256 at %r"
            % (identity,))
    if attk_sha != manifest["attacker_best_sha256"]:
        raise ManifestError(
            "attacker checkpoint bytes do not match attacker_best_sha256 "
            "at %r" % (identity,))

    # post-manifest modification rejection (mtime ordering)
    man_mtime = os.stat(os.path.join(run_dir, RUN_MANIFEST_NAME)).st_mtime
    for fname in (PREDICTIONS_NAME, ATTACKER_CKPT_NAME):
        out_mtime = os.stat(os.path.join(run_dir, fname)).st_mtime
        if out_mtime > man_mtime + 1e-6:
            raise ManifestError(
                "%s modified after run-manifest creation at %r"
                % (fname, identity))


def _validate_paired_order_contract(m, protocol, identity):
    """Full paired-order contract validation incl. recomputation."""
    seed = m["master_seed"]
    stop_epoch = m["stop_epoch"]
    derived = m["derived_seeds"]
    if not isinstance(derived, dict):
        raise ManifestError("derived_seeds must be an object at %r" % (identity,))
    locked_domains = set(protocol["attacker_protocol"]["domains"])
    if set(derived.keys()) != locked_domains:
        raise ManifestError(
            "derived_seeds key set mismatch at %r: missing=%s extra=%s"
            % (identity,
               sorted(locked_domains - set(derived)),
               sorted(set(derived) - locked_domains)))
    eoh = m["epoch_order_hashes"]
    if not isinstance(eoh, dict) or not eoh:
        # CE2 regression: empty order-hash mapping
        raise ManifestError("epoch_order_hashes empty or invalid at %r"
                            % (identity,))
    expected_keys = {str(e) for e in range(0, stop_epoch + 1)}
    if set(eoh.keys()) != expected_keys:
        raise ManifestError(
            "epoch keys mismatch at %r: missing=%s extra=%s"
            % (identity,
               sorted(expected_keys - set(eoh)),
               sorted(set(eoh) - expected_keys)))
    train_count = protocol.get("pair_files", {}).get("train", {}).get(
        "pair_count")
    if not isinstance(train_count, int) or train_count <= 0:
        raise ManifestError(
            "locked protocol lacks auditable TRAIN pair_count; refusing to "
            "recompute expected order hashes")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from deterministic_sampler import build_permutation, order_hash
    for e in range(0, stop_epoch + 1):
        _require_hex64(eoh[str(e)], "order_hash[%d]" % e, identity)
        expected = order_hash(build_permutation(seed, e, train_count),
                              seed, e, train_count)
        if eoh[str(e)] != expected:
            raise ManifestError(
                "recomputed order hash mismatch at %r epoch %d" % (identity, e))


def aggregate_manifests(runs_root, protocol, protocol_sha256, runner_commit,
                        stage):
    """Protocol-aware, fail-closed deterministic aggregation (P0_2_2)."""
    if stage == "screen":
        expected_seeds = sorted(protocol["seeds"]["screen"])
    elif stage == "full":
        expected_seeds = sorted(protocol["seeds"]["full"])
    else:
        raise ManifestError("unknown stage: %r" % stage)
    arms = sorted(protocol["arms"].keys())
    expected_count = len(arms) * len(expected_seeds)

    # closed output-tree identity set at the runs root
    root_entries = []
    if os.path.isdir(runs_root):
        root_entries = sorted(os.listdir(runs_root))
    for entry in root_entries:
        if entry not in arms:
            # CE4 regression: undeclared arm directory under runs root
            raise ManifestError("undeclared arm directory at runs root: %r"
                                % entry)

    records = []
    seen = set()
    for arm in arms:
        arm_spec = protocol["arms"][arm]
        arm_dir = os.path.join(runs_root, str(arm))
        if not os.path.isdir(arm_dir) or os.path.islink(arm_dir):
            raise ManifestError("missing/symlink arm directory: %s" % arm_dir)
        declared_seeds = {str(s) for s in expected_seeds}
        for entry in sorted(os.listdir(arm_dir)):
            if entry not in declared_seeds:
                raise ManifestError("undeclared seed directory %r under %s"
                                    % (entry, arm))
            ep = os.path.join(arm_dir, entry)
            if os.path.islink(ep) or not os.path.isdir(ep):
                raise ManifestError(
                    "non-directory/symlink seed entry %r under %s"
                    % (entry, arm))
        for seed in expected_seeds:
            identity = (str(arm), int(seed))
            if identity in seen:
                raise ManifestError("duplicate internal identity: %r"
                                    % (identity,))
            seen.add(identity)
            run_dir = os.path.join(arm_dir, str(seed))
            m = load_run_manifest(run_dir)

            for field in REQUIRED_IDENTITY_FIELDS:
                if field not in m:
                    raise ManifestError("run manifest missing field %s at %r"
                                        % (field, (identity,)))
            if m["arm"] != arm:
                raise ManifestError(
                    "identity mismatch: directory arm %r vs manifest arm %r "
                    "at %r" % (arm, m["arm"], (identity,)))
            if m["master_seed"] != int(seed):
                raise ManifestError(
                    "identity mismatch: directory seed %d vs manifest seed %r "
                    "at %r" % (seed, m["master_seed"], (identity,)))
            if m["schema_version"] != RUN_MANIFEST_SCHEMA:
                raise ManifestError(
                    "wrong run-manifest schema at %r" % ((identity,),))
            if m["protocol_schema"] != protocol["schema_version"]:
                raise ManifestError("stale protocol schema at %r"
                                    % ((identity,),))
            if m["protocol_sha256"] != protocol_sha256:
                raise ManifestError("stale protocol hash at %r"
                                    % ((identity,),))
            if m["runner_commit"] != runner_commit:
                raise ManifestError("runner-commit mismatch at %r"
                                    % ((identity,),))
            if m["generator_role"] != arm or \
                    m["generator_sha256"] != arm_spec["generator_sha256"]:
                raise ManifestError("generator role/hash mismatch at %r"
                                    % ((identity,),))
            if m["train_pair_sha256"] != \
                    protocol["pair_files"]["train"]["sha256"]:
                raise ManifestError("TRAIN pair hash mismatch at %r"
                                    % ((identity,),))
            if m["val_pair_sha256"] != \
                    protocol["pair_files"]["val"]["sha256"]:
                raise ManifestError("VAL pair hash mismatch at %r"
                                    % ((identity,),))
            _require_hex64(m["initial_attacker_state_hash"],
                           "initial_attacker_state_hash", (identity,))
            _validate_paired_order_contract(m, protocol, identity)
            _validate_output_tree_and_bind_bytes(run_dir, m, identity)
            if m["score_direction"] != \
                    protocol["attacker_protocol"]["score_direction"]:
                raise ManifestError("score direction mismatch at %r"
                                    % ((identity,),))
            if not (0 <= m["best_epoch"] <= m["stop_epoch"]
                    <= protocol["attacker_protocol"]["max_epochs"] - 1):
                raise ManifestError("invalid best/stop epochs at %r"
                                    % ((identity,),))
            if m["status"] != "COMPLETE":
                raise ManifestError("incomplete status at %r" % ((identity,),))
            if not m.get("started_utc") or not m.get("finished_utc"):
                raise ManifestError("missing timestamps at %r"
                                    % ((identity,),))
            if not isinstance(m.get("environment_provenance"), dict) or \
                    not m["environment_provenance"]:
                raise ManifestError("missing environment provenance at %r"
                                    % ((identity,),))
            records.append((identity, m))

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
