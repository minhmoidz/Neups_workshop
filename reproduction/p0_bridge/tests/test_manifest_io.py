"""Standalone CPU-only manifest tests (P0_2_2 revision).

Every regression test exercises PRODUCTION code. All artifacts are small
synthetic byte files in temporary directories.
Run: CUDA_VISIBLE_DEVICES="" python test_manifest_io.py
"""
import copy
import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manifest_io import (ManifestError, aggregate_manifests,
                         claim_run_directory,
                         load_run_manifest,
                         write_aggregate_manifest_atomic,
                         write_run_manifest_atomic)
from deterministic_sampler import build_permutation, order_hash
from seed_contract import derive_seed

ARMS = ["D_BDEV", "U_PUBLISHED"]
GEN_SHA = {"U_PUBLISHED":
           "4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153",
           "D_BDEV":
           "18381d92c64bb3d646b62d5fb9d0ed8c208cf2cb3154f8aa1dac4b1baff610cd"}
TRAIN_SHA = "t" * 64
VAL_SHA = "v" * 64
PROTO_SHA = "a" * 64
RUNNER_COMMIT = "c" * 40

PROTOCOL = {
    "schema_version": "P0_PROTOCOL_V1_1",
    "seeds": {"screen": [42, 43], "full": [42, 43, 44, 45]},
    "arms": {a: {"generator_sha256": GEN_SHA[a]} for a in ARMS},
    "pair_files": {"train": {"sha256": TRAIN_SHA, "pair_count": 10000},
                   "val": {"sha256": VAL_SHA, "pair_count": 2000}},
    "attacker_protocol": {
        "domains": ["attacker_weight_init", "train_order",
                    "dataloader_worker_base", "statistical_sensitivity"],
        "score_direction": "sigmoid_logit_higher_means_same_patient_fixed_apriori",
        "max_epochs": 100},
}


def _epoch_hashes(seed, stop_epoch):
    return {str(e): order_hash(build_permutation(seed, e, 10000),
                               seed, e, 10000)
            for e in range(0, stop_epoch + 1)}


def make_valid_run(root, arm, seed, **overrides):
    """Production-path run creation: fresh claim -> output bytes -> manifest."""
    rd = claim_run_directory(root, arm, int(seed))
    pred = ("predictions-%s-%d" % (arm, seed)).encode()
    attk = ("attacker-%s-%d" % (arm, seed)).encode()
    open(os.path.join(rd, "predictions.parquet"), "wb").write(pred)
    open(os.path.join(rd, "attacker_best.pth"), "wb").write(attk)
    m = {
        "schema_version": "P0_RUN_MANIFEST_V1_2",
        "protocol_schema": PROTOCOL["schema_version"],
        "protocol_sha256": overrides.get("protocol_sha", PROTO_SHA),
        "runner_commit": overrides.get("runner_commit", RUNNER_COMMIT),
        "arm": arm,
        "master_seed": int(seed),
        "derived_seeds": copy.deepcopy(overrides.get("derived_seeds")) or {
            d: derive_seed(int(seed), d)
            for d in PROTOCOL["attacker_protocol"]["domains"]},
        "generator_role": overrides.get("generator_role", arm),
        "generator_sha256": GEN_SHA[overrides.get("generator_role", arm)],
        "train_pair_sha256": TRAIN_SHA,
        "val_pair_sha256": VAL_SHA,
        "initial_attacker_state_hash":
            overrides.get("init_hash") or
            hashlib.sha256(("init-attacker-%d" % int(seed)).encode()).hexdigest(),
        "epoch_order_hashes": overrides.get(
            "epoch_order_hashes", _epoch_hashes(int(seed), 8)),
        "attacker_best_sha256": hashlib.sha256(attk).hexdigest(),
        "best_epoch": 3,
        "stop_epoch": 8,
        "score_direction": PROTOCOL["attacker_protocol"]["score_direction"],
        "predictions_sha256": hashlib.sha256(pred).hexdigest(),
        "environment_provenance": {"torch": "cpu-test"},
        "status": overrides.get("status", "COMPLETE"),
        "started_utc": "2026-08-23T00:00:00+00:00",
        "finished_utc": "2026-08-23T01:00:00+00:00",
    }
    for k, v in overrides.items():
        if k in m and k not in ("protocol_sha", "runner_commit"):
            m[k] = v
    write_run_manifest_atomic(rd, m)
    return rd


def _grid(root, seeds, **overrides):
    for s in seeds:
        for a in ARMS:
            make_valid_run(root, a, s, **copy.deepcopy(overrides))


def _expect_merr(fn, needle=None):
    try:
        fn()
    except ManifestError as e:
        if needle is not None:
            assert needle in str(e), "message %r lacks %r" % (str(e), needle)
        return str(e)
    raise AssertionError("expected ManifestError")


def test_valid_screen_and_full_counts_with_locked_protocol():
    real = dict(PROTOCOL)
    real["seeds"] = {"screen": [42, 43, 44, 45, 46], "full": list(range(42, 68))}
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43, 44, 45, 46])
        agg = aggregate_manifests(root, real, PROTO_SHA, RUNNER_COMMIT,
                                  "screen")
        assert agg["count"] == 10               # exactly 2 arms x 5 seeds
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, list(range(42, 68)))
        agg = aggregate_manifests(root, real, PROTO_SHA, RUNNER_COMMIT, "full")
        assert agg["count"] == 52               # exactly 2 arms x 26 seeds


def test_ce1_missing_output_bytes_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43])
        # delete actual prediction bytes from one valid run
        victim = os.path.join(root, "U_PUBLISHED", "42")
        os.remove(os.path.join(victim, "predictions.parquet"))
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "required output files missing")


def test_ce2_empty_order_hashes_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43], epoch_order_hashes={})
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "epoch_order_hashes empty")


def test_ce3_extra_derived_seeds_key_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        bad = {d: derive_seed(42, d)
               for d in PROTOCOL["attacker_protocol"]["domains"]}
        bad["sneaky"] = 1
        _grid(root, [42, 43], derived_seeds=bad)
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "derived_seeds key set mismatch")


def test_ce4_undeclared_arm_dir_rejected():
    real = dict(PROTOCOL)
    real["seeds"] = {"screen": [42, 43, 44, 45, 46], "full": list(range(42, 68))}
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43, 44, 45, 46])
        rogue = os.path.join(root, "ROGUE_ARM")
        os.makedirs(rogue)
        open(os.path.join(rogue, "note.txt"), "w").close()
        msg = _expect_merr(lambda: aggregate_manifests(
            root, real, PROTO_SHA, RUNNER_COMMIT, "screen"))
        assert "ROGUE_ARM" in msg


def test_undeclared_seed_dir_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43])
        os.makedirs(os.path.join(root, "U_PUBLISHED", "99"))
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "undeclared seed directory")


def test_symlink_run_entry_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43])
        target = os.path.join(root, "U_PUBLISHED", "42")
        link = os.path.join(root, "U_PUBLISHED", "43")
        os.rename(os.path.join(root, "U_PUBLISHED", "43"),
                  os.path.join(tmp, "moved43"))
        os.symlink(target, link)
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "symlink")


def test_byte_hash_binding_rejects_wrong_bytes():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43])
        victim_dir = os.path.join(root, "D_BDEV", "42")
        man_mtime = os.stat(os.path.join(victim_dir,
                                         "run_manifest.json")).st_mtime
        victim = os.path.join(victim_dir, "predictions.parquet")
        open(victim, "wb").write(b"TAMPERED-BYTES")   # bytes no longer match
        os.utime(victim, (man_mtime - 1, man_mtime - 1))  # keep mtime ordering
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "predictions bytes do not match")


def test_post_manifest_modification_rejected():
    import time
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43])
        victim = os.path.join(root, "D_BDEV", "42")
        man_mtime = os.stat(os.path.join(victim,
                                         "run_manifest.json")).st_mtime
        future = man_mtime + 100
        p = os.path.join(victim, "predictions.parquet")
        os.utime(p, (future, future))     # newer than manifest => tampered
        time.sleep(0.01)
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "modified after run-manifest creation")


def test_order_hash_contract_regressions():
    # malformed (non-hex) order value
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "r1")
        oh = _epoch_hashes(42, 8); oh["3"] = "Z" * 64
        _grid(root, [42, 43], epoch_order_hashes=oh)
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "lowercase 64-char hex")
    # incorrect (recomputed-mismatch) order value
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "r2")
        oh = _epoch_hashes(42, 8); oh["5"] = "a" * 64   # plausible but wrong
        _grid(root, [42, 43], epoch_order_hashes=oh)
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "recomputed order hash mismatch")
    # extra epoch key beyond stop_epoch
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "r3")
        oh = _epoch_hashes(42, 8); oh["9"] = "a" * 64
        _grid(root, [42, 43], epoch_order_hashes=oh)
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "epoch keys mismatch")
    # missing epoch key
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "r4")
        oh = _epoch_hashes(42, 8); del oh["0"]
        _grid(root, [42, 43], epoch_order_hashes=oh)
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "epoch keys mismatch")


def test_non_hex_hash_fields_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        grid_ok = []
        for s in [42, 43]:
            for a in ARMS:
                rd = claim_run_directory(root, a, s)
                pred = b"x" * s
                open(os.path.join(rd, "predictions.parquet"), "wb").write(pred)
                open(os.path.join(rd, "attacker_best.pth"), "wb").write(b"y")
                m = {
                    "schema_version": "P0_RUN_MANIFEST_V1_2",
                    "protocol_schema": PROTOCOL["schema_version"],
                    "protocol_sha256": PROTO_SHA,
                    "runner_commit": RUNNER_COMMIT,
                    "arm": a, "master_seed": s,
                    "derived_seeds": {d: derive_seed(s, d)
                                      for d in PROTOCOL["attacker_protocol"]["domains"]},
                    "generator_role": a, "generator_sha256": GEN_SHA[a],
                    "train_pair_sha256": TRAIN_SHA, "val_pair_sha256": VAL_SHA,
                    "initial_attacker_state_hash": "NOT-HEX",
                    "epoch_order_hashes": _epoch_hashes(s, 8),
                    "attacker_best_sha256":
                        hashlib.sha256(b"y").hexdigest(),
                    "best_epoch": 3, "stop_epoch": 8,
                    "score_direction":
                        PROTOCOL["attacker_protocol"]["score_direction"],
                    "predictions_sha256": hashlib.sha256(pred).hexdigest(),
                    "environment_provenance": {"t": 1},
                    "status": "COMPLETE",
                    "started_utc": "2026-08-23T00:00:00+00:00",
                    "finished_utc": "2026-08-23T01:00:00+00:00",
                }
                write_run_manifest_atomic(rd, m)
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "initial_attacker_state_hash must be lowercase")


def test_identity_binding_one_arm_stale_status_still_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43])
        # identity mismatch: swap internal arm label inside U_PUBLISHED/42
        rd = os.path.join(root, "U_PUBLISHED", "42")
        m = load_run_manifest(rd)
        m["arm"] = "D_BDEV"          # actual internal-label swap
        m.pop("manifest_payload_sha256", None)
        payload = json.dumps(m, sort_keys=True, separators=(",", ":")) + "\n"
        m["manifest_payload_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        final = os.path.join(rd, "run_manifest.json")
        os.remove(final)
        with open(final, "wb") as f:
            f.write(json.dumps(m, sort_keys=True,
                               separators=(",", ":")).encode() + b"\n")
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "identity mismatch")
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43])
        # only one arm present -> missing/symlink arm dir rejection
        import shutil as _sh
        _sh.rmtree(os.path.join(root, "D_BDEV"))
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "arm directory")
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _grid(root, [42, 43], status="PARTIAL")
        _expect_merr(lambda: aggregate_manifests(
            root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT, "screen"),
            "incomplete status")


if __name__ == "__main__":
    names = sorted(k for k in globals() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("PASS", name)
    print("ALL PASS (%d)" % len(names))
