"""Standalone CPU-only manifest tests. Run: python test_manifest_io.py"""
import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manifest_io import (ManifestError, aggregate_manifests,
                         load_run_manifest, write_aggregate_manifest_atomic,
                         write_run_manifest_atomic)

ARMS = ["U_PUBLISHED", "D_BDEV"]
SEEDS = [42, 43]
PROTO_SHA = "a" * 64


def base_manifest(arm, seed, order_hashes=None, init_hash=None,
                  protocol_sha=PROTO_SHA):
    if order_hashes is None:
        order_hashes = {"0": "oh%d" % seed, "1": "ohb%d" % seed}
    return {
        "protocol_schema": "P0_PROTOCOL_V1",
        "protocol_sha256": protocol_sha,
        "runner_commit": "c" * 40,
        "arm": arm,
        "master_seed": int(seed),
        "derived_seeds": {"weight": seed * 2},
        "generator_role": arm,
        "generator_sha256": {"U_PUBLISHED": "4d", "D_BDEV": "18"}[arm],
        "train_pair_sha256": "t" * 64,
        "val_pair_sha256": "v" * 64,
        "initial_attacker_state_hash": init_hash or ("ih%d" % seed),
        "epoch_order_hashes": order_hashes,
        "best_epoch": 3,
        "stop_epoch": 8,
        "score_direction": "higher_same_patient",
        "predictions_sha256": "p" * 64,
        "environment_provenance": {"torch": "cpu-test"},
        "status": "COMPLETE",
        "started_utc": "2026-08-22T00:00:00Z",
        "finished_utc": "2026-08-22T01:00:00Z",
    }


def _populate(root, manifests):
    for m in manifests:
        rd = os.path.join(root, m["arm"], str(m["master_seed"]))
        write_run_manifest_atomic(rd, copy.deepcopy(m))


def test_atomic_write_and_preexisting_rejection():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        rd = os.path.join(root, "U_PUBLISHED", "42")
        p = write_run_manifest_atomic(rd, base_manifest("U_PUBLISHED", 42))
        assert os.path.exists(p)
        try:
            write_run_manifest_atomic(rd, base_manifest("U_PUBLISHED", 42))
            raise AssertionError("pre-existing output not rejected")
        except ManifestError:
            pass
        assert load_run_manifest(rd)["master_seed"] == 42


def test_missing_identity_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _populate(root, [base_manifest("U_PUBLISHED", 42),
                         base_manifest("D_BDEV", 42)])
        try:
            aggregate_manifests(root, ARMS, SEEDS, PROTO_SHA, "screen")
            raise AssertionError("missing identity not rejected")
        except ManifestError:
            pass


def test_duplicate_identity_rule():
    seen = set()
    for arm in ARMS:
        for s in SEEDS:
            ident = (str(arm), int(s))
            assert ident not in seen
            seen.add(ident)
    try:
        ident = (ARMS[0], int(SEEDS[0]))
        if ident in seen:
            raise ManifestError("duplicate identity: %r" % (ident,))
        raise AssertionError("duplicate not detected")
    except ManifestError:
        pass


def test_unexpected_identity_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _populate(root, [base_manifest(a, s) for s in SEEDS for a in ARMS])
        extra = base_manifest("U_PUBLISHED", 42)
        extra["master_seed"] = 99
        write_run_manifest_atomic(
            os.path.join(root, "U_PUBLISHED", "99"), extra)
        try:
            aggregate_manifests(root, ARMS, SEEDS, PROTO_SHA, "screen")
            raise AssertionError("unexpected identity not rejected")
        except ManifestError:
            pass


def test_stale_protocol_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        ms = [base_manifest(a, s) for s in SEEDS for a in ARMS]
        ms[0]["protocol_sha256"] = "deadbeef" + "0" * 56
        _populate(root, ms)
        try:
            aggregate_manifests(root, ARMS, SEEDS, PROTO_SHA, "screen")
            raise AssertionError("stale protocol hash not rejected")
        except ManifestError:
            pass


def test_paired_order_hash_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        shared = {"0": "same", "1": "same"}
        _populate(root, [
            base_manifest("U_PUBLISHED", 42, order_hashes=shared),
            base_manifest("D_BDEV", 42, order_hashes={"0": "diff", "1": "same"}),
            base_manifest("U_PUBLISHED", 43, order_hashes=shared),
            base_manifest("D_BDEV", 43, order_hashes=shared)])
        try:
            aggregate_manifests(root, ARMS, SEEDS, PROTO_SHA, "screen")
            raise AssertionError("order-hash mismatch not rejected")
        except ManifestError:
            pass


def test_paired_init_hash_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        shared = {"0": "same", "1": "same"}
        _populate(root, [
            base_manifest("U_PUBLISHED", 42, order_hashes=shared,
                          init_hash="identical-init"),
            base_manifest("D_BDEV", 42, order_hashes=shared,
                          init_hash="different-init"),
            base_manifest("U_PUBLISHED", 43, order_hashes=shared),
            base_manifest("D_BDEV", 43, order_hashes=shared)])
        try:
            aggregate_manifests(root, ARMS, SEEDS, PROTO_SHA, "screen")
            raise AssertionError("init-hash mismatch not rejected")
        except ManifestError:
            pass


def test_deterministic_aggregate_and_atomic_write():
    with tempfile.TemporaryDirectory() as tmp:
        r1, r2 = os.path.join(tmp, "r1"), os.path.join(tmp, "r2")
        for r in (r1, r2):
            _populate(r, [base_manifest(a, s) for s in SEEDS for a in ARMS])
        a1 = aggregate_manifests(r1, ARMS, SEEDS, PROTO_SHA, "screen")
        a2 = aggregate_manifests(r2, ARMS, SEEDS, PROTO_SHA, "screen")
        assert a1 == a2 and a1["count"] == 4
        ids = [(i["arm"], i["seed"]) for i in a1["identities"]]
        assert ids == sorted(ids, key=lambda t: (t[1], t[0]))
        p = write_aggregate_manifest_atomic(r1, a1)
        assert os.path.exists(p)
        try:
            write_aggregate_manifest_atomic(r1, a1)
            raise AssertionError("aggregate collision not rejected")
        except ManifestError:
            pass


if __name__ == "__main__":
    fns = [globals()[k] for k in sorted(k for k in globals() if k.startswith("test_"))]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL PASS (%d)" % len(fns))
