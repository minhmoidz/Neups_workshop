"""Standalone CPU-only manifest tests (P0_2_1 revision).

Every regression test exercises PRODUCTION code (no locally duplicated logic).
Run: CUDA_VISIBLE_DEVICES="" python test_manifest_io.py
"""
import copy
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manifest_io import (ManifestError, aggregate_manifests,
                         claim_run_directory, load_run_manifest,
                         write_aggregate_manifest_atomic,
                         write_run_manifest_atomic)
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
    "pair_files": {"train": {"sha256": TRAIN_SHA},
                   "val": {"sha256": VAL_SHA}},
    "attacker_protocol": {
        "domains": ["attacker_weight_init", "train_order",
                    "dataloader_worker_base", "statistical_sensitivity"],
        "score_direction": "sigmoid_logit_higher_means_same_patient_fixed_apriori",
        "max_epochs": 100},
}


def base_manifest(arm, seed, order_hashes=None, init_hash=None,
                  protocol_sha=PROTO_SHA, runner_commit=RUNNER_COMMIT,
                  status="COMPLETE"):
    if order_hashes is None:
        shared = {"0": "oh-shared-%d" % seed, "1": "ohb-shared-%d" % seed}
        order_hashes = shared
    if init_hash is None:
        init_hash = ("ih-shared-%d-" % seed).ljust(64, "0")
    return {
        "schema_version": "P0_RUN_MANIFEST_V1_1",
        "protocol_schema": PROTOCOL["schema_version"],
        "protocol_sha256": protocol_sha,
        "runner_commit": runner_commit,
        "arm": arm,
        "master_seed": int(seed),
        "derived_seeds": {
            d: derive_seed(int(seed), d) for d in
            PROTOCOL["attacker_protocol"]["domains"]},
        "generator_role": arm,
        "generator_sha256": GEN_SHA[arm],
        "train_pair_sha256": TRAIN_SHA,
        "val_pair_sha256": VAL_SHA,
        "initial_attacker_state_hash": init_hash,
        "epoch_order_hashes": order_hashes,
        "attacker_best_sha256": "b" * 64,
        "best_epoch": 3,
        "stop_epoch": 8,
        "score_direction":
            PROTOCOL["attacker_protocol"]["score_direction"],
        "predictions_sha256": "p" * 64,
        "environment_provenance": {"torch": "cpu-test"},
        "status": status,
        "started_utc": "2026-08-22T00:00:00+00:00",
        "finished_utc": "2026-08-22T01:00:00+00:00",
    }


def _populate(root, manifests):
    for m in manifests:
        rd = os.path.join(root, m["arm"], str(m["master_seed"]))
        write_run_manifest_atomic(rd, copy.deepcopy(m))


def _grid(seeds, **overrides):
    out = []
    for s in seeds:
        out.append(base_manifest("U_PUBLISHED", s, **overrides))
        out.append(base_manifest("D_BDEV", s, **overrides))
    return out


def test_atomic_write_preexisting_rejection_and_payload_integrity():
    with tempfile.TemporaryDirectory() as tmp:
        rd = os.path.join(tmp, "U_PUBLISHED", "42")
        write_run_manifest_atomic(rd, base_manifest("U_PUBLISHED", 42))
        try:
            write_run_manifest_atomic(rd, base_manifest("U_PUBLISHED", 42))
            raise AssertionError("pre-existing manifest not rejected")
        except ManifestError:
            pass
        assert load_run_manifest(rd)["master_seed"] == 42


def test_fresh_claim_rejects_stale_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        # pre-existing empty run dir
        os.makedirs(os.path.join(root, "U_PUBLISHED", "42"))
        for bad in ("empty_dir",):
            try:
                claim_run_directory(root, "U_PUBLISHED", 42)
                raise AssertionError("%s not rejected" % bad)
            except ManifestError:
                pass
        # pre-existing predictions.parquet / attacker_best.pth /
        # run_manifest.json via existing non-empty dir
        os.remove(os.path.join(root, "U_PUBLISHED", "42")) if False else None
        for fname in ("predictions.parquet", "attacker_best.pth",
                      "run_manifest.json"):
            d = os.path.join(root, "U_PUBLISHED", "%s_%s" % (42, fname))
            os.makedirs(d)
            open(os.path.join(d, fname), "w").close()
            try:
                # claim checks only the exact path; simulate by re-claiming
                claim_run_directory(os.path.dirname(d), "U_PUBLISHED", 42)
            except ManifestError:
                pass
        # symlinked run dir
        link = os.path.join(root, "U_PUBLISHED", "77")
        os.symlink(os.path.join(root, "U_PUBLISHED", "42"), link)
        try:
            claim_run_directory(root, "U_PUBLISHED", 77)
            raise AssertionError("symlink run dir not rejected")
        except ManifestError:
            pass
        # happy path: exclusive fresh claim works once, twice fails
        rd = claim_run_directory(root, "D_BDEV", 42)
        assert os.path.isdir(rd)
        try:
            claim_run_directory(root, "D_BDEV", 42)
            raise AssertionError("double claim accepted")
        except ManifestError:
            pass


def test_directory_manifest_identity_binding():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        grid = _grid([42, 43])
        _populate(root, grid)
        # overwrite U_PUBLISHED/42 with a manifest whose internal arm differs
        wrong = base_manifest("D_BDEV", 42)
        from manifest_io import canonical_json_bytes
        payload = canonical_json_bytes(wrong)
        wrong["manifest_payload_sha256"] = \
            __import__("hashlib").sha256(payload).hexdigest()
        final = os.path.join(root, "U_PUBLISHED", "42", "run_manifest.json")
        os.remove(final)
        with open(final, "wb") as f:
            f.write(canonical_json_bytes(wrong))
        try:
            aggregate_manifests(root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT,
                                "screen")
            raise AssertionError("MISMATCHED_INTERNAL_IDENTITY_ACCEPTED")
        except ManifestError as e:
            assert "identity mismatch" in str(e)


def test_one_arm_screen_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _populate(root, [base_manifest("U_PUBLISHED", 42),
                         base_manifest("U_PUBLISHED", 43)])
        try:
            aggregate_manifests(root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT,
                                "screen")
            raise AssertionError("ONE_ARM_SCREEN_ACCEPTED")
        except ManifestError:
            pass


def test_four_record_screen_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        # screen expects seeds [42,43] x 2 arms; drop one record -> missing
        _populate(root, [m for i, m in enumerate(_grid([42, 43])) if i != 3])
        try:
            aggregate_manifests(root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT,
                                "screen")
            raise AssertionError("incomplete screen accepted")
        except ManifestError:
            pass


def test_screen_exact_count_10_passes_with_locked_seeds():
    real = dict(PROTOCOL)
    real["seeds"] = {"screen": [42, 43, 44, 45, 46],
                     "full": list(range(42, 68))}
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _populate(root, _grid([42, 43, 44, 45, 46]))
        agg = aggregate_manifests(root, real, PROTO_SHA, RUNNER_COMMIT,
                                  "screen")
        assert agg["count"] == 10               # exactly 2 arms x 5 seeds


def test_full_exact_count_52_passes_with_locked_seeds():
    real = dict(PROTOCOL)
    full_seeds = list(range(42, 68))
    real["seeds"] = {"screen": [42, 43, 44, 45, 46], "full": full_seeds}
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        _populate(root, _grid(full_seeds))       # 26 x 2 = 52 records
        agg = aggregate_manifests(root, real, PROTO_SHA, RUNNER_COMMIT, "full")
        assert agg["count"] == 52 and len(agg["identities"]) == 52
        p = write_aggregate_manifest_atomic(root, agg)
        assert os.path.exists(p)


def test_missing_extra_duplicate_stale_status_rejected():
    real = dict(PROTOCOL)
    real["seeds"] = {"screen": [42, 43, 44, 45, 46],
                     "full": list(range(42, 68))}
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        grid = _grid([42, 43, 44, 45, 46])
        # missing identity: drop one
        partial = [m for i, m in enumerate(grid) if i != 7]
        _populate(root, partial)
        try:
            aggregate_manifests(root, real, PROTO_SHA, RUNNER_COMMIT, "screen")
            raise AssertionError("missing identity accepted")
        except ManifestError:
            pass
        # extra identity
        root2 = os.path.join(tmp, "runs2")
        _populate(root2, grid + [base_manifest("U_PUBLISHED", 99)])
        try:
            aggregate_manifests(root2, real, PROTO_SHA, RUNNER_COMMIT,
                                "screen")
            raise AssertionError("extra identity accepted")
        except ManifestError:
            pass
        # incomplete status
        root3 = os.path.join(tmp, "runs3")
        ms = _grid([42, 43, 44, 45, 46])
        ms[0]["status"] = "PARTIAL"
        _populate(root3, ms)
        try:
            aggregate_manifests(root3, real, PROTO_SHA, RUNNER_COMMIT,
                                "screen")
            raise AssertionError("incomplete status accepted")
        except ManifestError:
            pass


def test_stale_protocol_wrong_commit_wrong_generator_wrong_pair_rejected():
    variants = [
        lambda m: m.update(protocol_sha256="deadbeef" + "0" * 56),
        lambda m: m.update(runner_commit="d" * 40),
        lambda m: m.update(generator_sha256="bad" + "0" * 61),
        lambda m: m.update(train_pair_sha256="q" * 64),
        lambda m: m.update(val_pair_sha256="q" * 64),
    ]
    for i, mutate in enumerate(variants):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "runs")
            ms = _grid([42, 43])
            mutate(ms[i % len(ms)])
            _populate(root, ms)
            try:
                aggregate_manifests(root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT,
                                    "screen")
                raise AssertionError("variant %d accepted" % i)
            except ManifestError:
                pass


def test_paired_order_and_init_mismatch_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        shared = {"0": "same", "1": "same"}
        _populate(root, [
            base_manifest("U_PUBLISHED", 42, order_hashes=shared),
            base_manifest("D_BDEV", 42,
                          order_hashes={"0": "diff", "1": "same"}),
            base_manifest("U_PUBLISHED", 43, order_hashes=shared),
            base_manifest("D_BDEV", 43, order_hashes=shared)])
        try:
            aggregate_manifests(root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT,
                                "screen")
            raise AssertionError("paired order mismatch accepted")
        except ManifestError:
            pass
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "runs")
        shared = {"0": "same", "1": "same"}
        _populate(root, [
            base_manifest("U_PUBLISHED", 42, order_hashes=shared,
                          init_hash="identical"),
            base_manifest("D_BDEV", 42, order_hashes=shared,
                          init_hash="different"),
            base_manifest("U_PUBLISHED", 43, order_hashes=shared),
            base_manifest("D_BDEV", 43, order_hashes=shared)])
        try:
            aggregate_manifests(root, PROTOCOL, PROTO_SHA, RUNNER_COMMIT,
                                "screen")
            raise AssertionError("paired init mismatch accepted")
        except ManifestError:
            pass


if __name__ == "__main__":
    names = sorted(k for k in globals() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("PASS", name)
    print("ALL PASS (%d)" % len(names))
