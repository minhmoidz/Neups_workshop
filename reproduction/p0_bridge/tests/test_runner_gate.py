"""Standalone CPU-only runner-gate tests. Run: python test_runner_gate.py"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(HERE, "run_p0_bridge.py")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")


def _run(argv):
    return subprocess.run([sys.executable, RUNNER] + argv,
                          capture_output=True, text=True, env=ENV)


def test_import_has_no_side_effects():
    code = (
        "import sys; before=set(sys.modules);"
        "sys.path.insert(0,%r); import run_p0_bridge;"
        "new=set(sys.modules)-before;"
        "bad=[m for m in new if any(k in m for k in"
        "('torch','torchvision','cv2','PIL','sklearn'))];"
        "print('SCIENTIFIC_IMPORTS:', bad)" % HERE)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=ENV)
    assert r.returncode == 0, r.stderr
    assert "SCIENTIFIC_IMPORTS: []" in r.stdout


def test_validate_protocol_only_clean():
    r = _run(["--validate-protocol-only"])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["cuda_touched"] is False and out["real_data_opened"] is False
    assert len(out["protocol_sha256"]) == 64


def test_no_bypass_flags_accepted():
    # Behavioral: every prohibited flag must be rejected by argparse.
    for flag in ("--force", "--skip-hash", "--ignore-mismatch",
                 "--replace-seed", "--flip-score"):
        r = _run([flag])
        assert r.returncode != 0, flag
        blob = r.stderr + r.stdout
        assert "unrecognized" in blob.lower(), (flag, blob[-200:])


def test_execute_without_manifest_fails_closed():
    r = _run(["--execute"])
    assert r.returncode != 0
    assert "missing execution manifest" in (r.stderr + r.stdout)


def _load_protocol():
    sys.path.insert(0, HERE)
    from run_p0_bridge import load_protocol, runner_commit
    return load_protocol(), runner_commit()


def _full_approval(tmp, **overrides):
    (protocol_sha, _), rc = _load_protocol()
    protocol, sha = None, protocol_sha
    sys.path.insert(0, HERE)
    from run_p0_bridge import load_protocol
    _, real_sha = load_protocol()
    approval = {
        "authorization_status": "APPROVED",
        "approved_protocol_sha256": real_sha,
        "approved_runner_commit": rc,
        "approved_screen_or_full_stage": "screen",
        "approved_seed_list": [42, 43, 44, 45, 46],
        "approved_generator_roles_and_hashes": {
            "U_PUBLISHED":
                "4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153",
            "D_BDEV":
                "18381d92c64bb3d646b62d5fb9d0ed8c208cf2cb3154f8aa1dac4b1baff610cd"},
        "approved_pair_hashes": {
            "train": "3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268",
            "val": "9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7"},
        "approved_SEOI": 0.03,
        "approved_output_root": os.path.join(tmp, "runs_out"),
        "approved_by": "human-test",
        "approval_timestamp": "2026-08-22T00:00:00Z",
        "active_process_clearance": True,
    }
    approval.update(overrides)
    p = os.path.join(tmp, "approval.json")
    with open(p, "w") as f:
        json.dump(approval, f)
    return p


def test_execute_not_authorized_fails():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "approval.json")
        fields = ["authorization_status", "approved_protocol_sha256",
                  "approved_runner_commit", "approved_screen_or_full_stage",
                  "approved_seed_list", "approved_generator_roles_and_hashes",
                  "approved_pair_hashes", "approved_SEOI",
                  "approved_output_root", "approved_by", "approval_timestamp",
                  "active_process_clearance"]
        approval = {f: None for f in fields}
        approval["authorization_status"] = "NOT_AUTHORIZED"
        with open(p, "w") as f:
            json.dump(approval, f)
        r = _run(["--execute", "--execution-manifest", p])
        assert r.returncode != 0
        blob = r.stderr + r.stdout
        assert ("NOT_AUTHORIZED" in blob) or ("authorization_status" in blob)


def test_execute_mismatched_protocol_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ap = _full_approval(tmp, approved_protocol_sha256="f" * 64)
        r = _run(["--execute", "--execution-manifest", ap])
        blob = r.stderr + r.stdout
        assert r.returncode != 0 and "protocol-hash mismatch" in blob


def test_execute_changed_seed_list_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ap = _full_approval(tmp, approved_seed_list=[42, 43])
        r = _run(["--execute", "--execution-manifest", ap])
        blob = r.stderr + r.stdout
        assert r.returncode != 0 and "seed-list mismatch" in blob


def test_execute_stage_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ap = _full_approval(tmp, approved_screen_or_full_stage="bogus")
        r = _run(["--execute", "--execution-manifest", ap])
        blob = r.stderr + r.stdout
        assert r.returncode != 0 and "stage mismatch" in blob


def test_execute_changed_checkpoint_fails():
    with tempfile.TemporaryDirectory() as tmp:
        ap = _full_approval(
            tmp, approved_generator_roles_and_hashes={
                "U_PUBLISHED": "bad", "D_BDEV": "bad"})
        r = _run(["--execute", "--execution-manifest", ap])
        blob = r.stderr + r.stdout
        assert r.returncode != 0 and (
            "generator role/checkpoint hash changed" in blob)


def test_active_process_gate_fails_closed():
    sys.path.insert(0, HERE)
    import run_p0_bridge as runner
    orig = runner.active_training_process_running
    runner.active_training_process_running = lambda: (True, "fake")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ap = _full_approval(tmp)
            try:
                runner.load_and_check_approval(ap)
                raise AssertionError("active-process gate did not fail closed")
            except PermissionError as e:
                assert "run_hardened_verifier" in str(e)
    finally:
        runner.active_training_process_running = orig


def test_validate_mode_never_initializes_cuda():
    r = _run(["--validate-protocol-only"])
    assert r.returncode == 0
    check = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(torch.cuda.is_initialized())"],
        capture_output=True, text=True, env=ENV)
    assert check.stdout.strip() == "False"


if __name__ == "__main__":
    fns = [globals()[k] for k in sorted(k for k in globals() if k.startswith("test_"))]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL PASS (%d)" % len(fns))
