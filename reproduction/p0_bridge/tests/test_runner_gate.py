"""Standalone CPU-only runner-gate tests (P0_2_1 revision).

Run: CUDA_VISIBLE_DEVICES="" python test_runner_gate.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(HERE, "run_p0_bridge.py")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")

PROTO_SHA_FALLBACK = "b63f98af8e37a294b45ea6686282e5f392b4a26ff68179cf9ff86ea4a732e731"


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
    assert out["protocol_sha256"] != PROTO_SHA_FALLBACK  # revision bumped


def test_no_bypass_flags_accepted():
    for flag in ("--force", "--skip-hash", "--ignore-mismatch",
                 "--replace-seed", "--flip-score"):
        r = _run([flag])
        assert r.returncode != 0, flag


def _load_protocol():
    sys.path.insert(0, HERE)
    from run_p0_bridge import load_protocol, runner_commit
    return load_protocol(), runner_commit()


def _full_approval(tmp, **overrides):
    (_, sha), rc = _load_protocol()
    approval = {
        "authorization_status": "APPROVED",
        "approved_protocol_sha256": sha,
        "approved_runner_commit": rc or ("c" * 40),
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
        "approved_output_root": "reproduction/p0_bridge/runs",
        "approved_by": "human-reviewer-1",
        "approval_timestamp": "2026-08-22T00:00:00+00:00",
        "active_process_clearance": True,
    }
    approval.update(overrides)
    p = os.path.join(tmp, "approval.json")
    with open(p, "w") as f:
        json.dump(approval, f)
    return p


def test_execute_without_manifest_fails_closed():
    r = _run(["--execute"])
    assert r.returncode != 0
    assert "missing execution manifest" in (r.stderr + r.stdout)


def test_execute_not_authorized_fails():
    with tempfile.TemporaryDirectory() as tmp:
        fields = ["authorization_status", "approved_protocol_sha256",
                  "approved_runner_commit", "approved_screen_or_full_stage",
                  "approved_seed_list", "approved_generator_roles_and_hashes",
                  "approved_pair_hashes", "approved_SEOI",
                  "approved_output_root", "approved_by", "approval_timestamp",
                  "active_process_clearance"]
        p = os.path.join(tmp, "approval.json")
        json.dump({f: None for f in fields} |
                  {"authorization_status": "NOT_AUTHORIZED"}, open(p, "w"))
        r = _run(["--execute", "--execution-manifest", p])
        blob = r.stderr + r.stdout
        assert r.returncode != 0 and "NOT_AUTHORIZED" in blob


def test_blank_approver_rejected():
    sys.path.insert(0, HERE)
    import run_p0_bridge as runner
    with tempfile.TemporaryDirectory() as tmp:
        ap = _full_approval(tmp, approved_by="   ")
        try:
            runner.load_and_check_approval(ap)
            raise AssertionError("EMPTY_OUTPUT_AND_BLANK_APPROVER_ACCEPTED")
        except PermissionError as e:
            assert "non-empty" in str(e)


def test_naive_timestamp_rejected():
    sys.path.insert(0, HERE)
    import run_p0_bridge as runner
    with tempfile.TemporaryDirectory() as tmp:
        ap = _full_approval(tmp, approval_timestamp="2026-08-22T00:00:00")
        try:
            runner.load_and_check_approval(ap)
            raise AssertionError("naive timestamp accepted")
        except PermissionError as e:
            assert "timezone-aware" in str(e)


def test_output_root_mismatch_and_traversal_rejected():
    sys.path.insert(0, HERE)
    import run_p0_bridge as runner
    with tempfile.TemporaryDirectory() as tmp:
        ap = _full_approval(tmp, approved_output_root="/tmp/elsewhere")
        try:
            runner.load_and_check_approval(ap)
            raise AssertionError("arbitrary output root accepted")
        except PermissionError:
            pass
    with tempfile.TemporaryDirectory() as tmp:
        ap = _full_approval(
            tmp, approved_output_root="reproduction/../elsewhere/runs")
        try:
            runner.load_and_check_approval(ap)
            raise AssertionError("traversal output root accepted")
        except PermissionError as e:
            assert ("does not equal" in str(e)) or ("traversal" in str(e))


def test_unknown_field_rejected():
    sys.path.insert(0, HERE)
    import run_p0_bridge as runner
    with tempfile.TemporaryDirectory() as tmp:
        ap_path = _full_approval(tmp)
        approval = json.load(open(ap_path))
        approval["sneaky_extra"] = True
        json.dump(approval, open(ap_path, "w"))
        try:
            runner.load_and_check_approval(ap_path)
            raise AssertionError("unknown approval field accepted")
        except PermissionError as e:
            assert "unknown approval fields" in str(e)


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


def test_dirty_worktree_rejected():
    """Adversarial regression: dirty tracked worktree must fail the gate.

    Uses a REAL synthetic git repository so production code is exercised.
    """
    sys.path.insert(0, HERE)
    import run_p0_bridge as runner
    orig_clean = runner.worktree_is_clean_tracked
    orig_rc = runner.runner_commit
    # build a real tiny repo to obtain a genuine clean/dirty signal
    with tempfile.TemporaryDirectory() as tmp:
        repo = os.path.join(tmp, "repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        open(os.path.join(repo, "f.txt"), "w").write("v1")
        subprocess.run(["git", "-C", repo, "add", "f.txt"], check=True)
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
                   GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        subprocess.run(["git", "-C", repo, "commit", "-qm", "init"],
                       check=True, env=env)
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
        dirty_call = {"n": 0}

        def fake_git(args, cwd):
            dirty_call["n"] += 1
            if args[1:2] == ["status"]:
                return subprocess.CompletedProcess(
                    args, 0, " M f.txt\n", "")      # DIRTY tracked file
            if args[1:2] == ["rev-parse"]:
                return subprocess.CompletedProcess(
                    args, 0, (head + "\n").encode(), b"")
            raise AssertionError("unexpected git call %r" % (args,))
        runner_commit_orig = runner.runner_commit
        runner.runner_commit = lambda: head

        orig_git = runner.subprocess.run
        def patched_run(args, **kw):
            if args[0] == "git":
                return fake_git(args, kw.get("cwd"))
            return orig_git(args, **kw)
        runner.subprocess.run = patched_run
        try:
            with tempfile.TemporaryDirectory() as tmp2:
                ap = _full_approval(tmp2)     # approved commit == real HEAD here;
                # force commit match via our fake rev-parse output:
                approval = json.load(open(ap))
                approval["approved_runner_commit"] = head
                json.dump(approval, open(ap, "w"))
                try:
                    runner.load_and_check_approval(ap)
                    raise AssertionError("dirty worktree accepted")
                except PermissionError as e:
                    assert "dirty" in str(e).lower()
                assert dirty_call["n"] > 0    # production git path exercised
        finally:
            runner.subprocess.run = orig_git
            runner.worktree_is_clean_tracked = orig_clean
            runner.runner_commit = orig_rc


def test_actual_artifact_hash_mismatch_rejected():
    """Production verify_artifact_sha against a synthetic injected root."""
    sys.path.insert(0, HERE)
    import run_p0_bridge as runner
    with tempfile.TemporaryDirectory() as tmp:
        content = b"synthetic-checkpoint-bytes"
        good_sha = hashlib.sha256(content).hexdigest()
        open(os.path.join(tmp, "gen.pth"), "wb").write(content)
        assert runner.verify_artifact_sha(tmp, "gen.pth", good_sha) is True
        try:
            runner.verify_artifact_sha(tmp, "gen.pth", "e" * 64)
            raise AssertionError("byte-hash mismatch accepted")
        except PermissionError as e:
            assert "hash mismatch" in str(e)
        try:
            runner.verify_artifact_sha(tmp, "../outside.pth", good_sha)
            raise AssertionError("escaping path accepted")
        except PermissionError:
            pass


def test_imagenet_unresolved_blocks_execution_gate():
    sys.path.insert(0, HERE)
    import run_p0_bridge as runner
    protocol = {
        "schema_version": "P0_PROTOCOL_V1_1",
        "arms": {},
        "pair_files": {},
        "imagenet_weight_artifact": {"status": "UNRESOLVED_BLOCKER"},
    }
    try:
        runner.verify_all_artifacts(protocol, "/tmp/nonexistent-root")
        raise AssertionError("UNRESOLVED imagenet artifact not refused")
    except PermissionError as e:
        assert "ImageNet weight artifact" in str(e)


if __name__ == "__main__":
    names = sorted(k for k in globals() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("PASS", name)
    print("ALL PASS (%d)" % len(names))
