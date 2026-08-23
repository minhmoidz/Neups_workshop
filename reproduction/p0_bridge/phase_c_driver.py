"""P0 Phase C driver: (A) bridge extension seeds 47..67 + (B) I_M2 diagnostic.

Phase B runs FIRST (10 runs, ~7h) because it answers the mechanism question
(fine-tuning regression vs weak initialization) fastest.

Resume-safe: a run whose sealed COMPLETE manifest already exists is skipped.
"""
import json
import os
import shutil
import subprocess
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else \
    "/home/minhtt/Neups_workshop"
os.chdir(REPO)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "research_agent"))

from run_p0_bridge import load_protocol, verify_all_artifacts  # noqa: E402

IMAGE_ROOT = "/home/minhtt/datasets/nih/images/"
IM2_SEEDS = list(range(42, 52))          # 10 diagnostic runs
BRIDGE_SEEDS = list(range(47, 68))       # 21 x 2 arms = 42 extension runs


def run_one(runner_cls, protocol, proto_sha, rc, arm, seed, runs_root,
            image_root):
    from manifest_io import load_run_manifest, ManifestError
    rd = os.path.join(runs_root, arm, str(seed))
    try:
        m = load_run_manifest(rd)
        if m.get("status") == "COMPLETE":
            print("[SKIP complete]", arm, seed, flush=True)
            return {"arm": arm, "seed": seed, "raw_roc_auc": m["raw_roc_auc"],
                    "best_epoch": m["best_epoch"], "stop_epoch": m["stop_epoch"],
                    "skipped": True}
    except Exception:
        # incomplete/corrupt attempt: purge and redo cleanly
        if os.path.isdir(rd):
            shutil.rmtree(rd, ignore_errors=True)
    t0 = time.time()
    r = runner_cls(arm=arm, master_seed=seed, protocol=protocol,
                   repo_root=REPO, image_root=image_root,
                   protocol_sha256=proto_sha, runner_commit=rc)
    r.run_training()
    _, man = r.finalize(runs_root=runs_root)
    rec = {"arm": arm, "seed": seed, "raw_roc_auc": man["raw_roc_auc"],
           "best_epoch": man["best_epoch"], "stop_epoch": man["stop_epoch"],
           "elapsed_sec": round(time.time() - t0, 1), "run_dir": rd}
    print("[RUN DONE]", json.dumps(rec), flush=True)
    return rec


def main():
    protocol, proto_sha = load_protocol()
    rc = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                 cwd=REPO).decode().strip()
    verify_all_artifacts(protocol, REPO)
    print("[phaseC] artifacts verified; commit", rc[:12], flush=True)

    from attacker_loop import P0AttackerRunner
    from manifest_io import aggregate_manifests

    # ---------------- Phase B: I_M2 diagnostic (first) ----------------
    im2_root = os.path.join(REPO, "reproduction/p0_bridge/runs_im2")
    diag = json.loads(json.dumps(protocol))          # deep copy
    diag["arms"] = {"I_M2": {
        "role": "initial M2 generator (diagnostic anchor)",
        "generator_path": "networks/pretrained_generator_prichexy_net.pth",
        "generator_sha256":
            "101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb",
        "execution_path": "networks/pretrained_generator_prichexy_net.pth"}}
    diag["seeds"] = {"screen": IM2_SEEDS, "full": IM2_SEEDS}
    results_im2 = []
    for seed in IM2_SEEDS:
        results_im2.append(run_one(P0AttackerRunner, diag, proto_sha, rc,
                                   "I_M2", seed, im2_root, IMAGE_ROOT))
    agg = aggregate_manifests(im2_root, diag, proto_sha, rc, "screen")
    with open(os.path.join(im2_root, "im2_summary.json"), "w") as f:
        json.dump({"count": agg["count"], "results": results_im2}, f, indent=1)
    print("[IM2 COMPLETE]", json.dumps(results_im2, indent=1), flush=True)

    # ------------- Phase A: bridge extension seeds 47..67 -------------
    runs_screen = os.path.join(REPO, "reproduction/p0_bridge/runs_screen")
    results_ext = []
    for seed in BRIDGE_SEEDS:
        for arm in ("U_PUBLISHED", "D_BDEV"):
            results_ext.append(run_one(P0AttackerRunner, protocol, proto_sha,
                                       rc, arm, seed, runs_screen, IMAGE_ROOT))
    agg_full = aggregate_manifests(runs_screen, protocol, proto_sha, rc,
                                   "full")
    with open(os.path.join(runs_screen, "bridge_summary.json"), "w") as f:
        json.dump({"count": agg_full["count"], "results": results_ext},
                  f, indent=1)
    print("[BRIDGE COMPLETE] count=", agg_full["count"], flush=True)


if __name__ == "__main__":
    main()
