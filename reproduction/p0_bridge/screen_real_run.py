"""P0 diagnostic screen driver — REAL GPU runs, sequential.

Runs the locked attacker protocol for screen seeds {42..46} on BOTH arms,
sharing the eagerly-decoded datasets across seeds within each arm.
Diagnostic only: no scientific verdict is issued by this script.

Usage (repository root):
  setsid nohup python reproduction/p0_bridge/screen_real_run.py \
      --image-root /home/minhtt/datasets/nih/images/ \
      > /tmp/p0_screen.log 2>&1 < /dev/null &
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", default="/home/minhtt/datasets/nih/images/")
    ap.add_argument("--repo-root", default=DEFAULT_REPO)
    ap.add_argument("--runs-root", default="reproduction/p0_bridge/runs_screen")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    repo_root = os.path.abspath(args.repo_root)
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, "research_agent"))

    from run_p0_bridge import (load_protocol, verify_all_artifacts)
    protocol, proto_sha = load_protocol()
    rc = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                 cwd=repo_root).decode().strip()
    print("[screen] protocol_sha:", proto_sha[:16], "...")
    print("[screen] runner_commit:", rc[:12])
    verify_all_artifacts(protocol, repo_root)
    print("[screen] governed artifacts verified; starting")

    from attacker_loop import P0AttackerRunner

    arms = ["U_PUBLISHED", "D_BDEV"]
    results = []
    t_all = time.time()
    for arm in arms:
        for seed in seeds:
            t0 = time.time()
            r = P0AttackerRunner(
                arm=arm, master_seed=seed, protocol=protocol,
                repo_root=repo_root, image_root=args.image_root,
                protocol_sha256=proto_sha, runner_commit=rc)
            r.run_training()                      # locked budget: <=100 epochs, patience 5
            rd, man = r.finalize(runs_root=os.path.join(repo_root,
                                                        args.runs_root))
            rec = {"arm": arm, "seed": seed,
                   "raw_roc_auc": man["raw_roc_auc"],
                   "best_epoch": man["best_epoch"],
                   "stop_epoch": man["stop_epoch"],
                   "elapsed_sec": round(time.time() - t0, 1),
                   "run_dir": rd}
            results.append(rec)
            print("[SCREEN DONE]", json.dumps(rec), flush=True)

    # diagnostic aggregate (production code; screen expects exactly 10)
    from manifest_io import aggregate_manifests
    agg = aggregate_manifests(os.path.join(repo_root, args.runs_root),
                              protocol, proto_sha, rc, "screen")
    summary = {"stage": "screen", "count": agg["count"],
               "total_elapsed_sec": round(time.time() - t_all, 1),
               "results": sorted(results, key=lambda r: (r["seed"], r["arm"]))}
    with open(os.path.join(repo_root, args.runs_root,
                           "screen_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("[SCREEN COMPLETE]", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
