"""P0 mu-curve driver — REAL GPU runs, sequential, arm list parametrized.

WHY THIS EXISTS
---------------
`screen_real_run.py` hard-codes `arms = ["U_PUBLISHED", "D_BDEV"]`. Evaluating
the other released endpoints needs an arm list supplied at call time, and — the
part that actually matters — each arm must be anonymized at ITS OWN mu.
Protocol V1_2 moves mu onto the arm; `attacker_loop.build_anonymize_from_
generator` now honours it. Scoring a mu=0.001 checkpoint with a mu=0.01
deformation would silently measure the wrong object.

WHAT IT IS FOR
--------------
The published paper reports re-identification AUC of 74.7 / 64.3 / 57.7 at
mu = 0.001 / 0.005 / 0.01 (Table 1), measured with an SNN retrained on deformed
images and evaluated on the TEST fold. Under the governed adaptive-attacker
harness on VAL, this project measures the mu=0.01 released endpoint at 0.6985
(n=26). This driver measures the OTHER TWO released endpoints under the same
harness so the published curve can be compared point-by-point against a curve
produced by one protocol, one fold and one environment.

No generator is trained here. Every arm is a released upstream artifact whose
SHA-256 is verified against the protocol before execution.

DIAGNOSTIC ONLY: this script issues no scientific verdict. Classification is
made once, elsewhere, from the sealed manifests.

Usage (repository root):
  setsid nohup python reproduction/p0_bridge/mu_curve_driver.py \
      --arms U_MU0001,U_MU0005 \
      --seeds 42,43,44,45,46 \
      --runs-root reproduction/p0_bridge/runs_mucurve \
      --image-root /home/minhtt/datasets/nih/images/ \
      > /tmp/p0_mucurve.log 2>&1 < /dev/null &
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
    ap.add_argument("--arms", required=True,
                    help="comma-separated arm names, each must exist in the protocol")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--image-root", default="/home/minhtt/datasets/nih/images/")
    ap.add_argument("--repo-root", default=DEFAULT_REPO)
    ap.add_argument("--runs-root", default="reproduction/p0_bridge/runs_mucurve")
    ap.add_argument("--approval", required=True,
                    help="path to the human-signed execution approval manifest")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",")]

    repo_root = os.path.abspath(args.repo_root)
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, "research_agent"))

    # Every gate BEFORE any scientific import or GPU touch: signature,
    # protocol hash, runner commit, complete arm role/hash set, pair hashes,
    # SEOI, output root, competing-trainer check, clean tracked worktree, and
    # byte-verification of every governed artifact. An unsigned or stale
    # manifest stops the run here.
    from run_p0_bridge import load_and_check_approval, load_protocol
    protocol, proto_sha, stage, approval = load_and_check_approval(args.approval)
    print("[mucurve] approval OK: stage=%s approved_by=%r at %s"
          % (stage, approval["approved_by"], approval["approval_timestamp"]))

    unknown = [a for a in arms if a not in protocol["arms"]]
    if unknown:
        raise SystemExit("arms not present in the locked protocol: %s" % unknown)

    rc = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                 cwd=repo_root).decode().strip()
    default_mu = protocol["attacker_protocol"]["flow_operator"]["mu"]
    print("[mucurve] protocol_sha :", proto_sha[:16], "...")
    print("[mucurve] schema       :", protocol.get("schema_version"))
    print("[mucurve] runner_commit:", rc[:12])
    for a in arms:
        print("[mucurve] arm %-12s mu=%s  %s"
              % (a, protocol["arms"][a].get("mu", default_mu),
                 protocol["arms"][a]["execution_path"]))

    print("[mucurve] governed artifacts verified by the approval gate; starting")

    from attacker_loop import P0AttackerRunner

    results = []
    t_all = time.time()
    for arm in arms:
        for seed in seeds:
            t0 = time.time()
            r = P0AttackerRunner(
                arm=arm, master_seed=seed, protocol=protocol,
                repo_root=repo_root, image_root=args.image_root,
                protocol_sha256=proto_sha, runner_commit=rc)
            r.run_training()          # locked budget: <=100 epochs, patience 5
            rd, man = r.finalize(runs_root=os.path.join(repo_root,
                                                        args.runs_root))
            rec = {"arm": arm, "seed": seed,
                   "resolved_mu": man.get("resolved_mu"),
                   "raw_roc_auc": man["raw_roc_auc"],
                   "best_epoch": man["best_epoch"],
                   "stop_epoch": man["stop_epoch"],
                   "elapsed_sec": round(time.time() - t0, 1),
                   "run_dir": rd}
            results.append(rec)
            print("[MUCURVE DONE]", json.dumps(rec), flush=True)

    summary = {
        "stage": "mu_curve",
        "protocol_sha256": proto_sha,
        "protocol_schema": protocol.get("schema_version"),
        "runner_commit": rc,
        "arms": arms,
        "seeds": seeds,
        "total_elapsed_sec": round(time.time() - t_all, 1),
        "results": sorted(results, key=lambda r: (r["arm"], r["seed"])),
    }
    out = os.path.join(repo_root, args.runs_root, "mu_curve_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    print("[MUCURVE COMPLETE]", json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
