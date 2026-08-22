"""Smoke driver: run ONE real attacker trajectory end-to-end.

CPU-free of scientific shortcuts — this is the real GPU path:
real frozen generator checkpoint, real NIH images, real pair files,
full deterministic sampler + manifest byte binding.

Usage (from repository root):
  CUDA_VISIBLE_DEVICES=0 python reproduction/p0_bridge/smoke_real_run.py \
      --arm U_PUBLISHED --seed 42 --epochs 2 \
      --image-root /home/minhtt/datasets/nih/images/ \
      --runs-root reproduction/p0_bridge/runs_smoke
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["U_PUBLISHED", "D_BDEV"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--image-root", default="/home/minhtt/datasets/nih/images/")
    ap.add_argument("--runs-root",
                    default="reproduction/p0_bridge/runs_smoke")
    ap.add_argument("--repo-root", default=REPO_ROOT,
                    help="repository holding REAL governed artifact bytes")
    args = ap.parse_args()

    from run_p0_bridge import load_protocol, runner_commit
    protocol, proto_sha = load_protocol()
    repo_root = os.path.abspath(args.repo_root)
    os.chdir(repo_root)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, "research_agent"))
    rc = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                 cwd=repo_root).decode().strip()
    print("[smoke] repo_root:", repo_root)
    print("[smoke] repo HEAD:", rc)

    # governed artifact gates BEFORE any training (production functions)
    from run_p0_bridge import verify_all_artifacts
    verify_all_artifacts(protocol, repo_root)
    print("[smoke] all governed artifacts verified by SHA")

    from attacker_loop import P0AttackerRunner
    runner = P0AttackerRunner(
        arm=args.arm, master_seed=args.seed, protocol=protocol,
        repo_root=repo_root, image_root=args.image_root,
        protocol_sha256=proto_sha, runner_commit=rc)

    # smoke override of the locked epoch budget (diagnostic only; the full
    # bridge uses the locked max_epochs/patience unchanged)
    t0 = time.time()
    runner.max_epochs = args.epochs
    runner.run_training()
    score = runner.score_anon_real()
    elapsed = time.time() - t0

    runs_root = os.path.join(REPO_ROOT, args.runs_root)
    rd, man = runner.finalize(runs_root=runs_root)

    print(json.dumps({
        "smoke": "PASS",
        "arm": args.arm,
        "seed": args.seed,
        "epochs_run": args.epochs,
        "elapsed_sec": round(elapsed, 1),
        "raw_roc_auc_anon_real": man["raw_roc_auc"],
        "best_epoch": man["best_epoch"],
        "stop_epoch": man["stop_epoch"],
        "predictions_sha256": man["predictions_sha256"][:16] + "...",
        "attacker_best_sha256": man["attacker_best_sha256"][:16] + "...",
        "run_dir": rd,
        "cuda_initialized": torch.cuda.is_initialized(),
    }, indent=1))


if __name__ == "__main__":
    import torch  # noqa: E402  (used only for cuda flag print)
    main()
