"""Classification-utility evaluation: original vs U_PUBLISHED vs D_BDEV.

Reuses the GOVERNED VAL-only classification evaluator unchanged
(m2_dev.eval_classifier_val) — same frozen DenseNet-121, same fold==val,
same fingerprints — swapping ONLY the frozen anonymizer per arm.

--wait-gpu: poll until GPU memory drops below threshold, then run.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(sys.argv[sys.argv.index("--repo-root") + 1]) \
    if "--repo-root" in sys.argv else "/home/minhtt/Neups_workshop"
os.chdir(REPO)
for pth in (HERE, REPO, os.path.join(REPO, "research_agent")):
    if pth not in sys.path:
        sys.path.insert(0, pth)


def gpu_mem_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    try:
        return int(out.splitlines()[0])
    except Exception:
        return 0


def wait_gpu(threshold_mb, poll_sec=120):
    while True:
        used = gpu_mem_used()
        print("[utility] GPU memory used: %d MiB (threshold %d)" %
              (used, threshold_mb), flush=True)
        if used < threshold_mb:
            return
        time.sleep(poll_sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", default="/home/minhtt/datasets/nih/images/")
    ap.add_argument("--repo-root", default=REPO)
    ap.add_argument("--wait-gpu-threshold-mb", type=int, default=3000)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--arms", default="U_PUBLISHED,D_BDEV",
                    help="comma-separated arm names from the locked protocol")
    ap.add_argument("--out-name", default="utility_results.json",
                    help="output filename; a distinct name avoids clobbering "
                         "an existing result set")
    args = ap.parse_args()

    import torch
    from run_p0_bridge import load_protocol, verify_all_artifacts

    protocol, proto_sha = load_protocol()
    rc = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                 cwd=args.repo_root).decode().strip()
    verify_all_artifacts(protocol, args.repo_root)
    print("[utility] governed artifacts verified", flush=True)

    if args.device == "cuda":
        wait_gpu(args.wait_gpu_threshold_mb)

    device = torch.device(args.device)

    from m2_dev.evaluator_common import (
        SCIENTIFIC_IMAGE_ROOT, build_anonymize_fn,
        make_flow_field_components, MU)
    from m2_dev.eval_classifier_val import (
        load_frozen_classifier, classify_val_dataset)
    from networks.UNet_PriCheXyNet import UNet

    assert args.image_root == SCIENTIFIC_IMAGE_ROOT, \
        "scientific image root required"

    config = {"image_path": SCIENTIFIC_IMAGE_ROOT, "image_size": 256}

    # build VAL dataloader ONCE via the governed evaluator internals
    import chexnet.cxr_dataset as CXR
    from m2_dev.evaluator_common import (
        FROZEN_CLASSIFICATION_VAL_N_IMAGES, firewall_check)
    firewall_check("dev")
    dataset = CXR.CXRDataset(path_to_images=args.image_root, fold="val",
                             transform=None, perturbation_type="flow_field")
    assert len(dataset) == FROZEN_CLASSIFICATION_VAL_N_IMAGES
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)

    model = load_frozen_classifier(device)

    default_mu = protocol["attacker_protocol"]["flow_operator"]["mu"]

    def make_anon(arm_spec):
        gen_path = os.path.join(args.repo_root, arm_spec["execution_path"])
        gen = UNet(1, 2, 32).to(device)
        gen.load_state_dict(torch.load(gen_path, map_location=device,
                                       weights_only=False))
        grid_identity, gauss_filter = make_flow_field_components(device, 256)
        # P0_PROTOCOL_V1_2: mu belongs to the generator, not the evaluator.
        # Using the global MU here would deform a mu=0.001 endpoint at 0.01
        # and report the utility of a model that was never trained.
        mu = float(arm_spec.get("mu", default_mu))
        return build_anonymize_fn(gen, grid_identity, gauss_filter, mu), mu

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in protocol["arms"]]
    if unknown:
        raise SystemExit("arms not in the locked protocol: %s" % unknown)
    variants = [("original", None, None)]
    for arm in arms:
        fn, mu = make_anon(protocol["arms"][arm])
        variants.append((arm, fn, mu))

    results = {}
    for name, anon_fn, arm_mu in variants:
        t0 = time.time()
        _, auc_df, macro_auc = classify_val_dataset(
            model, dataloader, anon_fn, "flow_field",
            device=device, batch_size=args.batch_size)
        results[name] = {
            "macro_auc": macro_auc,
            "resolved_mu": arm_mu,
            "per_class": dict(zip(auc_df["label"].tolist(),
                                  [round(a, 4) for a in auc_df["auc"]]))}
        print("[UTILITY DONE] %s macro=%.4f (%.0fs)" %
              (name, macro_auc, time.time() - t0), flush=True)

    results["_meta"] = {"protocol_sha256": proto_sha[:16],
                        "runner_commit": rc[:12],
                        "finished_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    out_dir = os.path.join(args.repo_root,
                           "reproduction/p0_bridge/runs_utility")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, args.out_name), "w") as f:
        json.dump(results, f, indent=1)
    print("[UTILITY COMPLETE]", json.dumps(
        {k: v.get("macro_auc") if isinstance(v, dict) else v
         for k, v in results.items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
