"""Record generator provenance for the corrected baseline training run.

Writes generator_provenance.json into the experiment archive. Contains ONLY
train/validation-side provenance — never TEST metrics.
"""

import argparse
import hashlib
import json
import os
import sys
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_id', required=True)
    ap.add_argument('--config_path', required=True)
    ap.add_argument('--checkpoint_path', required=True)
    ap.add_argument('--git_commit', required=True)
    ap.add_argument('--run_start_timestamp', required=True)
    ap.add_argument('--run_end_timestamp', required=True)
    ap.add_argument('--best_epoch', required=True)
    ap.add_argument('--best_val_total_loss', required=True)
    ap.add_argument('--out_path', required=True)
    args = ap.parse_args()

    config_sha = sha256_file(args.config_path)
    checkpoint_sha = sha256_file(args.checkpoint_path)
    with open(args.config_path) as f:
        config = json.load(f)

    provenance = {
        "experiment_id": args.experiment_id,
        "git_commit": args.git_commit,
        "config_path": args.config_path,
        "config_sha256": config_sha,
        "transform_mode": config.get("transform_mode", "legacy"),
        "mu": config.get("mu"),
        "stochastic_lambda": config.get("stochastic_lambda", 0.0),
        "generator_architecture": "UNet(1, 2, 32) — flow_field (PriCheXy-Net UNet, init_features=32)",
        "initialization_description": "load ./networks/pretrained_generator_prichexy_net.pth (canonical baseline_fixed init)",
        "optimizer": "Adam (generator); Adam (verification models); SGD(momentum=0.9, wd=1e-4) (AC classifier)",
        "learning_rate": config.get("learning_rate"),
        "scheduler": "none",
        "batch_size": config.get("batch_size"),
        "epochs": config.get("max_epochs"),
        "gradient_accumulation_settings": {
            "accumulation_steps": config.get("accumulation_steps", 1),
            "h2_zero_grad_placement_fix": True,
        },
        "train_split": {
            "path": "image_pairs/image_pairs_training_10000.txt",
            "sha256": sha256_file("image_pairs/image_pairs_training_10000.txt"),
        },
        "validation_split": {
            "path": "image_pairs/image_pairs_validation_2000.txt",
            "sha256": sha256_file("image_pairs/image_pairs_validation_2000.txt"),
        },
        "protocol_documents": {
            "01_ADAPTIVE_REID_PROTOCOL.md": sha256_file("research_agent/01_ADAPTIVE_REID_PROTOCOL.md"),
            "01B_PROTOCOL_AMENDMENT.md": sha256_file("research_agent/01B_PROTOCOL_AMENDMENT.md"),
            "03A_REAL_CORRECTED_SMOKE.md": sha256_file("research_agent/03A_REAL_CORRECTED_SMOKE.md"),
        },
        "seed_information": {
            "seed": config.get("seed", 42),
            "rng_pinning": "seed_all(): python.random, numpy, torch CPU, torch CUDA, cudnn.benchmark=False",
        },
        "checkpoint_selection_criterion": {
            "monitored_quantity": "validation total loss",
            "direction": "min",
        },
        "best_epoch": int(args.best_epoch),
        "best_val_total_loss": float(args.best_val_total_loss),
        "generator_checkpoint_path": args.checkpoint_path,
        "generator_checkpoint_sha256": checkpoint_sha,
        "run_start_timestamp": args.run_start_timestamp,
        "run_end_timestamp": args.run_end_timestamp,
        "test_firewall": "no test split constructed, no test pair file opened, no test metric computed",
    }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == '__main__':
    sys.exit(main())
