"""STEP 7F — CompDiff Model Manager

Downloads and manages the pretrained CompDiff model from Hugging Face.
Loads StableDiffusionPipeline in fp16 without demographic conditioning.
"""

import hashlib
import json
import os
import subprocess
import time
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler

REPO_ID = "mahmoudibra98/compdiff-chest-xray"
REVISION = "ff145044ca3f525dce25c2fcbe6a3c252ff1d2d1"
MODEL_DIR = os.path.abspath("research_agent/compdiff_model")

CONFIG_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "vae/config.json",
]

LARGE_FILES = [
    ("vae/diffusion_pytorch_model.safetensors", 335 * 1024 * 1024),
    ("text_encoder/model.safetensors", 680 * 1024 * 1024),
    ("unet/diffusion_pytorch_model.safetensors", 3400 * 1024 * 1024),
]


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except Exception:
        return "UNKNOWN"


def compute_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading model {REPO_ID} (commit {REVISION}) to {MODEL_DIR}...")
    base_url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}"
    
    # 1. Download Config Files
    for rel_path in CONFIG_FILES:
        dest_path = os.path.join(MODEL_DIR, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            url = f"{base_url}/{rel_path}"
            cmd = ["curl", "-s", "-L", "-o", dest_path, url]
            subprocess.run(cmd, check=True)
            print(f"  Config ready: {rel_path} ({os.path.getsize(dest_path)} bytes)", flush=True)

    # 2. Download Large Model Weights with Resume
    for rel_path, min_expected_size in LARGE_FILES:
        dest_path = os.path.join(MODEL_DIR, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) >= min_expected_size * 0.95:
            sz_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  Found cached {rel_path} ({sz_mb:.2f} MB)", flush=True)
        else:
            url = f"{base_url}/{rel_path}"
            print(f"  Downloading {rel_path} with resume...", flush=True)
            cmd = ["curl", "-L", "-C", "-", "-o", dest_path, url]
            subprocess.run(cmd, check=True)
            sz_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  -> {rel_path} complete ({sz_mb:.2f} MB)", flush=True)

    # Verify required subdirectories and configs
    required_components = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
    for comp in required_components:
        comp_dir = os.path.join(MODEL_DIR, comp)
        if not os.path.exists(comp_dir):
            raise FileNotFoundError(f"Missing required component directory: {comp_dir}")

    # Compute file stats and SHA256 of configs
    configs = {}
    config_files = [
        'model_index.json',
        'unet/config.json',
        'vae/config.json',
        'scheduler/scheduler_config.json',
        'text_encoder/config.json',
    ]
    for rel_path in config_files:
        full_path = os.path.join(MODEL_DIR, rel_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                content = json.load(f)
            configs[rel_path] = {
                'sha256': compute_file_sha256(full_path),
                'size_bytes': os.path.getsize(full_path),
                'config': content,
            }

    # Calculate total downloaded bytes
    total_bytes = 0
    file_hashes = {}
    for root, _, files in os.walk(MODEL_DIR):
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, MODEL_DIR)
            sz = os.path.getsize(fp)
            total_bytes += sz
            if sz < 10 * 1024 * 1024:  # Hash files under 10MB
                file_hashes[rel] = compute_file_sha256(fp)

    provenance = {
        'repo_id': REPO_ID,
        'revision': REVISION,
        'local_path': MODEL_DIR,
        'total_downloaded_bytes': total_bytes,
        'configs': configs,
        'file_hashes_sample': file_hashes,
    }
    return MODEL_DIR, provenance


def load_compdiff_pipeline(local_path, device='cuda', dtype=torch.float16):
    print(f"Loading StableDiffusionPipeline from {local_path} in {dtype} on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        local_path,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    
    # Freeze all parameters
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False)
    
    # Set up paired schedulers
    scheduler_config = pipe.scheduler.config
    inverse_scheduler = DDIMInverseScheduler.from_config(scheduler_config)
    forward_scheduler = DDIMScheduler.from_config(scheduler_config)
    
    return pipe, inverse_scheduler, forward_scheduler
