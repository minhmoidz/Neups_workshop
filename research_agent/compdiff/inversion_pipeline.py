"""STEP 7F — Inversion & Reconstruction Pipeline

Deterministic DDIM inversion and reconstruction using Diffusers.
Provides:
  - ARM S (Source)
  - ARM R (Resize Control: 256 -> 512 -> 256)
  - ARM C (CompDiff Inversion + Reconstruction)
Utility evaluation for Classification (CheXNet) and Segmentation (UNet Teacher).
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy import ndimage

from research_agent.ibr.frozen_models import (
    load_frozen_classifier,
    load_frozen_segmentation_teacher,
    CLASSIFIER_PATH,
    SEGMENTATION_PATH as SEG_TEACHER_PATH,
)

NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]
STRUCTS = ['Left Lung', 'Right Lung', 'Heart']
PROMPT = "a chest radiograph"


def load_utility_models(device='cuda'):
    print("Loading frozen classifier and segmentation teacher...")
    classifier, _ = load_frozen_classifier(device=device)
    classifier = classifier.to(device)
    seg_teacher, _ = load_frozen_segmentation_teacher(device=device)
    seg_teacher = seg_teacher.to(device)
    return classifier, seg_teacher



def preprocess_source_image(img_path):
    """Loads image, converts to 256x256 grayscale float [0, 1] tensor."""
    img = Image.open(img_path).convert('L')
    img_tensor = TF.to_tensor(img)  # (1, H, W) in [0, 1]
    if img_tensor.shape[1:] != (256, 256):
        img_tensor = TF.resize(img_tensor, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return img_tensor  # (1, 256, 256)


def create_resize_control(img_256):
    """ARM R: 256 -> 512 -> 256 bilinear antialias."""
    img_512 = TF.resize(img_256, [512, 512], interpolation=InterpolationMode.BILINEAR, antialias=True)
    img_rec_256 = TF.resize(img_512, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return img_rec_256.clamp(0.0, 1.0)


def run_compdiff_inversion_and_recon(pipe, inverse_scheduler, forward_scheduler, img_256, device='cuda', num_steps=30):
    """
    Executes:
      1. img_256 -> RGB 512 -> [-1, 1]
      2. VAE encode -> mode -> scale
      3. DDIM inversion 30 steps -> z_T
      4. DDIM reconstruction 30 steps -> z_0_recon
      5. VAE decode -> RGB 512 -> Grayscale -> 256x256
    """
    # 1. Prepare 512 RGB tensor in [-1, 1]
    img_rgb_256 = img_256.repeat(3, 1, 1)  # (3, 256, 256)
    img_rgb_512 = TF.resize(img_rgb_256, [512, 512], interpolation=InterpolationMode.BILINEAR, antialias=True)
    img_input = (img_rgb_512.unsqueeze(0) * 2.0 - 1.0).to(device=device, dtype=pipe.vae.dtype)  # (1, 3, 512, 512)
    
    # 2. Text embedding for "a chest radiograph"
    text_inputs = pipe.tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]  # (1, 77, 1024)
    
    # 3. Deterministic VAE encode (mode)
    with torch.no_grad():
        latent_dist = pipe.vae.encode(img_input).latent_dist
        init_latent = latent_dist.mode() * pipe.vae.config.scaling_factor  # (1, 4, 64, 64)
        
    # 4. DDIM Inversion (0 -> T)
    inverse_scheduler.set_timesteps(num_steps, device=device)
    inv_latents = init_latent.clone()
    
    t_inv_start = time.time()
    with torch.no_grad():
        for t in inverse_scheduler.timesteps:
            noise_pred = pipe.unet(inv_latents, t, encoder_hidden_states=text_embeddings).sample
            inv_latents = inverse_scheduler.step(noise_pred, t, inv_latents).prev_sample
    inversion_time = time.time() - t_inv_start
    z_T = inv_latents
    
    # 5. DDIM Reconstruction (T -> 0)
    forward_scheduler.set_timesteps(num_steps, device=device)
    rec_latents = z_T.clone()
    
    t_rec_start = time.time()
    with torch.no_grad():
        for t in forward_scheduler.timesteps:
            noise_pred = pipe.unet(rec_latents, t, encoder_hidden_states=text_embeddings).sample
            rec_latents = forward_scheduler.step(noise_pred, t, rec_latents, eta=0.0).prev_sample
    recon_time = time.time() - t_rec_start
    z_0_recon = rec_latents
    
    # 6. VAE Decode
    with torch.no_grad():
        decoded_rgb_512 = pipe.vae.decode(z_0_recon / pipe.vae.config.scaling_factor).sample
        decoded_rgb_512 = (decoded_rgb_512 / 2.0 + 0.5).clamp(0.0, 1.0)  # (1, 3, 512, 512)
        
    # 7. Convert to Grayscale & resize to 256x256
    # Luminance weights: 0.2989 R + 0.5870 G + 0.1140 B
    r = decoded_rgb_512[:, 0:1]
    g = decoded_rgb_512[:, 1:2]
    b = decoded_rgb_512[:, 2:3]
    gray_512 = (0.2989 * r + 0.5870 * g + 0.1140 * b).clamp(0.0, 1.0)
    gray_256 = TF.resize(gray_512.squeeze(0), [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
    
    timing = {
        'inversion_sec': inversion_time,
        'recon_sec': recon_time,
        'total_sec': inversion_time + recon_time,
    }
    
    latent_info = {
        'shape': list(init_latent.shape),
        'dtype': str(init_latent.dtype),
        'scaling_factor': float(pipe.vae.config.scaling_factor),
        'z0_norm': float(torch.norm(init_latent).item()),
        'zT_norm': float(torch.norm(z_T).item()),
    }
    
    return gray_256.cpu().to(torch.float32), decoded_rgb_512.squeeze(0).cpu().to(torch.float32), timing, latent_info


def eval_classification_prob(classifier, img_256, device='cuda'):
    """img_256: (1, 256, 256) float [0, 1]. Returns: (14,) numpy array of probs."""
    x = img_256.repeat(3, 1, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    x = TF.resize(x, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        out = classifier(x)
        probs = torch.sigmoid(out).cpu().numpy().flatten()
    return probs


def eval_segmentation_maps(seg_teacher, img_256, device='cuda'):
    """img_256: (1, 256, 256) float [0, 1]. Returns: (3, 256, 256) binary mask [Left Lung, Right Lung, Heart]."""
    x = (img_256.unsqueeze(0).to(device=device, dtype=torch.float32) - 0.5) / 0.5
    with torch.no_grad():
        out = seg_teacher(x)
        probs = torch.sigmoid(out)
        hard_masks = (probs >= 0.5).cpu().numpy()[0].astype(np.uint8)
    return hard_masks




def compute_hd95(mask_a, mask_b):
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float('nan')
    da = ndimage.distance_transform_edt(mask_b == 0)
    db = ndimage.distance_transform_edt(mask_a == 0)
    d_a_to_b = da[mask_a]
    d_b_to_a = db[mask_b]
    return float(max(np.percentile(d_a_to_b, 95), np.percentile(d_b_to_a, 95)))


def compute_seg_metrics_between(mask_pred, mask_true):
    """mask_pred, mask_true: (3, 256, 256) binary uint8."""
    metrics = {}
    for c, struct in enumerate(STRUCTS):
        a = mask_true[c].astype(bool)
        b = mask_pred[c].astype(bool)
        inter = np.logical_and(a, b).sum()
        uni = np.logical_or(a, b).sum()
        dice = float(2.0 * inter / (a.sum() + b.sum() + 1e-8))
        iou = float(inter / (uni + 1e-8))
        hd = compute_hd95(a, b)
        metrics[struct] = {'dice': dice, 'iou': iou, 'hd95': hd}
    
    macro_dice = float(np.mean([metrics[s]['dice'] for s in STRUCTS]))
    macro_iou = float(np.mean([metrics[s]['iou'] for s in STRUCTS]))
    valid_hds = [metrics[s]['hd95'] for s in STRUCTS if not np.isnan(metrics[s]['hd95'])]
    macro_hd = float(np.mean(valid_hds)) if valid_hds else float('nan')
    
    metrics['macro'] = {'dice': macro_dice, 'iou': macro_iou, 'hd95': macro_hd}
    return metrics
