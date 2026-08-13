"""STEP 7F — COMPDiff Pretrained CXR Manifold Inversion / Reconstruction Smoke Runner

Orchestrates:
  1. Pinning model & recording provenance
  2. Environment & VRAM smoke check
  3. Subset selection (48 train, 48 val)
  4. TRAIN engineering microcheck (8 images)
  5. Official smoke on 48 TRAIN and 48 VALIDATION images
  6. Evaluation of Classification AUC (CheXNet) and Segmentation (UNet teacher)
  7. Contact sheet generation
  8. Decision gate evaluation
"""

import json
import os
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score

from research_agent.compdiff.model_manager import (
    download_and_verify_model,
    load_compdiff_pipeline,
    compute_file_sha256,
    git_head,
)
from research_agent.compdiff.subset_selector import build_manifest, NIH_LABELS
from research_agent.compdiff.inversion_pipeline import (
    load_utility_models,
    preprocess_source_image,
    create_resize_control,
    run_compdiff_inversion_and_recon,
    eval_classification_prob,
    eval_segmentation_maps,
    compute_seg_metrics_between,
    CLASSIFIER_PATH,
    SEG_TEACHER_PATH,
)

OUT_DIR = 'research_agent/compdiff_artifacts/'
CONTACT_SHEETS_DIR = os.path.join(OUT_DIR, 'contact_sheets')


def create_contact_sheet(img_src, img_res, img_comp, save_path, title=""):
    """Creates a 3-panel horizontal comparison image: Source | Resize | CompDiff."""
    # Convert tensors (1, 256, 256) [0, 1] to PIL images
    src_pil = TF.to_pil_image(img_src)
    res_pil = TF.to_pil_image(img_res)
    comp_pil = TF.to_pil_image(img_comp)
    
    w, h = 256, 256
    sheet = Image.new('RGB', (w * 3, h + 30), color=(30, 30, 30))
    sheet.paste(src_pil.convert('RGB'), (0, 30))
    sheet.paste(res_pil.convert('RGB'), (w, 30))
    sheet.paste(comp_pil.convert('RGB'), (w * 2, 30))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sheet.save(save_path)


def run_single_image(pipe, inv_sched, fwd_sched, classifier, seg_teacher, img_path, device='cuda'):
    # Load Source (ARM S)
    img_src = preprocess_source_image(img_path)
    
    # Create Resize Control (ARM R)
    img_res = create_resize_control(img_src)
    
    # Run CompDiff (ARM C)
    img_comp, comp_rgb_512, timing, latent_info = run_compdiff_inversion_and_recon(
        pipe, inv_sched, fwd_sched, img_src, device=device, num_steps=30
    )
    
    # Numerical Sanity checks on CompDiff output
    comp_np = img_comp.numpy()
    finite = bool(np.isfinite(comp_np).all())
    out_min = float(comp_np.min())
    out_max = float(comp_np.max())
    out_mean = float(comp_np.mean())
    out_var = float(comp_np.var())
    src_mean = float(img_src.numpy().mean())
    mean_shift = float(out_mean - src_mean)
    sat_frac = float(((comp_np <= 0.001) | (comp_np >= 0.999)).mean())
    
    sanity = {
        'finite': finite,
        'min': out_min,
        'max': out_max,
        'mean': out_mean,
        'variance': out_var,
        'mean_intensity_shift': mean_shift,
        'saturation_fraction': sat_frac,
    }
    
    # Classification Probabilities
    p_src = eval_classification_prob(classifier, img_src, device=device)
    p_res = eval_classification_prob(classifier, img_res, device=device)
    p_comp = eval_classification_prob(classifier, img_comp, device=device)
    
    # Segmentation Maps
    m_src = eval_segmentation_maps(seg_teacher, img_src, device=device)
    m_res = eval_segmentation_maps(seg_teacher, img_res, device=device)
    m_comp = eval_segmentation_maps(seg_teacher, img_comp, device=device)
    
    # Segmentation Metrics
    seg_res_vs_src = compute_seg_metrics_between(m_res, m_src)
    seg_comp_vs_src = compute_seg_metrics_between(m_comp, m_src)
    
    return {
        'img_src': img_src,
        'img_res': img_res,
        'img_comp': img_comp,
        'p_src': p_src,
        'p_res': p_res,
        'p_comp': p_comp,
        'm_src': m_src,
        'm_res': m_res,
        'm_comp': m_comp,
        'seg_res_vs_src': seg_res_vs_src,
        'seg_comp_vs_src': seg_comp_vs_src,
        'timing': timing,
        'latent_info': latent_info,
        'sanity': sanity,
    }


def compute_subset_metrics(results_list, records, evaluable_labels):
    # Classification AUC across evaluable labels
    y_true = np.array([[r['labels'][lbl] for lbl in NIH_LABELS] for r in records])  # (N, 14)
    p_src_all = np.array([res['p_src'] for res in results_list])                      # (N, 14)
    p_res_all = np.array([res['p_res'] for res in results_list])                      # (N, 14)
    p_comp_all = np.array([res['p_comp'] for res in results_list])                    # (N, 14)
    
    auc_src = {}
    auc_res = {}
    auc_comp = {}
    delta_class_comp = {}
    delta_class_res = {}
    
    for idx, lbl in enumerate(NIH_LABELS):
        if lbl in evaluable_labels:
            try:
                a_s = float(roc_auc_score(y_true[:, idx], p_src_all[:, idx]))
                a_r = float(roc_auc_score(y_true[:, idx], p_res_all[:, idx]))
                a_c = float(roc_auc_score(y_true[:, idx], p_comp_all[:, idx]))
                auc_src[lbl] = a_s
                auc_res[lbl] = a_r
                auc_comp[lbl] = a_c
                delta_class_comp[lbl] = a_c - a_s
                delta_class_res[lbl] = a_r - a_s
            except Exception as e:
                print(f"Warning: could not compute AUC for {lbl}: {e}")
                
    macro_auc_src = float(np.mean(list(auc_src.values()))) if auc_src else 0.0
    macro_auc_res = float(np.mean(list(auc_res.values()))) if auc_res else 0.0
    macro_auc_comp = float(np.mean(list(auc_comp.values()))) if auc_comp else 0.0
    macro_delta_class_comp = macro_auc_comp - macro_auc_src
    macro_delta_class_res = macro_auc_res - macro_auc_src
    
    # Segmentation Metrics averages
    structs = ['Left Lung', 'Right Lung', 'Heart']
    comp_seg_by_struct = {s: {'dice': [], 'iou': [], 'hd95': []} for s in structs}
    res_seg_by_struct = {s: {'dice': [], 'iou': [], 'hd95': []} for s in structs}
    
    for res in results_list:
        for s in structs:
            comp_seg_by_struct[s]['dice'].append(res['seg_comp_vs_src'][s]['dice'])
            comp_seg_by_struct[s]['iou'].append(res['seg_comp_vs_src'][s]['iou'])
            if not np.isnan(res['seg_comp_vs_src'][s]['hd95']):
                comp_seg_by_struct[s]['hd95'].append(res['seg_comp_vs_src'][s]['hd95'])
                
            res_seg_by_struct[s]['dice'].append(res['seg_res_vs_src'][s]['dice'])
            res_seg_by_struct[s]['iou'].append(res['seg_res_vs_src'][s]['iou'])
            if not np.isnan(res['seg_res_vs_src'][s]['hd95']):
                res_seg_by_struct[s]['hd95'].append(res['seg_res_vs_src'][s]['hd95'])
                
    comp_seg_summary = {}
    for s in structs:
        comp_seg_summary[s] = {
            'dice_mean': float(np.mean(comp_seg_by_struct[s]['dice'])),
            'iou_mean': float(np.mean(comp_seg_by_struct[s]['iou'])),
            'hd95_mean': float(np.mean(comp_seg_by_struct[s]['hd95'])) if comp_seg_by_struct[s]['hd95'] else 0.0,
        }
    comp_macro_dice = float(np.mean([comp_seg_summary[s]['dice_mean'] for s in structs]))
    comp_macro_iou = float(np.mean([comp_seg_summary[s]['iou_mean'] for s in structs]))
    comp_macro_hd95 = float(np.mean([comp_seg_summary[s]['hd95_mean'] for s in structs]))
    comp_seg_summary['macro'] = {'dice': comp_macro_dice, 'iou': comp_macro_iou, 'hd95': comp_macro_hd95}
    
    res_seg_summary = {}
    for s in structs:
        res_seg_summary[s] = {
            'dice_mean': float(np.mean(res_seg_by_struct[s]['dice'])),
            'iou_mean': float(np.mean(res_seg_by_struct[s]['iou'])),
            'hd95_mean': float(np.mean(res_seg_by_struct[s]['hd95'])) if res_seg_by_struct[s]['hd95'] else 0.0,
        }
    res_macro_dice = float(np.mean([res_seg_summary[s]['dice_mean'] for s in structs]))
    res_macro_iou = float(np.mean([res_seg_summary[s]['iou_mean'] for s in structs]))
    res_macro_hd95 = float(np.mean([res_seg_summary[s]['hd95_mean'] for s in structs]))
    res_seg_summary['macro'] = {'dice': res_macro_dice, 'iou': res_macro_iou, 'hd95': res_macro_hd95}
    
    # Timing summary
    inv_times = [r['timing']['inversion_sec'] for r in results_list]
    rec_times = [r['timing']['recon_sec'] for r in results_list]
    
    return {
        'classification': {
            'evaluable_labels_count': len(auc_src),
            'evaluable_labels': list(auc_src.keys()),
            'auc_source': auc_src,
            'auc_resize_control': auc_res,
            'auc_compdiff': auc_comp,
            'delta_class_compdiff': delta_class_comp,
            'delta_class_resize': delta_class_res,
            'macro_auc_source': macro_auc_src,
            'macro_auc_resize_control': macro_auc_res,
            'macro_auc_compdiff': macro_auc_comp,
            'macro_delta_class_compdiff': macro_delta_class_comp,
            'macro_delta_class_resize': macro_delta_class_res,
        },
        'segmentation_compdiff_vs_source': comp_seg_summary,
        'segmentation_resize_vs_source': res_seg_summary,
        'timing': {
            'mean_inversion_sec': float(np.mean(inv_times)),
            'mean_recon_sec': float(np.mean(rec_times)),
            'mean_total_sec_per_img': float(np.mean(inv_times) + np.mean(rec_times)),
            'total_subset_sec': float(np.sum(inv_times) + np.sum(rec_times)),
        }
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CONTACT_SHEETS_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Record Initial Environment & VRAM
    torch.cuda.empty_cache()
    idle_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    
    env_info = {
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'idle_vram_mb': idle_vram_mb,
    }
    print("Environment:", env_info)
    
    # 1. Download & Verify Model
    local_path, model_provenance = download_and_verify_model()
    
    # 2. Load Pipeline & Schedulers
    pipe, inv_sched, fwd_sched = load_compdiff_pipeline(local_path, device=device, dtype=torch.float16)
    loaded_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    env_info['loaded_model_vram_mb'] = loaded_vram_mb
    print(f"Model loaded. VRAM: {loaded_vram_mb:.2f} MB")
    
    # Load Utility Models
    classifier, seg_teacher = load_utility_models(device=device)
    
    # Verify Segmentation Teacher Checkpoint SHA256
    seg_sha = compute_file_sha256(SEG_TEACHER_PATH)
    assert seg_sha.startswith('2dfdcf9b'), f"Segmentation teacher SHA256 mismatch: {seg_sha}"
    print(f"Verified segmentation teacher SHA256: {seg_sha}")
    
    # 3. Deterministic Subset Selection
    manifest = build_manifest()
    
    # 4. TRAIN-Only Engineering Microcheck (8 images)
    print("\n=== RUNNING 8-IMAGE TRAIN ENGINEERING MICROCHECK ===")
    micro_train = manifest['train_subset'][:8]
    for idx, r in enumerate(micro_train):
        t0 = time.time()
        res = run_single_image(pipe, inv_sched, fwd_sched, classifier, seg_teacher, r['image_path'], device=device)
        print(f"Microcheck [{idx+1}/8] {r['image_id']}: finite={res['sanity']['finite']} "
              f"inversion={res['timing']['inversion_sec']:.2f}s recon={res['timing']['recon_sec']:.2f}s "
              f"Dice={res['seg_comp_vs_src']['macro']['dice']:.4f}")
        assert res['sanity']['finite'], f"Non-finite values detected in microcheck on {r['image_id']}"
    print("TRAIN ENGINEERING MICROCHECK PASSED SUCCESSFULLY.\n")
    
    # 5. Full 48-Image TRAIN Smoke Run
    print("=== RUNNING FULL 48-IMAGE TRAIN SMOKE RUN ===")
    train_results = []
    train_start_time = time.time()
    for idx, r in enumerate(manifest['train_subset']):
        res = run_single_image(pipe, inv_sched, fwd_sched, classifier, seg_teacher, r['image_path'], device=device)
        train_results.append(res)
        if (idx + 1) % 12 == 0 or idx == 47:
            print(f"  TRAIN progress: [{idx+1}/48] images processed")
    train_total_time = time.time() - train_start_time
    train_metrics = compute_subset_metrics(train_results, manifest['train_subset'], manifest['train_evaluable_labels'])
    train_metrics['timing']['wall_clock_total_sec'] = train_total_time
    print(f"TRAIN completed in {train_total_time:.2f}s. Macro Dice: {train_metrics['segmentation_compdiff_vs_source']['macro']['dice']:.4f}, Delta Class: {train_metrics['classification']['macro_delta_class_compdiff']:.4f}")
    
    # 6. Full 48-Image VALIDATION Smoke Run
    print("\n=== RUNNING FULL 48-IMAGE VALIDATION SMOKE RUN ===")
    val_results = []
    val_start_time = time.time()
    for idx, r in enumerate(manifest['val_subset']):
        res = run_single_image(pipe, inv_sched, fwd_sched, classifier, seg_teacher, r['image_path'], device=device)
        val_results.append(res)
        # Generate and save contact sheet
        cs_path = os.path.join(CONTACT_SHEETS_DIR, f"val_{idx:02d}_{r['image_id']}")
        create_contact_sheet(res['img_src'], res['img_res'], res['img_comp'], cs_path, title=f"Val {idx:02d}: {r['image_id']}")
        if (idx + 1) % 12 == 0 or idx == 47:
            print(f"  VAL progress: [{idx+1}/48] images processed")
    val_total_time = time.time() - val_start_time
    val_metrics = compute_subset_metrics(val_results, manifest['val_subset'], manifest['val_evaluable_labels'])
    val_metrics['timing']['wall_clock_total_sec'] = val_total_time
    print(f"VAL completed in {val_total_time:.2f}s. Macro Dice: {val_metrics['segmentation_compdiff_vs_source']['macro']['dice']:.4f}, Delta Class: {val_metrics['classification']['macro_delta_class_compdiff']:.4f}")
    
    # Peak VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    env_info['peak_vram_mb'] = peak_vram_mb
    print(f"Peak VRAM during execution: {peak_vram_mb:.2f} MB")
    
    # 7. Decision Gate Evaluation on VALIDATION
    delta_class = val_metrics['classification']['macro_delta_class_compdiff']
    val_dice = val_metrics['segmentation_compdiff_vs_source']['macro']['dice']
    val_iou = val_metrics['segmentation_compdiff_vs_source']['macro']['iou']
    val_hd95 = val_metrics['segmentation_compdiff_vs_source']['macro']['hd95']
    all_finite = all(res['sanity']['finite'] for res in val_results)
    
    gate_delta_class_pass = bool(delta_class >= -0.020)
    gate_dice_pass = bool(val_dice >= 0.930)
    gate_iou_pass = bool(val_iou >= 0.870)
    gate_hd95_pass = bool(val_hd95 <= 3.0)
    
    all_gates_pass = gate_delta_class_pass and gate_dice_pass and gate_iou_pass and gate_hd95_pass and all_finite
    decision = "COMPDiff MANIFOLD SMOKE: PASS" if all_gates_pass else "COMPDiff MANIFOLD SMOKE: FAIL"
    
    print("\n==================================================")
    print("PREDECLARED DECISION GATE EVALUATION (VALIDATION):")
    print(f"  1. Delta Macro AUC: {delta_class:+.5f} (Target: >= -0.020) -> {'PASS' if gate_delta_class_pass else 'FAIL'}")
    print(f"  2. Macro Dice:      {val_dice:.5f} (Target: >= 0.930)  -> {'PASS' if gate_dice_pass else 'FAIL'}")
    print(f"  3. Macro IoU:       {val_iou:.5f} (Target: >= 0.870)  -> {'PASS' if gate_iou_pass else 'FAIL'}")
    print(f"  4. Macro HD95:      {val_hd95:.2f} px (Target: <= 3.0 px) -> {'PASS' if gate_hd95_pass else 'FAIL'}")
    print(f"  5. Numerical sanity: all finite = {all_finite}")
    print(f"DECISION: {decision}")
    print("==================================================\n")
    
    # Save full summary JSON
    summary = {
        'step': '7F',
        'title': 'COMPDiff Pretrained CXR Manifold Inversion / Reconstruction Smoke',
        'decision': decision,
        'all_gates_passed': all_gates_pass,
        'gates': {
            'delta_class': {'observed': delta_class, 'threshold': -0.020, 'passed': gate_delta_class_pass},
            'macro_dice': {'observed': val_dice, 'threshold': 0.930, 'passed': gate_dice_pass},
            'macro_iou': {'observed': val_iou, 'threshold': 0.870, 'passed': gate_iou_pass},
            'macro_hd95': {'observed': val_hd95, 'threshold': 3.0, 'passed': gate_hd95_pass},
            'numerical_finite': all_finite,
        },
        'environment': env_info,
        'model_provenance': model_provenance,
        'subset_manifest_sha256': manifest['manifest_sha256'],
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'contact_sheets_count': len(val_results),
        'commit': git_head(),
    }
    
    with open('research_agent/17_COMPDiff_INVERSION_SMOKE_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved research_agent/17_COMPDiff_INVERSION_SMOKE_summary.json")
    
    return summary


if __name__ == '__main__':
    main()
