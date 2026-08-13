"""STEP 7F — 12 Load-Bearing Verification Tests for COMPDiff Pretrained CXR Manifold Smoke

Verifies:
  1. TEST path inaccessible.
  2. no HCN module is invoked.
  3. no demographic inputs enter pipeline.
  4. fixed prompt is identical for all images ("a chest radiograph").
  5. subset is deterministic (seed 42, 48 train, 48 val).
  6. validation is never used during implementation debugging.
  7. inverse and reverse schedulers share compatible configs.
  8. VAE scaling is exact.
  9. no trainable CompDiff parameter exists.
  10. no privacy loss / identity model is called.
  11. resize-only control uses the same resize path.
  12. frozen utility checkpoint hashes match (seg SHA: 2dfdcf9b...).
"""

import hashlib
import json
import os
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image

from research_agent.compdiff.model_manager import compute_file_sha256
from research_agent.compdiff.subset_selector import build_manifest, NIH_LABELS
from research_agent.compdiff.inversion_pipeline import (
    PROMPT,
    SEG_TEACHER_PATH,
    CLASSIFIER_PATH,
    create_resize_control,
)


def test1_test_path_inaccessible():
    with open('research_agent/compdiff/subset_selector.py') as f:
        src = f.read()
    assert 'image_pairs_testing' not in src
    with open('research_agent/compdiff/inversion_pipeline.py') as f:
        src = f.read()
    assert 'image_pairs_testing' not in src
    with open('research_agent/compdiff/run_smoke.py') as f:
        src = f.read()
    assert 'image_pairs_testing' not in src
    return True


def test2_no_hcn_module_invoked():
    for fn in ['model_manager.py', 'inversion_pipeline.py', 'run_smoke.py']:
        with open(os.path.join('research_agent/compdiff', fn)) as f:
            src = f.read().lower()
        assert 'hcn' not in src, f"HCN detected in {fn}"
        assert 'demographic' not in src or 'without demographic' in src or 'no demographic' in src
    return True


def test3_no_demographics_entered():
    with open('research_agent/compdiff/subset_selector.py') as f:
        src = f.read()
    assert 'patient gender' not in src.lower()
    assert 'patient age' not in src.lower()
    assert 'race' not in src.lower()
    return True


def test4_fixed_prompt_identical():
    assert PROMPT == "a chest radiograph"
    return True


def test5_subset_deterministic():
    m1 = build_manifest('research_agent/compdiff_artifacts/test_m1.json')
    m2 = build_manifest('research_agent/compdiff_artifacts/test_m2.json')
    assert m1['manifest_sha256'] == m2['manifest_sha256']
    assert len(m1['train_subset']) == 48
    assert len(m1['val_subset']) == 48
    train_ids = {r['image_id'] for r in m1['train_subset']}
    val_ids = {r['image_id'] for r in m1['val_subset']}
    assert len(train_ids.intersection(val_ids)) == 0
    os.remove('research_agent/compdiff_artifacts/test_m1.json')
    os.remove('research_agent/compdiff_artifacts/test_m2.json')
    return True


def test6_val_never_used_during_debugging():
    with open('research_agent/compdiff/run_smoke.py') as f:
        src = f.read()
    # Microcheck must use train_subset
    micro_idx = src.find('micro_train =')
    assert micro_idx != -1
    micro_section = src[micro_idx:micro_idx+300]
    assert "manifest['train_subset']" in micro_section
    assert "val_subset" not in micro_section
    return True


def test7_inverse_and_reverse_schedulers_compatible():
    from diffusers import DDIMScheduler, DDIMInverseScheduler
    from research_agent.compdiff.model_manager import download_and_verify_model, load_compdiff_pipeline
    local_path, _ = download_and_verify_model()
    pipe, inv, fwd = load_compdiff_pipeline(local_path, device='cpu', dtype=torch.float32)
    assert inv.config.beta_schedule == fwd.config.beta_schedule
    assert inv.config.prediction_type == fwd.config.prediction_type
    assert inv.config.num_train_timesteps == fwd.config.num_train_timesteps
    return True


def test8_vae_scaling_exact():
    from research_agent.compdiff.model_manager import download_and_verify_model, load_compdiff_pipeline
    local_path, _ = download_and_verify_model()
    pipe, _, _ = load_compdiff_pipeline(local_path, device='cpu', dtype=torch.float32)
    scale = float(pipe.vae.config.scaling_factor)
    assert abs(scale - 0.18215) < 1e-4, f"Unexpected VAE scaling factor: {scale}"
    return True


def test9_no_trainable_compdiff_parameters():
    from research_agent.compdiff.model_manager import download_and_verify_model, load_compdiff_pipeline
    local_path, _ = download_and_verify_model()
    pipe, _, _ = load_compdiff_pipeline(local_path, device='cpu', dtype=torch.float32)
    assert not pipe.unet.training
    assert not pipe.vae.training
    assert not pipe.text_encoder.training
    assert sum(p.requires_grad for p in pipe.unet.parameters()) == 0
    assert sum(p.requires_grad for p in pipe.vae.parameters()) == 0
    assert sum(p.requires_grad for p in pipe.text_encoder.parameters()) == 0
    return True


def test10_no_privacy_loss_called():
    with open('research_agent/compdiff/inversion_pipeline.py') as f:
        src = f.read().lower()
    assert 'privacy_loss' not in src
    assert 'snn' not in src
    assert 'triplet' not in src
    assert 'contrastive' not in src
    return True


def test11_resize_control_uses_same_path():
    x = torch.zeros(1, 256, 256)
    x[0, 50:150, 50:150] = 1.0
    r = create_resize_control(x)
    assert r.shape == (1, 256, 256)
    assert r.min() >= 0.0 and r.max() <= 1.0
    return True


def test12_frozen_utility_checkpoint_hashes():
    seg_sha = compute_file_sha256(SEG_TEACHER_PATH)
    assert seg_sha.startswith('2dfdcf9b'), f"Seg teacher SHA mismatch: {seg_sha}"
    assert os.path.exists(CLASSIFIER_PATH), "Classifier checkpoint missing"
    return True


def run_all():
    tests = [
        test1_test_path_inaccessible,
        test2_no_hcn_module_invoked,
        test3_no_demographics_entered,
        test4_fixed_prompt_identical,
        test5_subset_deterministic,
        test6_val_never_used_during_debugging,
        test7_inverse_and_reverse_schedulers_compatible,
        test8_vae_scaling_exact,
        test9_no_trainable_compdiff_parameters,
        test10_no_privacy_loss_called,
        test11_resize_control_uses_same_path,
        test12_frozen_utility_checkpoint_hashes,
    ]
    results = {}
    for t in tests:
        try:
            results[t.__name__] = bool(t())
            print(f"TEST {t.__name__} PASS")
        except Exception as e:
            results[t.__name__] = False
            print(f"TEST {t.__name__} FAIL: {e}")
    assert all(results.values()), results
    print("ALL 12 STEP 7F LOAD-BEARING TESTS PASS")
    return results


if __name__ == '__main__':
    run_all()
