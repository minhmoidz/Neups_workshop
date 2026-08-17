"""M1.4c §9 / §35 CUDA Determinism & Micro-Certification.

Executes 3 identical runs of a 1-batch B_dev update from identical initial states,
tensors, and seeds on CUDA.
Measures max absolute difference in:
- losses (AC BCE, privacy, total)
- generator gradients
- verifier gradients
- classifier gradients
- post-step parameters
"""
import copy
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from networks.UNet_PriCheXyNet import UNet
from networks.SiameseNetwork import SiameseNetwork
from m2_dev.evaluator_common import (
    make_flow_field_components,
    anonymize,
    snn_preprocess,
    classifier_preprocess,
    FROZEN_CLASSIFIER_PATH,
    FROZEN_VERIFIER_PATH,
    INITIAL_GENERATOR_PATH,
    MU,
)
from research_agent.m0_port.ACLoss import ACLoss
from utils.VerificationLoss import VerificationLoss


def characterize_cuda_environment():
    """Record GPU, CUDA, cuDNN, and torch environment specifications."""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'gpu_model': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'device_capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        'total_vram_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else None,
    }
    return info


def test_strict_deterministic_algorithm_support(device=None):
    """Test if strict torch.use_deterministic_algorithms(True) works DURING the actual forward/backward ops."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    prev = torch.are_deterministic_algorithms_enabled()
    supported = True
    error_msg = None
    try:
        torch.use_deterministic_algorithms(True)
        img_size = 64
        bs = 2
        torch.manual_seed(999)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(999)
        g = UNet(1, 2, 32).to(device)
        from torchvision.models import densenet121
        clf = densenet121(num_classes=14).to(device)
        clf.classifier = nn.Sequential(clf.classifier, nn.Sigmoid())
        ver = SiameseNetwork().to(device)

        opt_g = optim.Adam(g.parameters(), lr=1e-4)
        opt_clf = optim.SGD(filter(lambda p: p.requires_grad, clf.parameters()), lr=1e-4, momentum=0.9, weight_decay=1e-4)
        opt_ver = optim.Adam(ver.parameters(), lr=1e-4)

        crit_ac = nn.BCELoss().to(device)
        crit_ver = nn.BCEWithLogitsLoss().to(device)

        x1 = torch.rand(bs, 1, img_size, img_size, device=device)
        x2 = torch.rand(bs, 1, img_size, img_size, device=device)
        y_clf = torch.randint(0, 2, (bs, 14), dtype=torch.float32, device=device)
        y_id = torch.tensor([1.0, 0.0], device=device)

        grid_id, gauss = make_flow_field_components(device, image_size=img_size)
        resize_224 = transforms.Resize((224, 224))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Step G
        g.train()
        fakes_1 = anonymize(x1, g, grid_id, gauss, mu=MU)
        ac_loss_mod = ACLoss(ac_model=clf, feature_loss_weight=0.0)
        ac_bce = ac_loss_mod(fakes_1, y_clf)
        in1_snn_g = normalize(fakes_1.expand(-1, 3, -1, -1))
        in2_snn_g = normalize(x2.expand(-1, 3, -1, -1))
        ver_logits_g = ver(in1_snn_g, in2_snn_g).squeeze()
        privacy_term = F.softplus(ver_logits_g).mean()
        tot = ac_bce + privacy_term
        opt_g.zero_grad()
        tot.backward()
        opt_g.step()

        # Step Verifier
        ver.train()
        in1_snn_v = normalize(fakes_1.detach().expand(-1, 3, -1, -1))
        in2_snn_v = normalize(x2.expand(-1, 3, -1, -1))
        ver_logits_v = ver(in1_snn_v, in2_snn_v).squeeze()
        loss_ver = crit_ver(ver_logits_v, y_id.type_as(ver_logits_v))
        opt_ver.zero_grad()
        loss_ver.backward()
        opt_ver.step()

        # Step AC
        clf.train()
        in_ac_c = normalize(resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))
        ac_probs_c = clf(in_ac_c)
        loss_ac = crit_ac(ac_probs_c, y_clf)
        opt_clf.zero_grad()
        loss_ac.backward()
        opt_clf.step()
    except Exception as e:
        supported = False
        error_msg = str(e)
    finally:
        torch.use_deterministic_algorithms(prev)

    return supported, error_msg


def run_single_cuda_step(seed=42):
    """Execute one deterministic step on CUDA and return metrics & gradients."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    img_size = 64
    bs = 2

    # Fixed seed for models and inputs
    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)

    g = UNet(1, 2, 32).to(device)
    from torchvision.models import densenet121
    clf = densenet121(num_classes=14).to(device)
    clf.classifier = nn.Sequential(clf.classifier, nn.Sigmoid())
    ver = SiameseNetwork().to(device)

    opt_g = optim.Adam(g.parameters(), lr=1e-4)
    opt_clf = optim.SGD(filter(lambda p: p.requires_grad, clf.parameters()), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    opt_ver = optim.Adam(ver.parameters(), lr=1e-4)

    crit_ac = nn.BCELoss().to(device)
    crit_ver = nn.BCEWithLogitsLoss().to(device)

    # Fixed inputs
    torch.manual_seed(54321)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(54321)
    x1 = torch.rand(bs, 1, img_size, img_size, device=device)
    x2 = torch.rand(bs, 1, img_size, img_size, device=device)
    y_clf = torch.randint(0, 2, (bs, 14), dtype=torch.float32, device=device)
    y_id = torch.tensor([1.0, 0.0], device=device)

    grid_id, gauss = make_flow_field_components(device, image_size=img_size)
    resize_224 = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 1. Generator step
    g.train()
    fakes_1 = anonymize(x1, g, grid_id, gauss, mu=MU)

    ac_loss_mod = ACLoss(ac_model=clf, feature_loss_weight=0.0)
    ac_bce = ac_loss_mod(fakes_1, y_clf)

    in1_snn_g = normalize(fakes_1.expand(-1, 3, -1, -1))
    in2_snn_g = normalize(x2.expand(-1, 3, -1, -1))
    ver_logits_g = ver(in1_snn_g, in2_snn_g).squeeze()
    privacy_term = F.softplus(ver_logits_g).mean()

    total_loss = ac_bce + privacy_term
    opt_g.zero_grad()
    total_loss.backward()
    gen_grads = [p.grad.detach().clone() for p in g.parameters() if p.grad is not None]
    opt_g.step()
    gen_params_post = [p.detach().clone() for p in g.parameters()]

    # 2. Verifier step
    ver.train()
    in1_snn_v = normalize(fakes_1.detach().expand(-1, 3, -1, -1))
    in2_snn_v = normalize(x2.expand(-1, 3, -1, -1))
    ver_logits_v = ver(in1_snn_v, in2_snn_v).squeeze()
    loss_ver = crit_ver(ver_logits_v, y_id.type_as(ver_logits_v))
    opt_ver.zero_grad()
    loss_ver.backward()
    ver_grads = [p.grad.detach().clone() for p in ver.parameters() if p.grad is not None]
    opt_ver.step()
    ver_params_post = [p.detach().clone() for p in ver.parameters()]
    ver.eval()

    # 3. Classifier step
    clf.train()
    in_ac_c = normalize(resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))
    ac_probs_c = clf(in_ac_c)
    loss_ac = crit_ac(ac_probs_c, y_clf)
    opt_clf.zero_grad()
    loss_ac.backward()
    clf_grads = [p.grad.detach().clone() for p in clf.parameters() if p.requires_grad and p.grad is not None]
    opt_clf.step()
    clf_params_post = [p.detach().clone() for p in clf.parameters() if p.requires_grad]
    clf.eval()

    return {
        'ac_bce': ac_bce.item(),
        'privacy_term': privacy_term.item(),
        'total_loss': total_loss.item(),
        'gen_grads': gen_grads,
        'gen_params_post': gen_params_post,
        'ver_grads': ver_grads,
        'ver_params_post': ver_params_post,
        'clf_grads': clf_grads,
        'clf_params_post': clf_params_post,
    }


def measure_cuda_reproducibility_envelope(n_runs=3):
    """Run n_runs identical steps and compute max absolute differences."""
    supported, error_detail = test_strict_deterministic_algorithm_support()
    results = [run_single_cuda_step(seed=42) for _ in range(n_runs)]

    # Pairwise comparisons
    max_loss_diff = 0.0
    max_gen_grad_diff = 0.0
    max_gen_param_diff = 0.0
    max_ver_grad_diff = 0.0
    max_ver_param_diff = 0.0
    max_clf_grad_diff = 0.0
    max_clf_param_diff = 0.0

    base = results[0]
    for r in results[1:]:
        max_loss_diff = max(max_loss_diff, abs(base['total_loss'] - r['total_loss']))
        for g1, g2 in zip(base['gen_grads'], r['gen_grads']):
            max_gen_grad_diff = max(max_gen_grad_diff, (g1 - g2).abs().max().item())
        for p1, p2 in zip(base['gen_params_post'], r['gen_params_post']):
            max_gen_param_diff = max(max_gen_param_diff, (p1 - p2).abs().max().item())
        for g1, g2 in zip(base['ver_grads'], r['ver_grads']):
            max_ver_grad_diff = max(max_ver_grad_diff, (g1 - g2).abs().max().item())
        for p1, p2 in zip(base['ver_params_post'], r['ver_params_post']):
            max_ver_param_diff = max(max_ver_param_diff, (p1 - p2).abs().max().item())
        for g1, g2 in zip(base['clf_grads'], r['clf_grads']):
            max_clf_grad_diff = max(max_clf_grad_diff, (g1 - g2).abs().max().item())
        for p1, p2 in zip(base['clf_params_post'], r['clf_params_post']):
            max_clf_param_diff = max(max_clf_param_diff, (p1 - p2).abs().max().item())

    env = {
        'n_runs': n_runs,
        'max_loss_diff': max_loss_diff,
        'max_gen_grad_diff': max_gen_grad_diff,
        'max_gen_param_diff': max_gen_param_diff,
        'max_ver_grad_diff': max_ver_grad_diff,
        'max_ver_param_diff': max_ver_param_diff,
        'max_clf_grad_diff': max_clf_grad_diff,
        'max_clf_param_diff': max_clf_param_diff,
        'deterministic_algo_supported': supported,
        'unsupported_operation_error': error_detail,
    }
    return env


if __name__ == '__main__':
    info = characterize_cuda_environment()
    print("CUDA Environment:", json.dumps(info, indent=2))
    env = measure_cuda_reproducibility_envelope(3)
    print("CUDA Reproducibility Envelope (3 runs):", json.dumps(env, indent=2))
