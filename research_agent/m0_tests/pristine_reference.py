"""F11 (M1.4c): Truly independent pristine upstream reference implementation.

This module implements minimal upstream training semantics from scratch WITHOUT
importing any production code under test:
  - m2_dev.anonymizer_runner
  - m2_dev.evaluator_common.anonymize
  - m2_dev.evaluator_common.make_flow_field_components
  - research_agent.m0_port.ACLoss

All functions are self-contained re-implementations from the pristine upstream
commit semantics, suitable for independent parity verification.

Pristine upstream reference commit: 29245d1f71571898d9527417df4ae3f63a8695f6
"""
import copy
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# 15A - Reference Operator
# ---------------------------------------------------------------------------
def pristine_identity_grid(image_size, device):
    """Reimplement identity grid exactly as upstream Agent.py."""
    d = torch.linspace(-1, 1, image_size)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    return grid_identity


def pristine_gaussian_kernel(channels=2, kernel_size=9, sigma=2.0, device='cpu'):
    """Reimplement Gaussian kernel construction exactly as upstream GaussianSmoothing."""
    dim = 2
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in kernel_size], 
        indexing='ij'
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    conv = nn.Conv2d(channels, channels, kernel_size[0], groups=channels,
                     bias=False, padding=(kernel_size[0] - 1) // 2)
    conv.weight.data = kernel
    conv.weight.requires_grad = False
    return conv.to(device)


def pristine_anonymize(image, generator, grid_identity, gauss_filter, mu=0.01):
    """Reimplement legacy flow_field anonymization operator.

    grid_identity - mu * generator_output
    zero-padded depthwise Gaussian convolution
    grid_sample border, align_corners=True
    """
    grids = generator(image)
    grids = grid_identity - mu * grids
    grids = gauss_filter(grids)
    grids = grids.permute(0, 2, 3, 1)
    return F.grid_sample(image, grids, padding_mode='border', align_corners=True)


# ---------------------------------------------------------------------------
# 15C - Reference Privacy Loss
# ---------------------------------------------------------------------------
def pristine_privacy_loss_float64(verifier_logits):
    """Direct implementation: sigmoid -> -log(1-p) using float64 for precision.

    This is mathematically equivalent to F.softplus(z):
        softplus(z) = log(1 + exp(z)) = -log(1 - sigmoid(z))
    """
    z = verifier_logits.to(dtype=torch.float64)
    p = torch.sigmoid(z)
    privacy = -torch.log(1.0 - p + 1e-45)
    return privacy.mean().to(dtype=verifier_logits.dtype)


def pristine_privacy_loss_softplus(verifier_logits):
    """Reference using F.softplus -- should match pristine_privacy_loss_float64."""
    return F.softplus(verifier_logits).mean()


# ---------------------------------------------------------------------------
# 15D/15E - Full One-Step Parity Reference
# ---------------------------------------------------------------------------
def pristine_one_step(generator, ac_model, verifier, inputs1, inputs2,
                      labels, labels_id, mu=0.01, lr=1e-4, image_size=None):
    """Fully independent one-step reference. Does NOT import any production code.

    Returns dict with all intermediate values for parity comparison.
    """
    device = inputs1.device
    if image_size is None:
        image_size = inputs1.shape[-1]

    # Build flow field components from scratch
    grid_identity = pristine_identity_grid(image_size, device)
    gauss_filter = pristine_gaussian_kernel(channels=2, kernel_size=9, sigma=2.0, device=device)

    # Transforms
    resize_224 = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 15D: Independent optimizer instantiation
    opt_g = optim.Adam(generator.parameters(), lr=lr)
    opt_ver = optim.Adam(verifier.parameters(), lr=lr)
    opt_ac = optim.SGD(
        filter(lambda p: p.requires_grad, ac_model.parameters()),
        lr=lr, momentum=0.9, weight_decay=1e-4
    )

    crit_ac = nn.BCELoss().to(device)
    crit_ver = nn.BCEWithLogitsLoss().to(device)

    # Ensure critics are in eval mode during G forward
    ac_model.eval()
    verifier.eval()

    # Step 1: Generator forward (anonymize)
    generator.train()
    fakes_1 = pristine_anonymize(inputs1, generator, grid_identity, gauss_filter, mu)

    # Step 2: AC BCE Loss using deepcopy (pristine upstream semantics)
    # The frozen classifier head ALREADY ends with Sigmoid(), so its output is
    # a probability in [0,1] and upstream applies nn.BCELoss() directly.
    ac_loss_model = copy.deepcopy(ac_model)
    ac_loss_model.eval()
    # Strip final Sigmoid for generator's logit-based AC loss
    if hasattr(ac_loss_model, 'classifier') and isinstance(ac_loss_model.classifier, nn.Sequential):
        layers = list(ac_loss_model.classifier.children())
        if len(layers) > 0 and isinstance(layers[-1], nn.Sigmoid):
            ac_loss_model.classifier = nn.Sequential(*layers[:-1])

    ac_preprocessed = normalize(resize_224(fakes_1.expand(-1, 3, -1, -1)))
    ac_features = ac_loss_model.features(ac_preprocessed)
    ac_features = F.relu(ac_features, inplace=True)
    ac_features = F.adaptive_avg_pool2d(ac_features, (1, 1))
    ac_features = torch.flatten(ac_features, 1)
    ac_logits = ac_loss_model.classifier(ac_features)
    ac_bce = nn.BCEWithLogitsLoss()(ac_logits, labels)

    # Step 3: Verifier Privacy Loss (anon/real) using independent probability-space formulation
    in1_snn_g = normalize(fakes_1.expand(-1, 3, -1, -1))
    in2_snn_g = normalize(inputs2.expand(-1, 3, -1, -1))
    ver_logits_g = verifier(in1_snn_g, in2_snn_g).squeeze()
    privacy_term = pristine_privacy_loss_float64(ver_logits_g)

    # Step 4: Generator total loss and step
    total_loss = 1.0 * ac_bce + 1.0 * privacy_term
    opt_g.zero_grad()
    total_loss.backward()
    gen_grads = [p.grad.detach().clone() for p in generator.parameters() if p.grad is not None]
    gen_params_pre = [p.detach().clone() for p in generator.parameters()]
    opt_g.step()
    gen_params_post = [p.detach().clone() for p in generator.parameters()]

    # Step 5: Update Verifier critic
    verifier.train()
    in1_snn_v = normalize(fakes_1.detach().expand(-1, 3, -1, -1))
    in2_snn_v = normalize(inputs2.expand(-1, 3, -1, -1))
    ver_logits_v = verifier(in1_snn_v, in2_snn_v).squeeze()
    loss_ver = crit_ver(ver_logits_v, labels_id.type_as(ver_logits_v))
    opt_ver.zero_grad()
    loss_ver.backward()
    ver_grads = [p.grad.detach().clone() for p in verifier.parameters() if p.grad is not None]
    ver_params_pre = [p.detach().clone() for p in verifier.parameters()]
    opt_ver.step()
    ver_params_post = [p.detach().clone() for p in verifier.parameters()]
    verifier.eval()

    # Step 6: Update AC critic
    ac_model.train()
    in_ac_c = normalize(resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))
    ac_probs_c = ac_model(in_ac_c)
    loss_ac = crit_ac(ac_probs_c, labels)
    opt_ac.zero_grad()
    loss_ac.backward()
    ac_grads = [p.grad.detach().clone() for p in ac_model.parameters() if p.requires_grad and p.grad is not None]
    ac_params_pre = [p.detach().clone() for p in ac_model.parameters() if p.requires_grad]
    opt_ac.step()
    ac_params_post = [p.detach().clone() for p in ac_model.parameters() if p.requires_grad]
    ac_model.eval()

    return {
        'fakes_1': fakes_1.detach(),
        'ac_bce': ac_bce.item(),
        'privacy_term': privacy_term.item(),
        'total_loss': total_loss.item(),
        'gen_grads': gen_grads,
        'gen_params_pre': gen_params_pre,
        'gen_params_post': gen_params_post,
        'ver_grads': ver_grads,
        'ver_params_pre': ver_params_pre,
        'ver_params_post': ver_params_post,
        'ac_grads': ac_grads,
        'ac_params_pre': ac_params_pre,
        'ac_params_post': ac_params_post,
        'loss_ver': loss_ver.item(),
        'loss_ac': loss_ac.item(),
    }
