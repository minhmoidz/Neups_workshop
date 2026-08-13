"""Phase-II IBR S1 - frozen utility models.

Loads the EXACT frozen classifier and segmentation teacher required by the
project. Hard-fails on SHA mismatch. Both are set to eval mode with
requires_grad=False (no parameters receive gradients), while gradients must
still propagate THROUGH their forward computations to x_anon where required
by L_path / L_anat.

Classifier checkpoint   : networks/pretrained_classifier.pth
Segmentation checkpoint : archive/train_seg_unet/best.pth (SHA begins 2dfdcf9b...)
"""

import hashlib
import torch

CLASSIFIER_PATH = "networks/pretrained_classifier.pth"
CLASSIFIER_SHA256 = "8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663"

SEGMENTATION_PATH = "archive/train_seg_unet/best.pth"
SEGMENTATION_SHA256_PREFIX = "2dfdcf9b"

LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
          'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_frozen_classifier(device='cuda'):
    """Load DenseNet-121 14-label classifier (full model object in checkpoint)."""
    actual = sha256(CLASSIFIER_PATH)
    assert actual == CLASSIFIER_SHA256, (
        "Classifier checkpoint SHA mismatch: got %s... expected %s..." % (actual[:16], CLASSIFIER_SHA256[:16]))
    chk = torch.load(CLASSIFIER_PATH, weights_only=False, map_location='cpu')
    model = chk['model']
    n_params = sum(p.numel() for p in model.parameters())
    _freeze(model)
    return model, {'path': CLASSIFIER_PATH, 'sha256': actual, 'architecture': type(model).__name__,
                   'params': n_params}


def load_frozen_segmentation_teacher(device='cuda'):
    """Load UNetSeg(1,3,16) segmentation teacher; SHA must begin 2dfdcf9b."""
    actual = sha256(SEGMENTATION_PATH)
    assert actual.startswith(SEGMENTATION_SHA256_PREFIX), (
        "Segmentation checkpoint SHA mismatch: got %s... expected prefix %s..." % (
            actual[:16], SEGMENTATION_SHA256_PREFIX))
    from networks.UNetSeg import UNetSeg
    chk = torch.load(SEGMENTATION_PATH, weights_only=False, map_location='cpu')
    model = UNetSeg(in_channels=1, out_channels=3,
                    init_features=32 if 'init_features' not in chk else chk['init_features'])
    model.load_state_dict(chk['model'], strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    _freeze(model)
    return model, {'path': SEGMENTATION_PATH, 'sha256': actual,
                   'architecture': 'UNetSeg(in=1, out=3, init_features=%d)' % chk['init_features'],
                   'params': n_params, 'epoch': chk.get('epoch'), 'mean_dice': chk.get('mean_dice')}