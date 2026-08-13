"""Phase-II IBR S1 - loss graph and utilities.

Implements exactly:

    L_S1 = lam_rec * L_rec + lam_path * L_path + lam_anat * L_anat
         + lam_zid * L_zid_pair + lam_adv * L_zmed_adv

Critical invariant: L_rec acts ONLY on x_self. There is NO direct pixel
reconstruction loss ||x_anon - x||.

Image domain: [-1, 1] (segmentation-dataset convention).
    - Decoder outputs tanh ([-1,1]).
    - Frozen classifier input: ((x+1)/2) -> expand 3ch -> Resize 224 -> ImageNet norm.
    - Frozen segmentation teacher input: x directly in [-1,1].

Coefficients frozen in STEP 6A lock: all = 1.0 (lambda_rec anchor 1.0).
GRL lambda = 1.0.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms

from research_agent.ibr.frozen_models import load_frozen_classifier, load_frozen_segmentation_teacher

# Frozen in STEP 6A lock #6 (loss-scale normalization based; do not tune).
LAMBDA_REC = 1.0
LAMBDA_PATH = 1.0
LAMBDA_ANAT = 1.0
LAMBDA_ZID = 1.0
LAMBDA_ADV = 1.0
GRL_LAMBDA = 1.0

_CLS_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_CLS_RESIZE = transforms.Resize((224, 224))


class FrozenUtility:
    """Loads the frozen classifier + segmentation teacher once; exposes forward calls.

    Both are eval / requires_grad=False. Gradients must still flow THROUGH their
    forward computations to x_anon for L_path / L_anat. We therefore never wrap
    the relevant forward in torch.no_grad().
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.classifier, self.classifier_meta = load_frozen_classifier(device)
        self.segmenter, self.seg_meta = load_frozen_segmentation_teacher(device)
        self.classifier = self.classifier.to(device)
        self.segmenter = self.segmenter.to(device)

    def path_logits(self, x_anon):
        """Map anon images ([-1,1]) to classifier logits (pre-sigmoid is inside model?).

        The frozen DenseNet checkpoint's classifier head ends with Sigmoid, so
        `model(x)` returns probabilities in [0,1]. We treat those as the
        predictions and compute BCE directly against labels.
        """
        im = (x_anon + 1.0) / 2.0                      # [-1,1] -> [0,1]
        im = im.expand(-1, 3, -1, -1)
        im = _CLS_RESIZE(im)
        im = _CLS_NORM(im)
        return self.classifier(im)                     # (B, 14) probabilities

    def anat_maps(self, x):
        """Frozen segmentation teacher output (sigmoid probability maps) for x in [-1,1]."""
        return self.segmenter(x)                        # (B, 3, 256, 256)


def classification_loss(path_prob, y_path):
    """L_path: BCE between classifier probabilities on x_anon and source labels.

    y_path: (B, 14) in {0,1}. Returns scalar BCE (mean over batch and labels).
    """
    return F.binary_cross_entropy(path_prob, y_path)


def anatomy_loss(anat_anon, anat_source):
    """L_anat: MSE/BCE between teacher(anon) and teacher(source) probability maps.

    Uses MSE on the 3-structure probability maps so the anon anatomy tracks the
    source anatomy. Equivalent-scale to the other BCE terms.
    """
    return F.mse_loss(anat_anon, anat_source)


def reconstruction_loss(x_self, x):
    """L_rec = || x_self - x ||_1 — ONLY on x_self."""
    return F.l1_loss(x_self, x)


def zid_pair_loss(verifier_logits, y_pair):
    """L_zid_pair: BCE verification on z_id. y_pair=1 same, 0 different."""
    return F.binary_cross_entropy_with_logits(verifier_logits, y_pair)


def zmed_adv_loss(adv_logits, y_pair):
    """L_zmed_adv: BCE for the adversary on z_med (post-GRL input).

    The gradient-reversal layer is applied to z_med BEFORE this call, so the
    encoder sees a reversed/confusion gradient while H_med itself trains normally.
    """
    return F.binary_cross_entropy_with_logits(adv_logits, y_pair)


def make_pair_labels(same_mask):
    """Given a bool tensor of same-patient flags, return (B,1) float labels."""
    return same_mask.float().unsqueeze(1)