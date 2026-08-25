import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class FeatureConsistencyLoss(nn.Module):
    """Feature-space consistency loss between anonymized and original images.

    DISCLOSURE (P0.2.2 audit, F1): unlike the CERTIFIED C4 feature loss
    (research_agent/m0_port/ACLoss.py), which deep-copies the AC classifier
    EVERY forward so its feature target CO-EVOLVES with the moving critic,
    this module snapshots a STATIONARY teacher once at construction time.
    This is an intentional methodological difference: any paper comparison
    against C4 must state it.

    Computes MSE between penultimate-layer features of a frozen DenseNet-121
    diagnostic backbone for the deformed image and the original image:

        L_feat = || f(I_anon) - f(I_orig) ||_2^2 / dim(f)

    Design guarantees:
      * The feature trunk is built ONCE in __init__ (a single deepcopy of the
        classifier with its final linear layer removed). Unlike utils.ACLoss,
        no per-forward deepcopy occurs, so this adds no allocation churn.
      * All trunk parameters have requires_grad=False. Gradients still reach
        the input images (the generator), but never update the backbone.
        Note: even if gradients were deposited into the backbone during the
        generator step, AgentV2 calls optimizer_ac.zero_grad() before the
        critic's own backward, mirroring the baseline update order, so no
        leakage into critic updates is possible.
      * Preprocessing mirrors utils.ACLoss.forward exactly: 1->3 channel
        expansion, Resize to 224, ImageNet normalization -- so both losses see
        identically preprocessed tensors.

    :param ac_model: nn.Module
        The pre-trained DenseNet-121 classifier (same object as used by
        ACLoss; it is only read, never written).
    """

    def __init__(self, ac_model):
        super().__init__()

        # Build the truncated feature extractor once.
        #
        # NOTE on what "features" means here: the released CheXNet checkpoint
        # wraps its classifier as Sequential(Linear(1024 -> 14), Sigmoid())
        # (see chexnet/model.py). Cutting only the Sigmoid would leave the
        # 14-dim class logits -- far too narrow a signal to constrain the
        # representation. Replacing the WHOLE classifier with Identity yields
        # the true 1024-dim pooled penultimate representation.
        trunk = copy.deepcopy(ac_model)
        if not hasattr(trunk, 'classifier'):
            raise ValueError('Expected a model with a `.classifier` attribute '
                             '(torchvision DenseNet), got: %s' % type(trunk))
        feat_dim = None
        try:
            feat_dim = trunk.classifier[0].in_features
        except Exception:
            pass
        if feat_dim is None:
            raise ValueError('Could not infer feature dimension from '
                             '`classifier[0].in_features`; refusing to guess.')
        trunk.classifier = torch.nn.Identity()
        trunk.eval()
        for p in trunk.parameters():
            p.requires_grad_(False)
        self.trunk = trunk
        self.feat_dim = int(feat_dim)

        # Same preprocessing as utils.ACLoss (inputs are square, so an int
        # Resize(224) and a tuple Resize((224, 224)) coincide; we use the
        # explicit tuple form for clarity).
        self.resize = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def _features(self, image):
        """Preprocess a 1-channel image batch and extract features."""
        if image.dim() != 4 or image.shape[1] != 1:
            raise ValueError('Expected shape (N, 1, H, W); got %s' % (tuple(image.shape),))
        x = image.expand(-1, 3, -1, -1)
        x = self.normalize(self.resize(x))
        feats = self.trunk(x)
        return feats

    def forward(self, deformed_image, original_image):
        """Compute the normalized MSE between feature representations.

        :param deformed_image: torch.Tensor (N, 1, H, W)
            The anonymized/deformed image.
        :param original_image: torch.Tensor (N, 1, H, W)
            The undeformed source image.
        :return torch.Tensor
            Scalar loss.
        """

        f_anon = self._features(deformed_image)
        f_orig = self._features(original_image)

        # Plain elementwise MSE over (N, C): identical weighting for every
        # sample and every feature dimension. No rescaling tricks -- the loss
        # weight lives in the config (feature_loss_weight), not here.
        return torch.mean((f_anon - f_orig) ** 2)
