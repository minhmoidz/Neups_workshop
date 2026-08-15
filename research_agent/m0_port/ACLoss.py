"""REPAIRED ACLoss for the C4/C2 port (M0 P0 fix).

Upstream/restored-baseline semantics: copy.deepcopy(self.ac_model) is taken on
EVERY forward, so the generator-facing loss_model always reflects the CURRENT
auxiliary-classifier weights and — critically — the deepcopy DETACHES the
generator gradient path from ac_model's parameters (the two are separate tensors).

The historical main-branch implementation built loss_model ONCE in __init__ and
only refreshed it via a refresh() method that was NEVER called, so the generator
fought a classifier frozen at its init snapshot. That is a P0 semantic drift.

The repair below is minimal and provably equivalent to upstream: refresh() is
invoked at the start of forward(), i.e. the loss model is rebuilt from the current
ac_model exactly as upstream's per-call deepcopy does. The C4 extensions
(pos_weight, feature_loss_weight with detached source features) are preserved with
their documented default-equivalent behavior (None / 0.0 reproduce baseline).

Files changed by M0: none in the pristine restored baseline (its ACLoss is already
the correct upstream deepcopy-every-forward). This file is the port-ready module
the C4/C2 dev configs reference.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ACLoss(nn.Module):
    def __init__(self, ac_model, reduction: str = 'mean', pos_weight=None, feature_loss_weight=0.0):
        """Auxiliary classifier loss driving anonymization.

        :param ac_model: nn.Module — pretrained abnormality classifier (DenseNet-121).
        :param reduction: str — 'mean' | 'sum' | 'none'.
        :param pos_weight: float|None — BCE positive-term weight; None = baseline BCE.
        :param feature_loss_weight: float — weight of the penultimate-feature MSE vs the
            real image; 0.0 disables it (baseline). Source features are always detached.
        """
        super().__init__()
        self.ac_model = ac_model
        self.ac_model.eval()
        for param in self.ac_model.parameters():
            param.requires_grad = True

        self.reduction = reduction
        self.feature_loss_weight = feature_loss_weight
        pw = None if pos_weight is None else torch.tensor(float(pos_weight)).cuda()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pw).cuda()

        self.refresh()

    def refresh(self):
        """Re-sync loss_model with the CURRENT ac_model weights (== upstream per-forward deepcopy)."""
        self.loss_model = copy.deepcopy(self.ac_model)
        self.loss_model.classifier = nn.Sequential(*list(self.loss_model.classifier.children())[:-1])

    def _preprocess(self, image):
        image = image.expand(-1, 3, -1, -1)
        resize = transforms.Resize(224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalize(resize(image))

    def _features(self, image):
        features = F.relu(self.loss_model.features(image))
        return torch.flatten(F.adaptive_avg_pool2d(features, (1, 1)), 1)

    def forward(self, deformed_image, target_labels, real_image=None):
        # M0 P0 repair: refresh() at forward start guarantees the generator sees the
        # CURRENT classifier state (and a detached gradient path), exactly like the
        # pristine upstream `loss_model = copy.deepcopy(self.ac_model)` per call.
        self.refresh()

        deformed_features = self._features(self._preprocess(deformed_image))
        ac_predictions = self.loss_model.classifier(deformed_features)
        loss = self.bce_loss(ac_predictions, target_labels)

        if self.feature_loss_weight > 0 and real_image is not None:
            real_features = self._features(self._preprocess(real_image)).detach()
            loss = loss + self.feature_loss_weight * F.mse_loss(deformed_features, real_features)

        return loss