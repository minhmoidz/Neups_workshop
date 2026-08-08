import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class ACLoss(nn.Module):
    def __init__(self, ac_model, reduction: str = 'mean', pos_weight=None, feature_loss_weight=0.0):
        """The auxiliary classifier loss which is intended to be used to ensure that underlying abnormality patterns are
        preserved during the anonymization process.

        :param ac_model: nn.Module
            A pre-trained abnormality classifier (DenseNet-121).
        :param reduction: str
            Loss reduction method: default value is 'mean'; other options are 'sum' or 'none'.
        :param pos_weight: float or None
            Weight applied to the positive term of the BCE. With only 7.5 % positive labels, the plain BCE is
            diluted roughly 12x by the negatives: a deformation that raises the positive-label BCE by 0.415
            while lowering the negative-label BCE by 0.016 barely moves the aggregate, so the generator can
            satisfy the critic while mean AUC drops by 3.5 points. None reproduces the original behaviour.
        :param feature_loss_weight: float
            Weight of an additional term that keeps the frozen classifier's penultimate features of the
            deformed image close to those of the real image. Unlike the label BCE this signal cannot be
            diluted by class imbalance. 0 disables it and reproduces the original behaviour.
        """

        super().__init__()
        self.ac_model = ac_model

        # Set model to evaluation mode
        self.ac_model.eval()

        # Turn on gradient computation
        for param in self.ac_model.parameters():
            param.requires_grad = True

        self.reduction = reduction
        self.feature_loss_weight = feature_loss_weight
        pw = None if pos_weight is None else torch.tensor(float(pos_weight)).cuda()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pw).cuda()

        # Build the loss model once; refresh() re-syncs it with the (ever-improving) ac_model
        self.refresh()

    def refresh(self):
        """Re-build the loss model from the current ac_model state so that the adversarial
        classifier tracks the auxiliary classifier as it keeps improving during training."""
        # Cut the last layer for the actual loss model
        self.loss_model = copy.deepcopy(self.ac_model)
        self.loss_model.classifier = nn.Sequential(*list(self.loss_model.classifier.children())[:-1])

    def _preprocess(self, image):
        # The abnormality classification model was trained with 3-channel inputs
        # --> expand tensors to have 3 identical channels
        image = image.expand(-1, 3, -1, -1)

        # Apply the ImageNet transform (since the classifier was trained with the ImageNet transform as well)
        resize = transforms.Resize(224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return normalize(resize(image))

    def _features(self, image):
        """Pooled penultimate features of the frozen classifier (same path as DenseNet's own forward)."""

        features = F.relu(self.loss_model.features(image))

        return torch.flatten(F.adaptive_avg_pool2d(features, (1, 1)), 1)

    def forward(self, deformed_image, target_labels, real_image=None):
        deformed_features = self._features(self._preprocess(deformed_image))

        # Compute the classification output
        ac_predictions = self.loss_model.classifier(deformed_features)
        loss = self.bce_loss(ac_predictions, target_labels)

        if self.feature_loss_weight > 0 and real_image is not None:
            real_features = self._features(self._preprocess(real_image)).detach()
            loss = loss + self.feature_loss_weight * F.mse_loss(deformed_features, real_features)

        return loss
