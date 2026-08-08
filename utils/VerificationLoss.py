import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms


class VerificationLoss(nn.Module):
    def __init__(self, verification_models, reduction: str = 'mean'):
        """The verification loss is intended to be used for learning a flow field that targetedly deforms a chest
        radiograph and thereby obfuscates the underlying biometric information.

        An ensemble of verification models is supported so that the generator cannot overfit a single adversary. At
        evaluation time the attacker is a freshly initialized SNN trained from scratch on anonymized images, so a
        single, continuously fine-tuned adversary during training under-estimates the real threat.

        :param verification_models: list of nn.Module
            The verification models (Siamese Neural Networks) forming the adversary ensemble.
        :param reduction: str
            Loss reduction method: default value is 'mean'; other options are 'sum' or 'none'.
        """

        super().__init__()
        self.verification_models = nn.ModuleList(verification_models)

        for model in self.verification_models:
            # Set model to evaluation mode
            model.eval()

            # Turn on gradient computation
            for param in model.parameters():
                param.requires_grad = True

        # Number of remaining warm-up iterations per model. A model that has just been re-initialized produces
        # meaningless scores, so it is excluded from the generator loss until it has been trained for a while.
        self.warmup_remaining = [0] * len(self.verification_models)

        self.reduction = reduction

    def active_indices(self):
        """Indices of the models that currently contribute to the generator loss."""

        return [i for i, w in enumerate(self.warmup_remaining) if w == 0]

    def sample_indices(self, n):
        """Randomly pick n of the currently active models for the generator loss.

        Back-propagating through the whole ensemble at once does not fit into 16 GB of VRAM, so each iteration the
        generator is confronted with a random subset. Every model is still trained every iteration.
        """

        active = self.active_indices()
        if not active:
            active = list(range(len(self.verification_models)))
        if n >= len(active):
            return active
        return random.sample(active, n)

    def decrement_warmup(self):
        """Advance the warm-up counters. Called once per training iteration."""

        self.warmup_remaining = [max(0, w - 1) for w in self.warmup_remaining]

    def forward(self, output1, output2, indices=None):
        """Compute the per-model similarity scores.

        :param indices: list of int or None
            The models to evaluate. Defaults to the currently active (warmed-up) models.
        :return verification_loss: torch.Tensor
            Shape [len(indices), batch_size] when reduction='none', a scalar otherwise.
        """

        if indices is None:
            indices = self.active_indices()
        if not indices:
            # Never leave the generator without an adversary (can only happen if every model is warming up)
            indices = list(range(len(self.verification_models)))

        # The verification model was trained with 3-channel inputs --> expand tensors to have 3 identical channels
        output1 = output1.expand(-1, 3, -1, -1)
        output2 = output2.expand(-1, 3, -1, -1)

        # Apply the ImageNet transform (since the verification model was trained with the ImageNet transform as well)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        output1 = normalize(output1)
        output2 = normalize(output2)

        # Compute the SNN output followed by a sigmoid activation function, for every model of the ensemble
        scores = [torch.sigmoid(self.verification_models[i](output1, output2).to(dtype=torch.float64)).squeeze(-1)
                  for i in indices]
        verification_loss = torch.stack(scores, dim=0)

        return self._reduce(verification_loss)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
