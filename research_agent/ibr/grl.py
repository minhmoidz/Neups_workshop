"""Phase-II IBR S1 - Gradient Reversal Layer (GRL).

Standard domain-adversarial gradient reversal (Ganin et al.). Forward is the
identity; backward multiplies the gradient by `-lambda`. Used to make the
encoder produce `z_med` representations from which the pairwise identity
adversary `H_med` cannot recover patient identity.

Do NOT call this mutual information. The defensible claim (:STEP 6A lock #4) is:
    "pairwise patient identity is adversarially suppressed in z_med."
"""

import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    """Applies gradient reversal to its input with scale `lambda`.

    Args:
        lambd: float, GRL gradient scale. Default 1.0 (frozen in STEP 6A lock).
    """

    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambd)