"""STEP 7A — patient-verifier attacker over the proposed SOURCE CONDITION.

Three arms, one experiment package. Inputs are ONLY the condition (anatomy maps
and/or pathology labels). NO image pixels, NO source appearance latent.

    ARM A — anatomy only: 3 soft maps -> e_map (ResNet-18, 3-ch) -> |e1-e2| -> logit
    ARM B — pathology only: 14 labels -> e_path (small MLP) -> |e1-e2| -> logit
    ARM C — joint: concat(e_map, e_path) -> |e1-e2| -> logit   (PRIMARY)

All pairwise comparisons are ORDER-INVARIANT: |e1 - e2| (never ordered concat).
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

Z = 128          # embedding dim
PATH_DIM = 14


class AnatomyEncoder(nn.Module):
    """ResNet-18 modified for 3-channel anatomy maps -> 128-d embedding."""

    def __init__(self, z=Z):
        super().__init__()
        self.backbone = tvm.resnet18(weights=None)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()          # keep 512-d pooled features
        self.proj = nn.Sequential(nn.Linear(in_f, z), nn.ReLU(inplace=True))
        self.z = z

    def forward(self, maps):
        return self.proj(self.backbone(maps))


class PathologyEncoder(nn.Module):
    """Small MLP: 14 pathology labels -> 128-d embedding."""

    def __init__(self, path_dim=PATH_DIM, z=Z):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(path_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, z), nn.ReLU(inplace=True))
        self.z = z

    def forward(self, y_path):
        return self.mlp(y_path)


class VerificationHead(nn.Module):
    """MLP on |e1 - e2| -> single verification logit."""

    def __init__(self, in_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1))

    def forward(self, diff):
        return self.mlp(diff)


class PairVerifier(nn.Module):
    """Generic order-invariant pairwise verifier: logit = MLP(|e1-e2|)."""

    def __init__(self, encoder, emb_dim):
        super().__init__()
        self.encoder = encoder
        self.head = VerificationHead(emb_dim)

    def pair_logit(self, e1, e2):
        return self.head(torch.abs(e1 - e2))


class ArmAModel(nn.Module):
    """ARM A — anatomy only."""

    def __init__(self, z=Z):
        super().__init__()
        self.encoder = AnatomyEncoder(z)
        self.head = VerificationHead(z)

    def forward(self, m1, m2, y1=None, y2=None):
        e1 = self.encoder(m1)
        e2 = self.encoder(m2)
        return self.head(torch.abs(e1 - e2))


class ArmBModel(nn.Module):
    """ARM B — pathology only."""

    def __init__(self, path_dim=PATH_DIM, z=Z):
        super().__init__()
        self.encoder = PathologyEncoder(path_dim, z)
        self.head = VerificationHead(z)

    def forward(self, m1=None, m2=None, y1=None, y2=None):
        e1 = self.encoder(y1)
        e2 = self.encoder(y2)
        return self.head(torch.abs(e1 - e2))


class ArmCModel(nn.Module):
    """ARM C — JOINT (PRIMARY): anatomy maps + pathology labels."""

    def __init__(self, path_dim=PATH_DIM, z=Z):
        super().__init__()
        self.map_encoder = AnatomyEncoder(z)
        self.path_encoder = PathologyEncoder(path_dim, z)
        self.head = VerificationHead(2 * z)

    def joint_embed(self, maps, y_path):
        e_map = self.map_encoder(maps)
        e_path = self.path_encoder(y_path)
        return torch.cat([e_map, e_path], dim=1)

    def forward(self, m1, m2, y1, y2):
        e1 = self.joint_embed(m1, y1)
        e2 = self.joint_embed(m2, y2)
        return self.head(torch.abs(e1 - e2))


def build_model(arm, path_dim=PATH_DIM, z=Z):
    if arm == 'A':
        return ArmAModel(z)
    if arm == 'B':
        return ArmBModel(path_dim, z)
    if arm == 'C':
        return ArmCModel(path_dim, z)
    raise ValueError(arm)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)