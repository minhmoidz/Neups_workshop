"""Phase-II IBR S1 - minimal graph modules.

Implements exactly the STEP 6A lock specification:

    (z_id, z_med) = E(x)
    x_self = G(z_id, z_med)
    x_anon = G(z_id_donor, z_med)

    z_id   : 128-d identity vector (identity-relevant variation)
    z_med  : spatial bottleneck map 16x16x512 (content / anatomy channel)

Critical invariant: L_rec acts ONLY on x_self; there is NO direct pixel
reconstruction loss ||x_anon - x|| in the S1 loss construction.
"""

import torch
import torch.nn as nn
from collections import OrderedDict


def _conv_block(in_ch, out_ch, name):
    return nn.Sequential(OrderedDict([
        (name + "_conv1", nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)),
        (name + "_norm1", nn.BatchNorm2d(out_ch)),
        (name + "_relu1", nn.ReLU(inplace=True)),
        (name + "_conv2", nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)),
        (name + "_norm2", nn.BatchNorm2d(out_ch)),
        (name + "_relu2", nn.ReLU(inplace=True)),
    ]))


class IBREncoder(nn.Module):
    """Shared encoder trunk (U-Net brain-segmentation skeleton, init_features=32).

    Input  : (B, 1, 256, 256)
    Outputs:
        z_id  : (B, 128)
        z_med : (B, 512, 16, 16)  spatial bottleneck map
        skips : list of encoder feature maps for the decoder skip connections
    """

    def __init__(self, in_channels=1, init_features=32, z_id_dim=128):
        super().__init__()
        f = init_features
        self.z_id_dim = z_id_dim
        self.enc1 = _conv_block(in_channels, f, "enc1")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = _conv_block(f, f * 2, "enc2")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = _conv_block(f * 2, f * 4, "enc3")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = _conv_block(f * 4, f * 8, "enc4")
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = _conv_block(f * 8, f * 16, "bottleneck")

        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.fc_id = nn.Linear(f * 16, z_id_dim)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        z_med = self.bottleneck(self.pool4(e4))  # (B, 512, 16, 16)
        z_id = self.fc_id(self.pool_global(z_med).flatten(1))  # (B, 128)
        return z_id, z_med, [e1, e2, e3, e4]


class IBRDecoder(nn.Module):
    """U-Net decoder symmetric to the trunk, with skip connections.

    z_id is injected at the bottleneck (additively broadcast onto the decoder
    latent) so the SELF branch can reconstruct the source identity while the
    ANON branch uses the donor's z_id on the same z_med content.

    Output: (B, 1, 256, 256), tanh head (matches pipeline image range).
    """

    def __init__(self, init_features=32, z_id_dim=128, out_channels=1):
        super().__init__()
        f = init_features
        self.fc_up = nn.Linear(z_id_dim, f * 16)

        self.upconv4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = _conv_block(f * 16, f * 8, "dec4")
        self.upconv3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(f * 8, f * 4, "dec3")
        self.upconv2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(f * 4, f * 2, "dec2")
        self.upconv1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = _conv_block(f * 2, f, "dec1")
        self.conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, z_id, z_med, skips):
        e1, e2, e3, e4 = skips
        latent = z_med + self.fc_up(z_id)[:, :, None, None]

        d4 = self.upconv4(latent)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return torch.tanh(self.conv(d1))


class ZIdVerifier(nn.Module):
    """Pairwise z_id verification head V.

    Input : [z_id1; z_id2] (B, 2*z_id_dim)
    Output: logit (B, 1) -> sigmoid for BCE (probability of same patient).

    Mirrors the existing SiameseNetwork fc_end head style. Trained with BCE.
    """

    def __init__(self, z_id_dim=128, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_id_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_id1, z_id2):
        return self.net(torch.cat([z_id1, z_id2], dim=1))


class ZMedAdversary(nn.Module):
    """Small pairwise identity adversary H_med operating on z_med.

    Architecture: 1x1 conv stack -> GAP -> Linear -> 1 logit.
    Each z_med is compressed to a 128-d descriptor; concatenated pair is
    scored as same/different patient.

    Input : (B, 512, 16, 16) z_med maps (x2)
    Output: logit (B, 1)

    Trained normally (its own optimizer). The encoder receives the REVERSED
    gradient through GradientReversalLayer on the z_med path.
    """

    def __init__(self, z_med_channels=512, hidden=128):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(z_med_channels, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out = nn.Linear(hidden * 2, 1)

    def forward(self, z_med1, z_med2):
        d1 = self.reduce(z_med1).flatten(1)
        d2 = self.reduce(z_med2).flatten(1)
        return self.out(torch.cat([d1, d2], dim=1))