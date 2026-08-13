"""Phase-II IBR S1 - combined model (E, G, V, H_med) and forward graph.

Provides a single container for constructing every S1 module with default
hyperparameters frozen in the STEP 6A lock, and a helper that runs the full
S1 training graph for a batch.

Core graph:
    (z_id, z_med) = E(x)
    x_self = G(z_id, z_med)
    x_anon = G(z_id_donor, z_med)
    L_rec acts ONLY on x_self.
"""

import torch
import torch.nn as nn

from research_agent.ibr.models import IBREncoder, IBRDecoder, ZIdVerifier, ZMedAdversary
from research_agent.ibr.grl import GradientReversalLayer


class IBRModel(nn.Module):
    """Full S1 module bundle.

    Args:
        init_features (int): U-Net trunk init features (default 32, STEP 6A lock).
        z_id_dim (int): identity dimension (must stay 128; do not change).
    """

    def __init__(self, init_features=32, z_id_dim=128):
        super().__init__()
        self.z_id_dim = z_id_dim
        self.encoder = IBREncoder(in_channels=1, init_features=init_features, z_id_dim=z_id_dim)
        self.decoder = IBRDecoder(init_features=init_features, z_id_dim=z_id_dim, out_channels=1)
        self.verifier = ZIdVerifier(z_id_dim=z_id_dim, hidden=128)
        self.adv = ZMedAdversary(z_med_channels=512, hidden=128)
        self.grl = GradientReversalLayer(lambd=1.0)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z_id, z_med, skips):
        return self.decoder(z_id, z_med, skips)

    def forward(self, x, x_donor):
        """Run the SELF and ANON branches for a source batch.

        Args:
            x: source images (B,1,256,256) in [-1,1]
            x_donor: donor images (B,1,256,256); donor identity injected via z_id

        Returns dict with z_id (source), z_med (source), skips, x_self, x_anon,
        and z_id_donor. The reconstruction is applied ONLY to x_self; x_anon is
        constrained only by task/anatomy/privacy terms in the loss assembly.
        """
        # self branch
        z_id, z_med, skips = self.encode(x)
        x_self = self.decode(z_id, z_med, skips)

        # anon branch: donor z_id, source z_med (donor identity independent of source)
        z_id_donor, z_med_donor, _ = self.encode(x_donor)
        x_anon = self.decode(z_id_donor, z_med, skips)

        return {
            'z_id': z_id,
            'z_med': z_med,
            'z_med_donor': z_med_donor,
            'skips': skips,
            'x_self': x_self,
            'x_anon': x_anon,
            'z_id_donor': z_id_donor,
        }

    def verify(self, z_id1, z_id2):
        return self.verifier(z_id1, z_id2)

    def adversary_logits(self, z_med1, z_med2):
        """Adversary on z_med pair; z_med passed through GRL so E gets reversed grad."""
        return self.adv(self.grl(z_med1), self.grl(z_med2))


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())