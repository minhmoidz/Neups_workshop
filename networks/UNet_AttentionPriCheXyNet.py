'''
MIT License

Copyright (c) 2019 mateuszbuda
Copyright (c) 2026 PriCheXy-Net V2 extensions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


import torch
import torch.nn as nn
from collections import OrderedDict

from networks.UNet_PriCheXyNet import UNet


class AttentionGate(nn.Module):
    """Additive attention gate (Oketa et al.-style, simplified).

    Deliberate deviations from the standard Attention U-Net gate head:
      * No BatchNorm inside the gate head. BatchNorm would re-center the
        pre-sigmoid activations and destroy any constant-bias initialization,
        making a near-identity initialization impossible. Removing BN keeps
        the gate deterministic at init.
      * The psi bias is set to GATE_INIT_BIAS so that sigmoid(bias) ~= 1, i.e.
        the gate starts out as (very nearly) the identity map on the skip
        connection and the network begins training as the pre-trained plain
        U-Net it was initialized from. This is essential for a controlled
        comparison against the baseline: epoch-0 behaviour matches the
        published starting point, and training can only deviate from it where
        gradients justify doing so.

    NEAR-IDENTITY MUST NOT BECOME A DEAD GRADIENT (fix 2026-08-28).
    An earlier revision zero-initialized W_g, W_x AND psi.weight together.
    That is a self-sustaining fixed point, not an initialization:

        a = relu(W_g(g) + W_x(x)) = relu(0) = 0
        dL/d(psi.weight) proportional to a        = 0  -> psi.weight stays 0
        dL/da            proportional to psi.weight = 0  -> W_g, W_x get NO gradient

    so every gate weight remains exactly zero forever and the gate collapses
    to the single learnable scalar sigmoid(psi.bias). This was confirmed on a
    completed 250-epoch run: all W_g/W_x/psi.weight tensors were still exactly
    0.0 and only the four psi.bias scalars had moved. The attention mechanism
    was therefore never active.

    The fix keeps W_g/W_x at their default (Kaiming-uniform) init so `a` is
    nonzero, and gives psi.weight a small nonzero init (PSI_INIT_WEIGHT_STD)
    so gradient flows back into W_g/W_x. The pre-sigmoid perturbation
    psi.weight . a stays orders of magnitude below GATE_INIT_BIAS, so the gate
    is still ~sigmoid(6) at init and the near-identity contract holds.

    :param F_l: int
        Number of channels of the skip-connection (encoder) feature map.
    :param F_g: int
        Number of channels of the gating signal (decoder/upconv feature map).
    :param F_int: int
        Number of internal bottleneck channels of the gate.
    """

    GATE_INIT_BIAS = 6.0  # sigmoid(6.0) ~= 0.9975 -> gate ~= identity at init
    # Small enough that psi.weight . a << GATE_INIT_BIAS at init (so the gate
    # stays near-identity), large enough that gradients reach W_g and W_x.
    PSI_INIT_WEIGHT_STD = 1e-3

    def __init__(self, F_l, F_g, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

        # Near-identity initialization WITH live gradients (see class docstring).
        # W_g and W_x keep nn.Conv2d's default Kaiming-uniform init so that
        # a = relu(W_g(g) + W_x(x)) is not identically zero.
        nn.init.normal_(self.psi.weight, mean=0.0, std=self.PSI_INIT_WEIGHT_STD)
        nn.init.constant_(self.psi.bias, self.GATE_INIT_BIAS)

    def forward(self, x_skip, g):
        """Apply the gate to a skip-connection feature map.

        :param x_skip: torch.Tensor
            Encoder skip-connection features, shape (N, F_l, H, W).
        :param g: torch.Tensor
            Decoder gating signal, shape (N, F_g, H, W) -- spatially aligned
            with x_skip (the caller must upsample it beforehand).
        :return tuple(torch.Tensor, torch.Tensor)
            The gated skip features (x_skip * gate) and the raw gate map in
            [0, 1], shape (N, 1, H, W). The raw map is returned for logging /
            analysis (hypothesis H1 evidence) at no extra compute cost.
        """
        a = self.relu(self.W_g(g) + self.W_x(x_skip))
        gate = torch.sigmoid(self.psi(a))
        return x_skip * gate, gate


class UNetAtt(UNet):
    """Attention-gated variant of the PriCheXy-Net flow field U-Net.

    Contract kept identical to networks.UNet_PriCheXyNet.UNet:
      * Constructor signature (in_channels, out_channels, init_features).
      * All encoder / decoder / bottleneck / final-conv parameter names are
        byte-for-byte identical to the plain UNet, so the released pre-trained
        generator checkpoint loads into every overlapping parameter.
      * Forward pass: input image -> 2-channel flow field bounded by tanh.

    Differences (all additive, none modify existing parameter tensors):
      * One AttentionGate per decoder level, applied to the corresponding
        skip connection. Gate parameters live under `att4` .. `att1`.
    """

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         init_features=init_features)

        f = init_features
        self.att4 = AttentionGate(F_l=f * 8, F_g=f * 8, F_int=f * 4)
        self.att3 = AttentionGate(F_l=f * 4, F_g=f * 4, F_int=f * 2)
        self.att2 = AttentionGate(F_l=f * 2, F_g=f * 2, F_int=f)
        self.att1 = AttentionGate(F_l=f,     F_g=f,     F_int=max(f // 2, 1))

    def forward(self, x, return_gates=False):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        # NOTE on variable naming and concat ORDER: each level concatenates
        # [upsampled_decoder_signal, attended_encoder_skip] -- exactly the
        # channel layout of the plain U-Net (which cats [upconv_out, enc]).
        # When every gate equals 1 the tensor fed to each decoder block is
        # bit-for-bit what the plain U-Net would produce, so pre-trained
        # decoder weights remain valid at initialization. The gating signal
        # and the gated skip are kept in DISTINCT variables; merging them
        # into one name silently corrupts the data flow.
        up4 = self.upconv4(bottleneck)
        skip4, g4 = self.att4(enc4, up4)
        dec4 = torch.cat((up4, skip4), dim=1)
        dec4 = self.decoder4(dec4)

        up3 = self.upconv3(dec4)
        skip3, g3 = self.att3(enc3, up3)
        dec3 = torch.cat((up3, skip3), dim=1)
        dec3 = self.decoder3(dec3)

        up2 = self.upconv2(dec3)
        skip2, g2 = self.att2(enc2, up2)
        dec2 = torch.cat((up2, skip2), dim=1)
        dec2 = self.decoder2(dec2)

        up1 = self.upconv1(dec2)
        skip1, g1 = self.att1(enc1, up1)
        dec1 = torch.cat((up1, skip1), dim=1)
        dec1 = self.decoder1(dec1)

        flow = torch.tanh(self.conv(dec1))
        if return_gates:
            return flow, {'g4': g4, 'g3': g3, 'g2': g2, 'g1': g1}
        return flow


def load_pretrained_into_unet_att(model, checkpoint_path, device='cpu'):
    """Load the released plain-U-Net generator checkpoint into a UNetAtt.

    Semantics are fail-closed:
      * Every key present in the checkpoint MUST exist in the model
        (unexpected_keys must be empty), otherwise ValueError.
      * Missing keys are tolerated ONLY if they belong to attention gates
        (`att1.` .. `att4.` prefixes), otherwise ValueError.
      * All overlapping tensors must match in shape, otherwise RuntimeError
        (raised by strict copy_ below).

    :param model: UNetAtt
        The target model (modified in place).
    :param checkpoint_path: str
        Path to `pretrained_generator_prichexy_net.pth`.
    :param device: str
        Device for torch.load.
    :return dict
        Summary {loaded, missing_gate_keys} for the provenance manifest.
    """

    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict):
        raise ValueError('Checkpoint is not a state dict: %s' % checkpoint_path)

    model_state = model.state_dict()

    unexpected = sorted(k for k in state if k not in model_state)
    if unexpected:
        raise ValueError('Checkpoint contains parameters unknown to UNetAtt '
                         '(refusing partial load): %s' % unexpected)

    missing_gate = []
    loaded = []
    with torch.no_grad():
        for k, v in state.items():
            if tuple(model_state[k].shape) != tuple(v.shape):
                raise RuntimeError('Shape mismatch for %s: ckpt %s vs model %s'
                                   % (k, tuple(v.shape), tuple(model_state[k].shape)))
            model_state[k].copy_(v)
            loaded.append(k)

    missing_gate = sorted(k for k in model_state
                          if k not in state and k.split('.')[0] in
                          {'att1', 'att2', 'att3', 'att4'})
    non_gate_missing = sorted(k for k in model_state
                              if k not in state and k.split('.')[0] not in
                              {'att1', 'att2', 'att3', 'att4'})
    if non_gate_missing:
        raise ValueError('Non-attention parameters missing after load '
                         '(refusing to continue): %s' % non_gate_missing)

    return {'num_loaded_tensors': len(loaded),
            'missing_attention_keys': missing_gate}
