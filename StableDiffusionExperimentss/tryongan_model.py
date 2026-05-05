"""
tryongan_model.py — GigaGAN-based Generator + Multi-Scale Discriminator
                     for conditional virtual try-on.

Architecture overview  (March 2026 — SOTA GAN design)
──────────────────────────────────────────────────────
Based on GigaGAN (Yu et al., "Scaling up GANs for Text-to-Image Synthesis",
CVPR 2023) with adaptations for pixel-space conditional generation.

Key innovations over StyleGAN2:
    1. **Adaptive kernel selection** — style vector picks and blends K basis
       kernels via softmax attention (sample-adaptive kernels).
    2. **Self-attention** at 32×32 resolution in both G and D.
    3. **Multi-scale discriminator** — 3 separate sub-discriminators at
       native, ½, and ¼ resolutions for richer feedback.
    4. **Hinge loss + R1 gradient penalty** (standard SOTA recipe).

Conditioning: **channel-wise concatenation**.
    gen_input = cat([person, cloth], dim=channel)  →  [B, 6, 512, 512]
    gen_output = Generator(gen_input)              →  [B, 3, 512, 512]

Generator:
    ConditionEncoder  →  per-layer style vectors + per-level feature maps (skip)
    U-Net Synthesis   →  adaptive modulated conv decoder (4×4 → 512×512)
                         with encoder skip connections + self-attention at 32×32.

Discriminator:
    Multi-Scale D:  cat([image, person, cloth], dim=channel) at {512, 256, 128}
    Each scale:     residual downsampling + self-attention → scalar score.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


# ============================================================
# CORE BUILDING BLOCKS
# ============================================================

class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate (Karras et al., 2018)."""

    def __init__(self, in_dim, out_dim, lr_mul=1.0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = (1.0 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        b = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, b)


class AdaptiveModulatedConv2d(nn.Module):
    """GigaGAN-style adaptive kernel selection + style modulation.

    Instead of a single weight tensor, this layer has K basis kernels.
    The style vector drives both:
        (a) softmax attention over the K kernels (kernel selection), and
        (b) per-input-channel modulation + demodulation (StyleGAN2-style).
    """

    def __init__(self, in_ch, out_ch, kernel_size, style_dim,
                 n_kernels=4, demodulate=True, upsample=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.demodulate = demodulate
        self.upsample = upsample
        self.padding = kernel_size // 2

        # K basis kernels: [K, 1, out_ch, in_ch, k, k]
        self.weight = nn.Parameter(
            torch.randn(n_kernels, 1, out_ch, in_ch, kernel_size, kernel_size))
        self.scale = 1.0 / math.sqrt(in_ch * kernel_size ** 2)

        # Style → kernel selection logits  (sample-adaptive kernel attention)
        self.kernel_attn = EqualLinear(style_dim, n_kernels)
        # Style → per-input-channel modulation
        self.modulation = EqualLinear(style_dim, in_ch)

    def forward(self, x, style):
        B, C, H, W = x.shape

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                              align_corners=False)
            H, W = H * 2, W * 2

        # 1) Adaptive kernel blending
        #    attn: [B, K, 1, 1, 1, 1]
        attn = F.softmax(self.kernel_attn(style), dim=1)
        attn = attn.view(B, self.n_kernels, 1, 1, 1, 1)
        # Blend K kernels → per-sample kernel  [B, out, in, k, k]
        w = (self.weight * attn).sum(0)  # broadcast [K,1,...] * [B,K,...] → sum over K

        # 2) Modulation
        s = self.modulation(style).view(B, 1, C, 1, 1)
        w = w * self.scale * s

        # 3) Demodulation
        if self.demodulate:
            denom = torch.rsqrt(w.pow(2).sum([2, 3, 4], keepdim=True) + 1e-8)
            w = w * denom

        # 4) Grouped convolution
        x = x.reshape(1, B * C, H, W)
        w = w.reshape(B * self.out_ch, C, self.kernel_size, self.kernel_size)
        out = F.conv2d(x, w, padding=self.padding, groups=B)
        return out.reshape(B, self.out_ch, H, W)


class NoiseInjection(nn.Module):
    """Per-pixel learned-scale noise injection."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3],
                                device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class SelfAttention(nn.Module):
    """Self-attention layer (BigGAN / GigaGAN style)."""

    def __init__(self, ch):
        super().__init__()
        mid = max(ch // 8, 1)
        self.query = nn.Conv2d(ch, mid, 1)
        self.key = nn.Conv2d(ch, mid, 1)
        self.value = nn.Conv2d(ch, ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, mid]
        k = self.key(x).view(B, -1, H * W)                       # [B, mid, HW]
        attn = F.softmax(torch.bmm(q, k) / math.sqrt(q.shape[-1]), dim=-1)
        v = self.value(x).view(B, C, H * W)                      # [B, C, HW]
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


class StyledConv(nn.Module):
    """Styled convolution: AdaptiveModConv → Noise → LeakyReLU."""

    def __init__(self, in_ch, out_ch, kernel_size, style_dim,
                 n_kernels=4, upsample=False):
        super().__init__()
        self.conv = AdaptiveModulatedConv2d(
            in_ch, out_ch, kernel_size, style_dim,
            n_kernels=n_kernels, upsample=upsample)
        self.noise = NoiseInjection()
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style, noise=None):
        x = self.conv(x, style)
        x = self.noise(x, noise)
        return self.act(x)


class ToRGB(nn.Module):
    """1×1 adaptive-modulated conv to RGB."""

    def __init__(self, in_ch, style_dim):
        super().__init__()
        self.conv = AdaptiveModulatedConv2d(
            in_ch, 3, 1, style_dim, n_kernels=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style):
        return self.conv(x, style) + self.bias


# ============================================================
# CONDITION ENCODER (with self-attention)
# ============================================================

class ConditionEncoder(nn.Module):
    """Encode channel-concatenated condition into per-layer style vectors
    and per-level feature maps for U-Net skip connections.

    Input:  [B, in_ch, 512, 512]
    Output: (styles, features)
        styles   — list of [B, style_dim] vectors (one per encoder level)
        features — list of feature maps at each resolution for skip connections
    """

    # 512→256→128→64→32→16→8→4
    CHANNELS = (64, 128, 256, 512, 512, 512, 512, 512)
    ATTN_RES = 32  # self-attention at feature maps of this spatial size

    def __init__(self, in_ch=6, style_dim=512):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_ch, self.CHANNELS[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.blocks = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.style_projections = nn.ModuleList()

        # First style from initial features
        self.style_projections.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            EqualLinear(self.CHANNELS[0], style_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ))

        prev_ch = self.CHANNELS[0]
        spatial = 512
        for ch in self.CHANNELS[1:]:
            spatial //= 2
            self.blocks.append(nn.Sequential(
                nn.Conv2d(prev_ch, ch, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ch, ch, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            # Self-attention at the matching resolution
            self.attns.append(
                SelfAttention(ch) if spatial == self.ATTN_RES else nn.Identity())
            self.style_projections.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                EqualLinear(ch, style_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            prev_ch = ch

    def forward(self, x):
        feat = self.initial(x)
        styles = [self.style_projections[0](feat)]
        features = [feat]  # for skip connections

        for block, attn, style_proj in zip(
                self.blocks, self.attns, self.style_projections[1:]):
            feat = block(feat)
            feat = attn(feat)
            styles.append(style_proj(feat))
            features.append(feat)

        return styles, features   # features[-1] is [B, 512, 4, 4]


# ============================================================
# GIGAGAN GENERATOR  (U-Net + adaptive modulated synthesis)
# ============================================================

class GigaGANTryOnGenerator(nn.Module):
    """GigaGAN-inspired conditional generator for virtual try-on.

    Input:  cat([person, cloth], dim=1) → [B, 6, 512, 512]   (or [B,3,...] in OOTD)
    Output: predicted try-on            → [B, 3, 512, 512]

    Pipeline:
        1. ConditionEncoder → per-layer style vectors + per-level features
        2. Synthesis with adaptive modulated convolutions (4×4 → 512×512)
        3. U-Net skip connections from encoder to decoder
        4. Self-attention at 32×32
        5. Progressive ToRGB with residual summation
    """

    # Decoder channel schedule: 4×4 → 8 → 16 → 32 → 64 → 128 → 256 → 512
    SYN_CHANNELS = (512, 512, 512, 512, 256, 128, 64, 32)

    def __init__(self, in_channels=6, style_dim=512, n_kernels=4):
        super().__init__()
        self.style_dim = style_dim

        # Condition encoder (produces styles + feature hierarchy for skips)
        self.encoder = ConditionEncoder(in_ch=in_channels, style_dim=style_dim)

        # Seed projection: deepest encoder feature → synthesis input
        self.seed_proj = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Skip connection proj: adapt encoder features to decoder channel counts
        # Encoder features (from shallowest to deepest):
        #   [64@512, 128@256, 256@128, 512@64, 512@32, 512@16, 512@8, 512@4]
        # Decoder at each level _before_ skip would have SYN_CHANNELS[i] channels.
        # After skip: SYN_CHANNELS[i] + proj(encoder_feat) → needs 1×1 adaptation.
        enc_chs = list(ConditionEncoder.CHANNELS)   # [64,128,256,512,512,512,512,512]
        dec_chs = list(self.SYN_CHANNELS)            # [512,512,512,512,256,128,64,32]
        # Match decoder level i to encoder level (N-1-i) — mirror order
        self.skip_projs = nn.ModuleList()
        for i, dch in enumerate(dec_chs):
            enc_idx = len(enc_chs) - 1 - i   # deepest enc ↔ first dec level
            ech = enc_chs[enc_idx]
            # 1×1 conv to adapt encoder channels to decoder channels for concat
            self.skip_projs.append(nn.Conv2d(ech, dch, 1))

        # Synthesis blocks with adaptive modulated convolutions
        self.styled_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.dec_attns = nn.ModuleList()

        prev_ch = 512
        spatial = 4
        for i, ch in enumerate(self.SYN_CHANNELS):
            upsample = (i > 0)
            if upsample:
                spatial *= 2
            # First conv (possibly upsampling), takes concat input (+skip)
            in_ch_first = (prev_ch + ch) if i > 0 else prev_ch  # skip concat
            self.styled_convs.append(
                StyledConv(in_ch_first, ch, 3, style_dim,
                           n_kernels=n_kernels, upsample=upsample))
            self.styled_convs.append(
                StyledConv(ch, ch, 3, style_dim, n_kernels=n_kernels))
            self.to_rgbs.append(ToRGB(ch, style_dim))
            # Self-attention at 32×32
            self.dec_attns.append(
                SelfAttention(ch) if spatial == 32 else nn.Identity())
            prev_ch = ch

        # 3 style slots per block × 8 blocks = 24 total
        self.n_styles = len(self.SYN_CHANNELS) * 3

    def forward(self, x):
        """x: [B, 6, 512, 512] → [B, 3, 512, 512] in [-1, 1]."""
        styles_raw, enc_features = self.encoder(x)

        # Cycle styles to fill all synthesis slots
        styles = [styles_raw[i % len(styles_raw)] for i in range(self.n_styles)]

        # Seed from deepest encoder feature
        feat = self.seed_proj(enc_features[-1])  # [B, 512, 4, 4]

        rgb = None
        si = 0
        for i in range(len(self.SYN_CHANNELS)):
            # U-Net skip connection (mirror: deepest encoder → first decoder level)
            enc_idx = len(enc_features) - 1 - i
            skip = self.skip_projs[i](enc_features[enc_idx])

            if i > 0:
                # Concat skip at decoder resolution before first styled conv
                skip = F.interpolate(skip, size=feat.shape[2:], mode='bilinear',
                                     align_corners=False)
                feat = torch.cat([feat, skip], dim=1)

            feat = self.styled_convs[2 * i](feat, styles[si]);     si += 1
            feat = self.styled_convs[2 * i + 1](feat, styles[si]); si += 1
            feat = self.dec_attns[i](feat)

            new_rgb = self.to_rgbs[i](feat, styles[si]); si += 1
            if rgb is None:
                rgb = new_rgb
            else:
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear',
                                    align_corners=False) + new_rgb

        return torch.tanh(rgb)


# ============================================================
# MULTI-SCALE DISCRIMINATOR  (GigaGAN design)
# ============================================================

class DiscResBlock(nn.Module):
    """Residual downsampling block for the discriminator."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0),
        )

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class ScaleDiscriminator(nn.Module):
    """Single-scale residual discriminator with self-attention.

    Downsamples until 4×4 → dense → scalar.
    """

    def __init__(self, in_channels, base_ch=64, max_ch=512, attn_at=32):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        ch = base_ch
        spatial = 512   # assume starting from 512; works at smaller inputs too
        while spatial > 4:
            next_ch = min(ch * 2, max_ch)
            blocks.append(DiscResBlock(ch, next_ch))
            spatial //= 2
            if spatial == attn_at:
                blocks.append(SelfAttention(next_ch))
            ch = next_ch
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Sequential(
            nn.Flatten(),
            EqualLinear(ch * 4 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
            EqualLinear(512, 1),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        return self.final(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator (GigaGAN design).

    Applies N separate sub-discriminators at progressively downscaled
    resolutions.  Returns a **list** of scalar predictions.

    Input: cat([image(3), person(3), cloth(3)], dim=channel) → [B, 9, 512, 512]
           (or [B, 6, ...] for OOTD)
    Output: list of [B, 1] scores (one per scale)
    """

    def __init__(self, in_channels=9, n_scales=3):
        super().__init__()
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList()

        for i in range(n_scales):
            # Each sub-D gets smaller input: 512, 256, 128 …
            base = max(32, 64 >> i)  # slightly smaller base ch at lower res
            self.discriminators.append(
                ScaleDiscriminator(in_channels, base_ch=64))

    def forward(self, image, condition):
        """
        image:     [B, 3, H, W] — real or fake try-on
        condition: [B, C, H, W] — cat([person, cloth], dim=1) or cloth-only
        Returns:   list of [B, 1] predictions (one per scale)
        """
        x = torch.cat([image, condition], dim=1)
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = F.avg_pool2d(x, 2)
            outputs.append(disc(x))
        return outputs


# ============================================================
# LOSSES
# ============================================================

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27], vgg[27:36],
        ])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weights = [1/32, 1/16, 1/8, 1/4, 1.0]

    def forward(self, x, y):
        x = ((x + 1) / 2 - self.mean) / self.std
        y = ((y + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for w, blk in zip(self.weights, self.blocks):
            x = blk(x); y = blk(y)
            loss += w * F.l1_loss(x, y)
        return loss


class GANLoss(nn.Module):
    """Hinge loss (standard in SOTA GANs: BigGAN, GigaGAN, StyleGAN-XL)."""

    def forward(self, pred, target_is_real):
        """pred: single tensor [B,1] **or** list of tensors (multi-scale D)."""
        if isinstance(pred, (list, tuple)):
            return sum(self._hinge(p, target_is_real) for p in pred) / len(pred)
        return self._hinge(pred, target_is_real)

    @staticmethod
    def _hinge(pred, target_is_real):
        if target_is_real:
            return F.relu(1.0 - pred).mean()
        return F.relu(1.0 + pred).mean()


def r1_penalty(real_pred, real_images):
    """R1 gradient penalty (Mescheder et al., 2018).

    Should be called with real_images that have requires_grad=True.
    ``real_pred`` can be a single tensor or list (multi-scale D).
    """
    if isinstance(real_pred, (list, tuple)):
        total = sum(p.sum() for p in real_pred)
    else:
        total = real_pred.sum()
    grad, = torch.autograd.grad(total, real_images, create_graph=True)
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


# ============================================================
# MODEL WRAPPER
# ============================================================

class TryOnGANModel:
    """Wraps GigaGAN-based generator + multi-scale discriminator."""

    def __init__(self, in_channels_g=6, in_channels_d=9,
                 style_dim=512, n_kernels=4, n_disc_scales=3):
        print("Initialising GigaGAN-based TryOnGAN …")
        self.generator = GigaGANTryOnGenerator(
            in_channels=in_channels_g, style_dim=style_dim,
            n_kernels=n_kernels)
        self.discriminator = MultiScaleDiscriminator(
            in_channels=in_channels_d, n_scales=n_disc_scales)
        g_p = sum(p.numel() for p in self.generator.parameters())
        d_p = sum(p.numel() for p in self.discriminator.parameters())
        print(f"✓ GigaGAN TryOnGAN  (G: {g_p:,}  D: {d_p:,}  total: {g_p + d_p:,})")

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        return self
