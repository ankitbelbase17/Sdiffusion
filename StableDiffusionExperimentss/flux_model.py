"""
flux_model.py — Flux 1.dev DiT model for CatVTON-style virtual try-on.

Architecture overview
─────────────────────
•  VAE        : Flux AutoencoderKL — 16 latent channels, scale=0.3611, shift=0.1159
•  Denoiser   : FluxTransformer2DModel (DiT) — UNMODIFIED architecture
•  Scheduler  : FlowMatchEulerDiscreteScheduler (rectified flow / OT)

CatVTON-style conditioning (spatial concatenation)
──────────────────────────────────────────────────
Unlike the SD 1.5 UNet approach (channel-concat → modified conv_in), the Flux DiT
needs NO architecture changes.  We spatially concatenate the conditioning with the
noisy target along the width axis:

    cond_lat    = VAE( cat([person, cloth], W) )    [B, 16, H, 2W]
    target_lat  = VAE( cat([gt,     cloth], W) )    [B, 16, H, 2W]
    noisy       = (1-σ)·target_lat + σ·noise        [B, 16, H, 2W]
    full_input  = cat([noisy, cond_lat], W)          [B, 16, H, 4W]
    pack → DiT → unpack → take left half → velocity loss

After packing (2×2 patches), the DiT just sees more tokens — the self-attention
naturally learns to attend between noisy-target and conditioning tokens.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import FluxTransformer2DModel

try:
    from config import FLUX_MODEL_NAME
except Exception:
    FLUX_MODEL_NAME = "black-forest-labs/FLUX.1-dev"


# ============================================================
# LATENT PACKING / UNPACKING  (2×2 patches → sequence)
# ============================================================

def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Pack 2D latents into a 1D sequence for the Flux DiT.

    [B, C, H, W]  →  [B, (H/2)·(W/2), C·4]

    Each 2×2 spatial patch is flattened into the channel dimension.
    """
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()   # [B, H/2, W/2, C, 2, 2]
    latents = latents.view(B, (H // 2) * (W // 2), C * 4)
    return latents


def unpack_latents(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Unpack a sequence back to 2D latents (inverse of ``pack_latents``).

    [B, (H/2)·(W/2), C·4]  →  [B, C, H, W]
    """
    B, _seq, dim = packed.shape
    C = dim // 4
    h, w = height // 2, width // 2
    packed = packed.view(B, h, w, C, 2, 2)
    packed = packed.permute(0, 3, 1, 4, 2, 5).contiguous()      # [B, C, h, 2, w, 2]
    return packed.view(B, C, height, width)


def prepare_image_ids(height: int, width: int, device, dtype) -> torch.Tensor:
    """
    Create rotary-embedding positional IDs for image patches (after 2×2 packing).

    Returns  [h·w, 3]  where h = height/2, w = width/2.
    Each row is ``(0, y, x)`` — channel index 0, spatial position (y, x).
    """
    h, w = height // 2, width // 2
    img_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
    img_ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
    return img_ids.reshape(h * w, 3)


# ============================================================
# FLUX DIT MODEL
# ============================================================

class FluxDiTModel:
    """
    Wraps Flux 1.dev components for CatVTON-style virtual try-on training.

    The DiT architecture is left **completely unmodified**: conditioning is achieved
    via spatial concatenation of tokens, not channel-concatenation on a conv stem.

    NOTE:
      •  Flux 1.dev is gated on HuggingFace — you must accept the license and
         run ``huggingface-cli login`` before loading the model.
      •  The full model (~12 B params) needs ~24 GB VRAM in bf16.
         Use ``gradient_checkpointing=True`` (default) and/or attention-only
         training (``--train_mode attention_only``) to reduce memory.
    """

    def __init__(self, dtype=torch.bfloat16, gradient_checkpointing: bool = True):
        self.dtype = dtype
        print(f"Loading Flux 1.dev from {FLUX_MODEL_NAME} (dtype={dtype}) ...")

        # ── VAE (16-channel latents) ────────────────────────────
        self.vae = AutoencoderKL.from_pretrained(
            FLUX_MODEL_NAME, subfolder="vae", torch_dtype=dtype,
        )
        self.vae.requires_grad_(False)
        self.vae_scale: float = self.vae.config.scaling_factor   # 0.3611
        self.vae_shift: float = self.vae.config.shift_factor     # 0.1159

        # ── Transformer (DiT) — no architecture changes ────────
        self.transformer = FluxTransformer2DModel.from_pretrained(
            FLUX_MODEL_NAME, subfolder="transformer", torch_dtype=dtype,
        )
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            print("   ✓ Gradient checkpointing enabled on Transformer")

        # ── Scheduler (flow matching) ──────────────────────────
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            FLUX_MODEL_NAME, subfolder="scheduler",
        )

        _n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"✓ Flux DiT loaded  (VAE: 16 ch, scale={self.vae_scale:.4f}, "
              f"shift={self.vae_shift:.4f};  Transformer: {_n_params:,} params)")

    # ── device helpers ──────────────────────────────────────────
    def to(self, device):
        self.vae.to(device)
        self.transformer.to(device)
        return self

    # ── latent encode / decode with Flux scaling ────────────────
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Pixel-space image ([-1,1]) → Flux latent (scaled + shifted)."""
        latent = self.vae.encode(image).latent_dist.sample()
        return (latent - self.vae_shift) * self.vae_scale

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Flux latent → pixel-space image clamped to [0,1]."""
        latent = latent / self.vae_scale + self.vae_shift
        image = self.vae.decode(latent).sample
        return (image / 2 + 0.5).clamp(0, 1)


# ============================================================
# PARAMETER UTILITIES  (Flux-specific)
# ============================================================

def count_parameters_flux(transformer, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    return sum(p.numel() for p in transformer.parameters())


def freeze_non_attention_flux(transformer):
    """Freeze everything except attention projections in the Flux DiT."""
    for param in transformer.parameters():
        param.requires_grad = False

    attn_modules = []
    for name, module in transformer.named_modules():
        if "attn" in name:
            for param in module.parameters():
                param.requires_grad = True
            attn_modules.append(name)

    print(f"✓ Unfroze {len(attn_modules)} attention modules in Flux DiT")
    return transformer


def print_trainable_params_flux(model: FluxDiTModel, mode: str) -> int:
    """Print detailed trainable parameter statistics for the Flux DiT."""
    print("\n" + "=" * 60)
    print(f"TRAINABLE PARAMETERS — Flux DiT ({mode})")
    print("=" * 60)

    total     = count_parameters_flux(model.transformer, trainable_only=False)
    trainable = count_parameters_flux(model.transformer, trainable_only=True)
    frozen    = total - trainable

    print(f"\n  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters:    {frozen:,}")
    print(f"  Trainable ratio:      {100 * trainable / total:.2f}%")

    print(f"\n  Top trainable layer groups:")
    layer_counts: dict[str, int] = {}
    for name, param in model.transformer.named_parameters():
        if param.requires_grad:
            parts = name.split(".")
            key = ".".join(parts[:3]) if len(parts) > 3 else name
            layer_counts[key] = layer_counts.get(key, 0) + param.numel()

    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"      {layer}: {count:,}")
    if len(layer_counts) > 20:
        print(f"      ... and {len(layer_counts) - 20} more groups")

    print("=" * 60 + "\n")
    return trainable
