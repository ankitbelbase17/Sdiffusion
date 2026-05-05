"""
hunyuan_model.py — HunyuanDiT v1.1 for CatVTON-style virtual try-on.

Architecture overview
─────────────────────
•  VAE        : Standard SD AutoencoderKL — 4 latent channels, scale=0.18215
•  Denoiser   : HunyuanDiT2DModel (DiT) — UNMODIFIED architecture
•  Scheduler  : DDPMScheduler (training noise)  +
                UniPCMultistepScheduler (inference denoising)

CatVTON-style conditioning (spatial concatenation)
──────────────────────────────────────────────────
NO architecture modifications needed. The conditioning is spatially concatenated
with the noisy target latents along the width axis so the DiT processes a wider
image — exactly as in the Flux CatVTON approach, but without sequence packing
(HunyuanDiT2DModel takes 2D [B, C, H, W] inputs directly).

    cond_lat    = VAE(cat([person, cloth], W))    [B, 4, H, 2W]
    target_lat  = VAE(cat([gt,     cloth], W))    [B, 4, H, 2W]
    noisy       = DDPM.add_noise(target_lat, ε, t) [B, 4, H, 2W]
    full_input  = cat([noisy, cond_lat], C)         [B, 8, H, 2W]
    → DiT → take first 4 channels → ε loss

Rotary positional embeddings (RoPE) are computed for latent spatial size H × 2W.

PREREQUISITES:
  • No gated license — public model on HuggingFace.
  • ~10–14 GB VRAM in float16 with gradient checkpointing.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from diffusers.models.transformers import HunyuanDiT2DModel

try:
    from config import HUNYUAN_MODEL_NAME
except Exception:
    HUNYUAN_MODEL_NAME = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"


def _get_parent_and_attr(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


# ============================================================
# ROTARY POSITIONAL EMBEDDING HELPER
# ============================================================

def prepare_rotary_pos_embed(
    H_lat: int, W_lat: int,
    head_dim: int, patch_size: int,
    device, dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute 2-D RoPE (cos, sin) embeddings for a latent of spatial size [H_lat, W_lat].

    The HunyuanDiT patchifies with `patch_size`×`patch_size` patches, so the grid is
    [H_lat // patch_size, W_lat // patch_size].

    Returns:
        cos, sin  — each [grid_h * grid_w, head_dim // 2]  on `device` / `dtype`
    """
    try:
        from diffusers.models.embeddings import get_2d_rotary_pos_embed
    except ImportError:
        from diffusers.models.attention import get_2d_rotary_pos_embed   # older diffusers

    grid_h = H_lat // patch_size
    grid_w = W_lat // patch_size

    cos, sin = get_2d_rotary_pos_embed(
        embed_dim=head_dim,
        crops_coords=((0, 0), (grid_h, grid_w)),
        grid_size=(grid_h, grid_w),
        use_real=True,
    )
    cos = torch.tensor(cos, device=device, dtype=dtype)
    sin = torch.tensor(sin, device=device, dtype=dtype)
    return cos, sin


# ============================================================
# HUNYUANDIT MODEL WRAPPER
# ============================================================

class HunyuanDiTModel:
    """
    Wraps HunyuanDiT v1.1 components for CatVTON-style virtual try-on.

    Key differences from Flux:
      • 4-channel VAE (standard SD),  not 16-channel
      • DDPM noise schedule (epsilon prediction),  not flow matching
      • 2D spatial DiT input — NO sequence packing / unpacking
      • RoPE positional embeddings recomputed per spatial size
      • ~1.5 B params in float16 (~6 GB VRAM)
    """

    def __init__(self, dtype=torch.float16, gradient_checkpointing: bool = True, use_vae: bool = True):
        self.dtype = dtype
        self.use_vae = use_vae
        print(f"Loading HunyuanDiT v1.1 from {HUNYUAN_MODEL_NAME} (dtype={dtype}) ...")

        if self.use_vae:
            self.vae = AutoencoderKL.from_pretrained(
                HUNYUAN_MODEL_NAME, subfolder="vae", torch_dtype=dtype,
            )
            self.vae.requires_grad_(False)
            self.vae_scale = self.vae.config.scaling_factor
            in_ch = 8  # noisy(4) + cond(4)
        else:
            self.vae = None
            self.vae_scale = 1.0
            in_ch = 6  # noisy(3) + cond(3), pixel-space

        self.transformer = HunyuanDiT2DModel.from_pretrained(
            HUNYUAN_MODEL_NAME, subfolder="transformer", torch_dtype=dtype,
        )
        self._adapt_input_channels(in_ch)
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            print("   Gradient checkpointing enabled on Transformer")

        self.scheduler = DDPMScheduler.from_pretrained(
            HUNYUAN_MODEL_NAME, subfolder="scheduler",
        )
        self.inference_scheduler = UniPCMultistepScheduler.from_config(
            self.scheduler.config,
        )

        cfg = self.transformer.config
        self.patch_size = getattr(cfg, "patch_size", 2)
        self.num_heads = getattr(cfg, "num_attention_heads", 16)
        self.hidden_size = getattr(cfg, "hidden_size", 1408)
        self.head_dim = self.hidden_size // self.num_heads
        self.cross_attn_dim = getattr(cfg, "cross_attention_dim", 1024)

        _n = sum(p.numel() for p in self.transformer.parameters())
        print(
            f"Loaded HunyuanDiT v1.1 "
            f"(VAE: {'enabled' if self.use_vae else 'disabled (pixel-space)'}, scale={self.vae_scale:.5f}; "
            f"Transformer: {_n:,} params, head_dim={self.head_dim}, patch={self.patch_size}x{self.patch_size})"
        )
    def _adapt_input_channels(self, in_channels: int):
        """
        Expand the first patch-embed conv to support channel-concat conditioning.
        Pretrained 4-channel weights are copied to the first 4 input channels;
        extra channels are initialized to zeros.
        """
        first_conv_name = None
        first_conv = None
        for n, m in self.transformer.named_modules():
            if isinstance(m, nn.Conv2d):
                first_conv_name = n
                first_conv = m
                break
        if first_conv is None:
            raise RuntimeError("Could not find transformer input Conv2d to adapt in_channels.")
        if first_conv.in_channels == in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=(first_conv.bias is not None),
            padding_mode=first_conv.padding_mode,
            device=first_conv.weight.device,
            dtype=first_conv.weight.dtype,
        )
        with torch.no_grad():
            new_conv.weight.zero_()
            keep = min(first_conv.in_channels, in_channels)
            new_conv.weight[:, :keep].copy_(first_conv.weight[:, :keep])
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        parent, attr = _get_parent_and_attr(self.transformer, first_conv_name)
        setattr(parent, attr, new_conv)
        if hasattr(self.transformer.config, "in_channels"):
            self.transformer.config.in_channels = in_channels
        print(f"   ✓ Adapted transformer input channels: {first_conv.in_channels} -> {in_channels}")

    # ── device helpers ──────────────────────────────────────────
    def to(self, device):
        if self.vae is not None:
            self.vae.to(device)
        self.transformer.to(device)
        return self

    # ── latent encode / decode ──────────────────────────────────
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Pixel-space image ([-1,1]) -> latent/pixel tensor used by DiT."""
        if self.use_vae:
            latent = self.vae.encode(image).latent_dist.sample()
            return latent * self.vae_scale
        return image

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent/pixel tensor -> pixel-space image in [0,1]."""
        if self.use_vae:
            image = self.vae.decode(latent / self.vae_scale).sample
            return (image / 2 + 0.5).clamp(0, 1)
        return (latent / 2 + 0.5).clamp(0, 1)

    # ── RoPE helper ─────────────────────────────────────────────
    def get_rope_embed(
        self, H_lat: int, W_lat: int, device, dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE (cos, sin) for a latent of size [H_lat, W_lat]."""
        return prepare_rotary_pos_embed(
            H_lat, W_lat, self.head_dim, self.patch_size, device, dtype,
        )


# ============================================================
# PARAMETER UTILITIES
# ============================================================

def count_parameters_hunyuan(transformer, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    return sum(p.numel() for p in transformer.parameters())


def freeze_non_attention_hunyuan(transformer):
    """Freeze everything except attention projections in the HunyuanDiT."""
    for param in transformer.parameters():
        param.requires_grad = False

    attn_modules = []
    for name, module in transformer.named_modules():
        if any(k in name for k in ("attn", "attention")):
            for param in module.parameters():
                param.requires_grad = True
            attn_modules.append(name)

    print(f"✓ Unfroze {len(attn_modules)} attention modules in HunyuanDiT")
    return transformer


def print_trainable_params_hunyuan(model: HunyuanDiTModel, mode: str) -> int:
    """Print detailed trainable parameter stats for the HunyuanDiT."""
    print("\n" + "=" * 60)
    print(f"TRAINABLE PARAMETERS — HunyuanDiT v1.1 ({mode})")
    print("=" * 60)

    total     = count_parameters_hunyuan(model.transformer, trainable_only=False)
    trainable = count_parameters_hunyuan(model.transformer, trainable_only=True)
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
        print(f"      … and {len(layer_counts) - 20} more groups")

    return trainable




