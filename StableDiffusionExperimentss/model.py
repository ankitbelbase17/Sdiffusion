"""
model.py — SDModel definition, parameter utilities, and freeze helpers.
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

try:
    from config import MODEL_NAME
except Exception:
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"


# ============================================================
# MODEL
# ============================================================
class SDModel:
    def __init__(self):
        print(f"Loading {MODEL_NAME}...")
        self.vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")
        self.scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

        # Freeze VAE
        self.vae.requires_grad_(False)
        # UNet input = cat([noisy_latent(4) ‖ cond_latent(4)], dim=1)
        # where cond_latent = VAE(cat([person, cloth], dim=3))  [B,4,64,128]
        old_conv = self.unet.conv_in                    # Conv2d(4, 320, kernel=3, padding=1)
        new_conv = nn.Conv2d(8, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             padding=old_conv.padding,
                             bias=old_conv.bias is not None)
        with torch.no_grad():
            new_conv.weight[:, :4] = old_conv.weight        # preserve noise-channel weights
            nn.init.xavier_uniform_(new_conv.weight[:, 4:]) # init cond channels
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv
        self.unet.config["in_channels"] = 8

        print("✓ Model loaded (8-ch UNet: 4 noise + 4 cond, channel-concat; "
              "cond = VAE(cat([person, cloth], width)))")

    def to(self, device):
        self.vae.to(device)
        self.unet.to(device)
        return self


# ============================================================
# PARAMETER UTILITIES
# ============================================================
def count_parameters(model, trainable_only=True):
    """Count total and trainable parameters"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_non_attention(unet):
    """Freeze all parameters except self-attention layers"""
    # First freeze everything
    for param in unet.parameters():
        param.requires_grad = False

    # Unfreeze only attention layers
    attention_modules = []
    for name, module in unet.named_modules():
        # Self-attention layers in UNet
        if 'attn1' in name or 'attn2' in name:
            for param in module.parameters():
                param.requires_grad = True
            attention_modules.append(name)

    print(f"✓ Unfroze {len(attention_modules)} attention modules")
    return unet


def print_trainable_params(model, mode):
    """Print detailed trainable parameters"""
    print("\n" + "="*60)
    print(f"TRAINABLE PARAMETERS ({mode})")
    print("="*60)

    total_params = count_parameters(model.unet, trainable_only=False)
    trainable_params = count_parameters(model.unet, trainable_only=True)
    frozen_params = total_params - trainable_params

    print(f"\nUNet Statistics:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Trainable ratio:      {100*trainable_params/total_params:.2f}%")

    # Group by layer type
    print(f"\nTrainable layers breakdown:")
    layer_counts = {}
    for name, param in model.unet.named_parameters():
        if param.requires_grad:
            # Get layer type from name
            parts = name.split('.')
            layer_type = '.'.join(parts[:3]) if len(parts) > 3 else name
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
            layer_counts[layer_type] += param.numel()

    for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    {layer}: {count:,}")

    if len(layer_counts) > 20:
        print(f"    ... and {len(layer_counts) - 20} more layers")

    print("="*60 + "\n")

    return trainable_params
