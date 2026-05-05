"""
inference_tryon.py — Standalone virtual try-on inference
=========================================================
Loads a trained UNet checkpoint and generates a try-on image
for a given (person, cloth) pair.

Usage
-----
python inference_tryon.py \
    --checkpoint  path/to/checkpoint_step_1000.pt \
    --person      path/to/person.png \
    --cloth       path/to/cloth.png \
    --output      result.png \
    --steps       50

All other flags have sensible defaults (512 × 512, fp16 on CUDA).
"""

import argparse
import os
import io

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)

try:
    from config import MODEL_NAME, IMAGE_SIZE
except Exception:
    MODEL_NAME = os.getenv("MODEL_NAME", "runwayml/stable-diffusion-v1-5")
    IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))


# ─────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────

def load_image(path: str, size: int) -> torch.Tensor:
    """Load a local image → normalised [-1, 1] tensor [1, 3, H, W]."""
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),                                # [0, 1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)   # [1, 3, H, W]


def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents → [B, 3, H, W] in [0, 1]."""
    with torch.no_grad():
        imgs = vae.decode(latents / 0.18215).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[1, 3, H, W] float [0,1] → PIL Image."""
    arr = (t[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_models(checkpoint: str, device: torch.device, fp16: bool):
    """
    Load SD-1.5 components from HuggingFace and overlay the UNet
    weights from the given checkpoint .pt file.

    Scheduler: DDIMScheduler (deterministic, high quality, 50 steps).
    """
    dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32

    print(f"Loading base models from  {MODEL_NAME}  ...")
    vae  = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device, dtype)
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device, dtype)

    # DDIM scheduler — set during inference; training used DDPMScheduler
    scheduler = DDIMScheduler.from_pretrained(
        MODEL_NAME,
        subfolder="scheduler",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    # Freeze everything (inference only)
    for m in (vae, unet):
        m.requires_grad_(False)
        m.eval()

    # ── Expand UNet conv_in: 4 → 12 channels ───────────────────
    old_conv = unet.conv_in
    new_conv = torch.nn.Conv2d(8, old_conv.out_channels,
                               kernel_size=old_conv.kernel_size,
                               padding=old_conv.padding,
                               bias=old_conv.bias is not None)
    with torch.no_grad():
        new_conv.weight[:, :4] = old_conv.weight
        torch.nn.init.xavier_uniform_(new_conv.weight[:, 4:])
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv
    unet.config["in_channels"] = 8

    # ── Load fine-tuned UNet weights ────────────────────────────
    print(f"Loading UNet checkpoint:  {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("unet_state_dict", ckpt)   # support both formats
    unet.load_state_dict(state, strict=True)
    print(f"  ✓ Checkpoint loaded  (step={ckpt.get('step', '?')}, "
          f"epoch={ckpt.get('epoch', '?')})")

    return vae, unet, scheduler


# ─────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    person_path: str,
    cloth_path:  str,
    checkpoint:  str,
    output_path: str,
    num_steps:   int   = 50,
    size:        int   = IMAGE_SIZE,
    guidance_scale: float = 1.0,   # classifier-free guidance scale (1.0 = unconditional)
    seed:        int   = 42,
    fp16:        bool  = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if (fp16 and device.type == "cuda") else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}  |  steps: {num_steps}")

    # ── Load images ─────────────────────────────────────────
    person = load_image(person_path, size).to(device, dtype)   # [1, 3, H, W]
    cloth  = load_image(cloth_path,  size).to(device, dtype)   # [1, 3, H, W]

    # ── Load models ─────────────────────────────────────────
    vae, unet, scheduler = load_models(checkpoint, device, fp16)

    # No text encoder — fixed zero embedding [1, 77, 768]
    text_emb = torch.zeros(1, 77, 768, device=device, dtype=dtype)

    # ── Encode condition: cat([person, cloth], width) → [1,3,512,1024] → VAE → [1,4,64,128]
    cond_input   = torch.cat([person, cloth], dim=3)                     # [1,3,512,1024]
    cond_latents = vae.encode(cond_input).latent_dist.sample() * 0.18215 # [1,4,64,128]
    cond_latents = cond_latents.to(dtype)

    # ── Noise initialisation ─────────────────────────────────
    # Same shape as cond_latents: [1, 4, 64, 128]  (double-wide spatial)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents   = torch.randn(cond_latents.shape,
                            generator=generator, device=device, dtype=dtype)  # [1,4,64,128]

    # ── DDIM denoising loop ──────────────────────────────────
    scheduler.set_timesteps(num_steps, device=device)
    latents = latents * scheduler.init_noise_sigma          # scale initial noise

    print(f"Running {num_steps} DDIM denoising steps ...")
    for i, t in enumerate(scheduler.timesteps):
        # Channel-concat: [noisy(4) ‖ cond(4)] → [1,8,64,128]
        unet_input = torch.cat([latents, cond_latents], dim=1)

        noise_pred = unet(unet_input, t, text_emb).sample   # [1,4,64,128]

        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0 or (i + 1) == num_steps:
            print(f"  step {i+1}/{num_steps}")

    # ── Decode to pixel space ───────────────────────────────
    # latents [1,4,64,128] → VAE decode → [1,3,512,1024] → left half = try-on
    decoded_wide = decode_latents(vae, latents)          # [1, 3, 512, 1024]
    tryon        = decoded_wide[:, :, :, :size]          # [1, 3, 512, 512]

    # ── Save ────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # Side-by-side comparison: person | cloth | try-on
    person_vis = (person / 2 + 0.5).clamp(0, 1)
    cloth_vis  = (cloth  / 2 + 0.5).clamp(0, 1)
    grid = torch.cat([person_vis, cloth_vis, tryon], dim=3)  # [1,3,H,3W]
    save_image(grid, output_path)
    print(f"\n✓ Saved try-on result → {output_path}")

    # Also save the try-on alone
    base, ext = os.path.splitext(output_path)
    tryon_only_path = f"{base}_tryon_only{ext}"
    save_image(tryon, tryon_only_path)
    print(f"✓ Saved try-on only   → {tryon_only_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Virtual Try-On Inference — SD-1.5 spatial-concat UNet"
    )
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the .pt checkpoint file (contains unet_state_dict)")
    parser.add_argument("--person",     type=str, required=True,
                        help="Path to the person / initial image (512×512 or any size, auto-resized)")
    parser.add_argument("--cloth",      type=str, required=True,
                        help="Path to the garment / cloth flat-lay image")
    # Output
    parser.add_argument("--output",     type=str, default="tryon_output.png",
                        help="Output image path (side-by-side grid saved here; "
                             "{name}_tryon_only.png also written)")
    # Optional tuning
    parser.add_argument("--size",       type=int,   default=IMAGE_SIZE,
                        help=f"Image size (default: {IMAGE_SIZE})")
    parser.add_argument("--steps",      type=int,   default=50,
                        help="Number of DDIM denoising steps (default: 50)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_fp16",    action="store_true",
                        help="Force float32 even on CUDA (slower, more VRAM)")

    args = parser.parse_args()

    # Validate inputs
    for flag, path in [("--checkpoint", args.checkpoint),
                       ("--person",     args.person),
                       ("--cloth",      args.cloth)]:
        if not os.path.isfile(path):
            parser.error(f"{flag}: file not found → {path}")

    run_inference(
        person_path  = args.person,
        cloth_path   = args.cloth,
        checkpoint   = args.checkpoint,
        output_path  = args.output,
        num_steps    = args.steps,
        size         = args.size,
        seed         = args.seed,
        fp16         = not args.no_fp16,
    )
