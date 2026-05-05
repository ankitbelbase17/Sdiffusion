"""
inference_gan.py — Self-contained GigaGAN inference for virtual try-on.

Architecture (channel-wise concatenation):
    gen_input  = cat([person, cloth], dim=1)   [B, 6, 512, 512]
    gen_output = Generator(gen_input)          [B, 3, 512, 512]
    tryon      = gen_output (direct pixel output, no VAE)

OOTD mode (--ootd):
    gen_input  = cloth only                    [B, 3, 512, 512]
    output     = gen_output

Usage:
    python inference_gan.py \
        --checkpoint /path/to/ckpt_final.pt \
        --person     /path/to/person.jpg \
        --cloth      /path/to/cloth.jpg \
        --output     result.png
"""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from tryongan_model import GigaGANTryOnGenerator

# ── Constants ──────────────────────────────────────────────────────
_IMAGE_SIZE = 512


# ── Helpers ────────────────────────────────────────────────────────
def _load_image(path: str, size: int) -> torch.Tensor:
    """Load an RGB image, resize to (size, size), normalise to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(img).unsqueeze(0)  # [1, 3, H, W]


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] tensor in [-1, 1] to a PIL image."""
    t01 = (t[0].clamp(-1, 1) + 1) / 2
    arr = (t01.permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


# ── Model loading ──────────────────────────────────────────────────
def build_generator(in_channels: int, style_dim: int, n_kernels: int) -> GigaGANTryOnGenerator:
    """Instantiate the GigaGAN generator."""
    print(f"Building GigaGANTryOnGenerator (in_channels={in_channels}, "
          f"style_dim={style_dim}, n_kernels={n_kernels}) …")
    G = GigaGANTryOnGenerator(
        in_channels=in_channels,
        style_dim=style_dim,
        n_kernels=n_kernels,
    )
    G.requires_grad_(False)
    return G


def load_checkpoint(G: GigaGANTryOnGenerator, ckpt_path: str):
    """Load fine-tuned generator weights from a training checkpoint .pt file."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "g_state_dict" in ckpt:
        state_dict = ckpt["g_state_dict"]
        step = ckpt.get("step", "?")
        print(f"  ↳ g_state_dict found  (saved at step {step})")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip 'module.' prefix added by DDP
    clean = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Strip '_orig_mod.' prefix added by torch.compile
    clean = {k.replace("_orig_mod.", ""): v for k, v in clean.items()}

    missing, unexpected = G.load_state_dict(clean, strict=False)
    if missing:
        print(f"  ⚠ Missing keys  ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    print("✓ Checkpoint loaded")


# ── Inference ──────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    G: GigaGANTryOnGenerator,
    person: torch.Tensor,        # [1, 3, H, W] in [-1, 1]
    cloth: torch.Tensor,         # [1, 3, H, W] in [-1, 1]
    device: torch.device,
    dtype: torch.dtype,
    ootd: bool = False,
) -> torch.Tensor:               # returns [1, 3, H, W] in [-1, 1]
    """Single forward pass through the generator (no iterative denoising)."""
    person = person.to(device, dtype)
    cloth = cloth.to(device, dtype)

    if ootd:
        gen_input = cloth                                     # [1, 3, H, W]
    else:
        gen_input = torch.cat([person, cloth], dim=1)         # [1, 6, H, W]

    print("  Running generator forward pass …")
    output = G(gen_input)                                     # [1, 3, H, W]
    return output


# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="GigaGAN inference — fine-tuned for virtual try-on"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained checkpoint (.pt file, e.g. ckpt_final.pt)")
    p.add_argument("--person", required=True,
                   help="Path to person / model image (jpg/png)")
    p.add_argument("--cloth", required=True,
                   help="Path to garment / cloth image (jpg/png)")
    p.add_argument("--output", default="tryon_result_gan.png",
                   help="Output image path (default: tryon_result_gan.png)")
    p.add_argument("--size", type=int, default=_IMAGE_SIZE,
                   help=f"Input/output resolution (default: {_IMAGE_SIZE})")
    p.add_argument("--device", default="cuda",
                   help="Device: cuda / cpu / cuda:1 etc. (default: cuda)")
    p.add_argument("--fp16", action="store_true",
                   help="Use float16 (faster on GPU, needs CUDA)")
    p.add_argument("--ootd", action="store_true",
                   help="OOTD mode: cloth-only conditioning (no person concat)")
    p.add_argument("--style_dim", type=int, default=512,
                   help="Style dimension (must match training, default: 512)")
    p.add_argument("--n_kernels", type=int, default=4,
                   help="Number of adaptive kernels (must match training, default: 4)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ─────────────────────────────────────────
    for label, path in [("--checkpoint", args.checkpoint),
                        ("--person", args.person),
                        ("--cloth", args.cloth)]:
        if not os.path.exists(path):
            print(f"[ERROR] {label} path does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    # ── Build & load model ──────────────────────────────────────
    in_channels = 3 if args.ootd else 6
    G = build_generator(in_channels, args.style_dim, args.n_kernels)
    load_checkpoint(G, args.checkpoint)
    G.to(device, dtype)
    G.eval()

    # ── Load images ─────────────────────────────────────────────
    print(f"Loading person: {args.person}")
    person = _load_image(args.person, args.size)
    print(f"Loading cloth:  {args.cloth}")
    cloth = _load_image(args.cloth, args.size)

    # ── Run inference ───────────────────────────────────────────
    print("\nRunning inference …")
    result = run_inference(
        G=G,
        person=person, cloth=cloth,
        device=device, dtype=dtype,
        ootd=args.ootd,
    )

    # ── Save output ──────────────────────────────────────────────
    out_img = _tensor_to_pil(result)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    out_img.save(args.output)
    print(f"\n✓ Saved try-on result → {args.output}")
    print(f"  Output size: {out_img.size[0]}×{out_img.size[1]}")


if __name__ == "__main__":
    main()
