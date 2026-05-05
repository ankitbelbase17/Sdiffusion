"""
inference_hunyuan.py — Self-contained HunyuanDiT v1.1 inference for virtual try-on.

Architecture (CatVTON-style spatial concatenation):
    cond_input  = cat([person, cloth], dim=3)   [B, 3, 512, 1024]
    cond_lat    = VAE(cond_input)               [B, 4, 64, 128]
    noisy       = randn(B, 4, 64, 128)
    full_input  = cat([noisy, cond_lat], dim=3) [B, 4, 64, 256]
    → HunyuanDiT2DModel (UniPC denoising) → take left-half noise pred → denoise
    pred_wide   = VAE_decode(denoised)          [B, 3, 512, 1024]
    tryon       = pred_wide[:, :, :, :512]      [B, 3, 512, 512]  ← left half

OOTD mode (--ootd):
    cond_input  = cloth only                    [B, 3, 512, 512]
    output      = full decoded image (no slice)

Usage:
    python inference_hunyuan.py \
        --checkpoint /path/to/ckpt_final.pt \
        --person     /path/to/person.jpg \
        --cloth      /path/to/cloth.jpg \
        --output     result.png \
        --steps      50
"""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from hunyuan_model import HunyuanDiTModel

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
    """Convert a [1, 3, H, W] tensor in [0, 1] to a PIL image."""
    arr = (t[0].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


# ── Model loading ──────────────────────────────────────────────────
def build_model(dtype: torch.dtype) -> HunyuanDiTModel:
    """Load HunyuanDiT v1.1 (VAE + Transformer + schedulers)."""
    model = HunyuanDiTModel(dtype=dtype, gradient_checkpointing=False)
    return model


def load_checkpoint(model: HunyuanDiTModel, ckpt_path: str):
    """Load fine-tuned transformer weights from a training checkpoint .pt file."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "transformer_state_dict" in ckpt:
        state_dict = ckpt["transformer_state_dict"]
        step = ckpt.get("step", "?")
        print(f"  ↳ transformer_state_dict found  (saved at step {step})")
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip 'module.' prefix added by DDP
    clean = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.transformer.load_state_dict(clean, strict=False)
    if missing:
        print(f"  ⚠ Missing keys  ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  ⚠ Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    print("✓ Checkpoint loaded")


# ── Inference ──────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model: HunyuanDiTModel,
    person: torch.Tensor,        # [1, 3, H, W] in [-1, 1]
    cloth: torch.Tensor,         # [1, 3, H, W] in [-1, 1]
    device: torch.device,
    dtype: torch.dtype,
    steps: int = 50,
    ootd: bool = False,
) -> torch.Tensor:               # returns [1, 3, H, W] in [0, 1]
    """Full UniPC denoising loop.  Returns try-on image in [0, 1]."""
    person = person.to(device, dtype)
    cloth = cloth.to(device, dtype)

    # ── Build conditioning input ────────────────────────────────
    if ootd:
        cond_px = cloth                                      # [1, 3, H, W]
    else:
        cond_px = torch.cat([person, cloth], dim=3)          # [1, 3, H, 2W]

    # ── VAE encode conditioning ─────────────────────────────────
    cond_latent = model.encode_image(cond_px)                # [1, 4, H/8, W_cond/8]
    B, C, H_lat, W_lat = cond_latent.shape

    # ── Start from pure noise (target shape) ────────────────────
    latents = torch.randn(B, C, H_lat, W_lat, device=device, dtype=dtype)

    # Full spatial concat width: noisy ‖ cond → [B, 4, H, 2W]
    W_full = W_lat * 2

    # Zero text conditioning (T5 + CLIP, 1 token each)
    txt_emb = torch.zeros(B, 1, model.cross_attn_dim, device=device, dtype=dtype)
    txt_mask = torch.ones(B, 1, device=device, dtype=torch.bool)

    # image_meta_size [B, 6]: (orig_H, orig_W, crop_top, crop_left, tgt_H, tgt_W)
    meta_size = torch.tensor(
        [[H_lat * 8, W_full * 8, 0, 0, H_lat * 8, W_full * 8]] * B,
        device=device, dtype=dtype,
    )
    style_ids = torch.zeros(B, device=device, dtype=torch.long)

    # RoPE embeddings for the full-width latent
    rope_cos, rope_sin = model.get_rope_embed(H_lat, W_full, device, dtype)

    model.inference_scheduler.set_timesteps(steps, device=device)

    # ── Denoising loop ──────────────────────────────────────────
    print(f"  Running {steps}-step UniPC denoising …")
    for i, t in enumerate(model.inference_scheduler.timesteps):
        full_lat = torch.cat([latents, cond_latent], dim=3)  # [B, 4, H, 2W]

        noise_pred_full = model.transformer(
            hidden_states=full_lat,
            timestep=t.expand(B),
            encoder_hidden_states=txt_emb,
            text_embedding_mask=txt_mask,
            encoder_hidden_states_t5=txt_emb,
            text_embedding_mask_t5=txt_mask,
            image_meta_size=meta_size,
            style=style_ids,
            image_rotary_emb=(rope_cos, rope_sin),
            return_dict=False,
        )[0]                                                  # [B, 4, H, 2W]

        noise_pred = noise_pred_full[:, :, :, :W_lat]        # left half only
        latents = model.inference_scheduler.step(noise_pred, t, latents).prev_sample

        if (i + 1) % 10 == 0 or i == steps - 1:
            print(f"    step {i + 1}/{steps}")

    # ── VAE decode ──────────────────────────────────────────────
    decoded = model.decode_latent(latents)                    # [1, 3, H, W_cond]

    # ── Slice try-on (left half) ─────────────────────────────────
    if ootd:
        return decoded                                        # [1, 3, H, W]
    else:
        W = person.shape[3]                                   # 512
        return decoded[:, :, :, :W]                           # left half = try-on


# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="HunyuanDiT v1.1 inference — fine-tuned for virtual try-on"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained checkpoint (.pt file, e.g. ckpt_final.pt)")
    p.add_argument("--person", required=True,
                   help="Path to person / model image (jpg/png)")
    p.add_argument("--cloth", required=True,
                   help="Path to garment / cloth image (jpg/png)")
    p.add_argument("--output", default="tryon_result_hunyuan.png",
                   help="Output image path (default: tryon_result_hunyuan.png)")
    p.add_argument("--steps", type=int, default=50,
                   help="Number of UniPC denoising steps (default: 50)")
    p.add_argument("--size", type=int, default=_IMAGE_SIZE,
                   help=f"Input/output resolution (default: {_IMAGE_SIZE})")
    p.add_argument("--device", default="cuda",
                   help="Device: cuda / cpu / cuda:1 etc. (default: cuda)")
    p.add_argument("--fp16", action="store_true",
                   help="Use float16 (faster on GPU, needs CUDA)")
    p.add_argument("--ootd", action="store_true",
                   help="OOTD mode: cloth-only conditioning (no person concat)")
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
    model = build_model(dtype)
    load_checkpoint(model, args.checkpoint)
    model.to(device)
    model.transformer.eval()

    # ── Load images ─────────────────────────────────────────────
    print(f"Loading person: {args.person}")
    person = _load_image(args.person, args.size)
    print(f"Loading cloth:  {args.cloth}")
    cloth = _load_image(args.cloth, args.size)

    # ── Run inference ───────────────────────────────────────────
    print("\nRunning inference …")
    result = run_inference(
        model=model,
        person=person, cloth=cloth,
        device=device, dtype=dtype,
        steps=args.steps,
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
