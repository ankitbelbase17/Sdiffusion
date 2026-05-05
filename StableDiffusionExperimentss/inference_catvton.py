"""
CatVTON single-image inference using the exact training/eval inference path.
"""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from model import SDModel
from utils import decode_latents, run_full_inference

_IMAGE_SIZE = 512


def _load_image(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size > 0:
        img = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(img)
    else:
        # Full-res native mode: ensure dimensions are multiples of 8 for VAE
        w, h = img.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        img = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)(img)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(img).unsqueeze(0)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t[0].permute(1, 2, 0).cpu().float().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def _clean_state_dict(sd):
    clean = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        clean[nk] = v
    return clean


def _load_checkpoint(unet, ckpt_path: str):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "unet_state_dict" in ckpt:
        state_dict = ckpt["unet_state_dict"]
        step = ckpt.get("step", "?")
        print(f"  -> unet_state_dict found (saved at step {step})")
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = unet.load_state_dict(_clean_state_dict(state_dict), strict=False)
    if missing:
        print(f"  [warn] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print("Checkpoint loaded")


@torch.no_grad()
def run_inference(model: SDModel, person: torch.Tensor, cloth: torch.Tensor, device: torch.device, steps: int, ootd: bool):
    person = person.to(device)
    cloth = cloth.to(device)

    cond_input = cloth if ootd else torch.cat([person, cloth], dim=3)
    cond_latents = model.vae.encode(cond_input).latent_dist.sample() * 0.18215

    print(f"  Running {steps}-step denoising (training/eval scheduler path)...")
    pred_latents = run_full_inference(model, cond_latents, num_inference_steps=steps)
    pred_wide = decode_latents(model.vae, pred_latents, decode_batch_size=1, vae_fp16=(device.type == "cuda"))

    if ootd:
        return pred_wide
    return pred_wide[:, :, :, : cloth.shape[-1]]


def parse_args():
    p = argparse.ArgumentParser(description="CatVTON inference (scheduler-aligned with training/eval)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--person", required=True)
    p.add_argument("--cloth", required=True)
    p.add_argument("--output", default="tryon_result.png")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--size", type=int, default=_IMAGE_SIZE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ootd", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    for label, path in (("--checkpoint", args.checkpoint), ("--person", args.person), ("--cloth", args.cloth)):
        if not os.path.exists(path):
            print(f"[ERROR] {label} path does not exist: {path}", file=sys.stderr)
            sys.exit(1)

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    print(f"Device: {device}")

    model = SDModel().to(device)
    model.unet.eval()
    model.vae.eval()
    _load_checkpoint(model.unet, args.checkpoint)

    print(f"Loading person: {args.person}")
    person = _load_image(args.person, args.size)
    print(f"Loading cloth:  {args.cloth}")
    cloth = _load_image(args.cloth, args.size)
    
    # Ensure cloth matches person dimensions (vital for concatenation)
    if cloth.shape[-2:] != person.shape[-2:]:
        cloth = torch.nn.functional.interpolate(cloth, size=person.shape[-2:], mode="bilinear", align_corners=False)

    print("\nRunning inference ...")
    result = run_inference(model, person, cloth, device, args.steps, args.ootd)

    out_img = _tensor_to_pil(result)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    out_img.save(args.output)
    print(f"\nSaved try-on result -> {args.output}")
    print(f"Output size: {out_img.size[0]}x{out_img.size[1]}")


if __name__ == "__main__":
    main()
