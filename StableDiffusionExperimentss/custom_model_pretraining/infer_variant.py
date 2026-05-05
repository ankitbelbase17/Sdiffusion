import argparse
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from meanflow_model import MeanFlowDiT250M, MeanFlowDiTConfig
from model import DiT250M, DiTConfig
from utils import make_beta_schedule, sample_ddim_like


def _build_datapred_model(args):
    cfg = DiTConfig(
        image_size=args.image_size,
        image_height=args.image_size,
        image_width=args.image_width if args.image_width > 0 else args.image_size * 2,
        in_channels=3,
        cond_in_channels=3,
        out_channels=3,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    return DiT250M(cfg)


def _build_meanflow_model(args):
    cfg = MeanFlowDiTConfig(
        image_size=args.image_size,
        image_height=args.image_size,
        image_width=args.image_width if args.image_width > 0 else args.image_size * 2,
        in_channels=3,
        out_channels=3,
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    )
    return MeanFlowDiT250M(cfg)


def _resolve_diffusion_steps(args: argparse.Namespace, ckpt: dict | None) -> int:
    if args.diffusion_steps > 0:
        return int(args.diffusion_steps)
    if ckpt is not None:
        diff_cfg = ckpt.get("diffusion", {})
        if isinstance(diff_cfg, dict) and int(diff_cfg.get("steps", 0)) > 0:
            return int(diff_cfg["steps"])
    raise ValueError("Could not resolve diffusion_steps. Set --diffusion_steps > 0 or use a checkpoint with diffusion.steps metadata.")


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.approach == "datapred":
        if not args.person_path or not args.cloth_path:
            raise ValueError("Datapred inference requires both --person_path and --cloth_path.")
        model = _build_datapred_model(args).to(device).eval()
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        diffusion_steps = _resolve_diffusion_steps(args, ckpt)

        betas = make_beta_schedule(diffusion_steps).to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        sqrt_ab = torch.sqrt(alpha_bar)
        sqrt_1mab = torch.sqrt(1 - alpha_bar)
        shape = (
            args.batch_size,
            3,
            args.image_size,
            args.image_width if args.image_width > 0 else args.image_size * 2,
        )
        sample_wide = sample_ddim_like(
            model=model,
            shape=shape,
            timesteps=diffusion_steps,
            sqrt_ab=sqrt_ab,
            sqrt_1mab=sqrt_1mab,
            device=device,
            cond=torch.cat([
                _load_rgb(args.person_path, args.image_size, args.image_size).unsqueeze(0).to(device),
                _load_rgb(args.cloth_path, args.image_size, args.image_size).unsqueeze(0).to(device),
            ], dim=3).repeat(args.batch_size, 1, 1, 1),
        )
    else:
        model = _build_meanflow_model(args).to(device).eval()
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        h = args.image_size
        w = args.image_width if args.image_width > 0 else args.image_size * 2
        z1 = torch.randn(args.batch_size, 3, h, w, device=device)
        r = torch.zeros((args.batch_size,), device=device)
        t = torch.ones((args.batch_size,), device=device)
        u = model(z1, t * args.time_embed_scale, r * args.time_embed_scale)
        sample_wide = z1 - u

    # Keep CATVTON-style output semantics; save tryon half for visualization.
    sample_tryon = sample_wide[:, :, :, : args.image_size]
    sample_tryon = (sample_tryon.clamp(-1, 1) + 1) * 0.5
    save_image(sample_tryon, args.output, nrow=min(args.batch_size, 4))
    print(f"Saved samples to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Inference for custom DiT variants (datapred / meanflow)")
    p.add_argument("--approach", type=str, required=True, choices=["datapred", "meanflow"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--image_width", type=int, default=-1)
    p.add_argument("--patch_size", type=int, default=2)
    p.add_argument("--hidden_size", type=int, default=1280)
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--num_heads", type=int, default=20)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--diffusion_steps", type=int, default=-1,
                   help=">0 overrides schedule steps. <=0 uses checkpoint diffusion.steps for datapred.")
    p.add_argument("--time_embed_scale", type=float, default=1000.0)
    p.add_argument("--person_path", type=str, default=None)
    p.add_argument("--cloth_path", type=str, default=None)
    return p


if __name__ == "__main__":
    run_inference(build_parser().parse_args())

def _load_rgb(path: str, h: int, w: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(Image.open(path).convert("RGB"))
