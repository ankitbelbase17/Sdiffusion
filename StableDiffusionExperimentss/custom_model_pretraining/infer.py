import argparse
import os

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

from model import DiT250M, DiTConfig
from utils import make_beta_schedule, sample_ddim_like


def _load_cfg(ckpt_cfg: dict) -> DiTConfig:
    return DiTConfig(
        image_size=ckpt_cfg.get("image_size", 64),
        image_height=ckpt_cfg.get("image_height", ckpt_cfg.get("image_size", 64)),
        image_width=ckpt_cfg.get("image_width", ckpt_cfg.get("image_size", 64) * 2),
        in_channels=ckpt_cfg.get("in_channels", 3),
        cond_in_channels=ckpt_cfg.get("cond_in_channels", 3),
        out_channels=ckpt_cfg.get("out_channels", 3),
        patch_size=ckpt_cfg.get("patch_size", 2),
        hidden_size=ckpt_cfg.get("hidden_size", 1280),
        depth=ckpt_cfg.get("depth", 9),
        num_heads=ckpt_cfg.get("num_heads", 20),
        mlp_ratio=ckpt_cfg.get("mlp_ratio", 4.0),
    )


def _load_rgb(path: str, h: int, w: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(Image.open(path).convert("RGB"))


def _resolve_diffusion_steps(args: argparse.Namespace, ckpt: dict) -> int:
    if args.diffusion_steps > 0:
        return int(args.diffusion_steps)
    diff_cfg = ckpt.get("diffusion", {})
    if isinstance(diff_cfg, dict) and int(diff_cfg.get("steps", 0)) > 0:
        return int(diff_cfg["steps"])
    raise ValueError("Could not resolve diffusion_steps. Set --diffusion_steps > 0 or use a checkpoint with diffusion.steps metadata.")


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if not args.person_path or not args.cloth_path:
        raise ValueError("Datapred inference requires both --person_path and --cloth_path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = _load_cfg(ckpt.get("cfg", {}))
    model = DiT250M(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"], strict=True)

    diffusion_steps = _resolve_diffusion_steps(args, ckpt)
    betas = make_beta_schedule(diffusion_steps).to(device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alpha_bar)
    sqrt_1mab = torch.sqrt(1 - alpha_bar)

    person = _load_rgb(args.person_path, cfg.image_height, cfg.image_size).unsqueeze(0).to(device)
    cloth = _load_rgb(args.cloth_path, cfg.image_height, cfg.image_size).unsqueeze(0).to(device)
    cond = torch.cat([person, cloth], dim=3).repeat(args.batch_size, 1, 1, 1)
    sample = sample_ddim_like(
        model,
        shape=(args.batch_size, cfg.out_channels if cfg.out_channels is not None else 3, cfg.image_height, cfg.image_width),
        timesteps=diffusion_steps,
        sqrt_ab=sqrt_ab,
        sqrt_1mab=sqrt_1mab,
        device=device,
        cond=cond,
    )
    sample = (sample.clamp(-1, 1) + 1) * 0.5
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_image(sample, args.output, nrow=min(args.batch_size, 4))
    print(f"Saved samples to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Inference for custom 250M DiT x0 model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="results/custom_dit_samples.png")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--diffusion_steps", type=int, default=-1,
                   help=">0 overrides schedule steps. <=0 uses checkpoint diffusion.steps.")
    p.add_argument("--person_path", type=str, default=None)
    p.add_argument("--cloth_path", type=str, default=None)
    main(p.parse_args())

