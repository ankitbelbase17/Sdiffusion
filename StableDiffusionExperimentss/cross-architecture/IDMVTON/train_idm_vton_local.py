"""IDM-VTON architecture-faithful local trainer for CurvTON.

This keeps an IDM-VTON-style component graph local:
- SDXL inpainting UNet modified to 13 input channels.
- Garment reference UNet from SDXL base.
- CLIP vision encoder plus IP-Adapter-style Resampler tokens.

Dataset adaptation:
- CurvTON has initial_person_image instead of agnostic-mask/densepose.
- The official 1-channel mask slot is filled from initial-person luminance.
- The official masked-person and pose latent slots both use VAE(initial_person).
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torchvision.utils import make_grid
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
)

from common import (
    add_common_args,
    batch_images,
    build_curvton_loader,
    cleanup_dist,
    latest_checkpoint,
    setup_dist,
    wrap_ddp,
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def raw_module(module):
    return module.module if hasattr(module, "module") else module


def _maybe_init_wandb(args, is_main):
    if not is_main or args.disable_wandb or os.environ.get("DISABLE_WANDB", "0") == "1":
        return None
    if wandb is None:
        return None
    try:
        return wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    except Exception:
        return None


def _to_wandb_image(batch: torch.Tensor, caption: str):
    if wandb is None:
        return None
    vis = (batch.clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(vis, nrow=min(4, vis.shape[0]))
    return wandb.Image(grid, caption=caption)


class Resampler(nn.Module):
    """IP-Adapter-style latent query resampler used by official IDM-VTON."""

    def __init__(self, dim, depth, dim_head, heads, num_queries, embedding_dim, output_dim, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim),
                ),
            ]))

    def forward(self, image_tokens):
        x = self.proj_in(image_tokens)
        latents = self.latents.repeat(x.shape[0], 1, 1)
        for attn, ff in self.layers:
            latents = latents + attn(latents, x, x, need_weights=False)[0]
            latents = latents + ff(latents)
        return self.norm_out(self.proj_out(latents))


class IDMVTONModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
            rescale_betas_zero_snr=True,
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
        self.unet_encoder = UNet2DConditionModel.from_pretrained(args.pretrained_garmentnet_path, subfolder="unet")
        self.unet_encoder.config.addition_embed_type = None
        self.unet_encoder.config["addition_embed_type"] = None
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            low_cpu_mem_usage=False,
            device_map=None,
        )

        self.unet.config.encoder_hid_dim = self.image_encoder.config.hidden_size
        self.unet.config.encoder_hid_dim_type = "ip_image_proj"
        self.unet.config["encoder_hid_dim"] = self.image_encoder.config.hidden_size
        self.unet.config["encoder_hid_dim_type"] = "ip_image_proj"
        self.image_proj_model = Resampler(
            dim=self.image_encoder.config.hidden_size,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=args.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.cross_attention_dim,
            ff_mult=4,
        )
        self.unet.encoder_hid_proj = self.image_proj_model

        self._replace_unet_input_conv()
        self.clip_processor = CLIPImageProcessor()

        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet_encoder.requires_grad_(False)

    @property
    def cross_attention_dim(self):
        dim = self.unet.config.cross_attention_dim
        return int(dim[0] if isinstance(dim, (tuple, list)) else dim)

    def _replace_unet_input_conv(self):
        old_conv = self.unet.conv_in
        new_conv = nn.Conv2d(
            13,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            nn.init.kaiming_normal_(new_conv.weight)
            new_conv.weight.mul_(0.0)
            copy_ch = min(old_conv.weight.shape[1], 9)
            new_conv.weight[:, :copy_ch].copy_(old_conv.weight[:, :copy_ch])
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.unet.conv_in = new_conv
        self.unet.config["in_channels"] = 13
        self.unet.config.in_channels = 13

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def prompt_embeds(self, batch_size, device, dtype):
        seq_len = 77
        hidden = torch.zeros(
            batch_size, seq_len, self.cross_attention_dim, device=device, dtype=dtype
        )
        # SDXL UNet expects pooled text vector for added_cond_kwargs["text_embeds"].
        proj_in = int(getattr(self.unet.config, "projection_class_embeddings_input_dim", 2816))
        pooled_dim = max(1, proj_in - 6 * 256)
        pooled = torch.zeros(batch_size, pooled_dim, device=device, dtype=dtype)
        return hidden, pooled

    def add_time_ids(self, batch_size, height, width, device):
        ids = torch.tensor([height, width, 0, 0, height, width], device=device)
        return ids.unsqueeze(0).repeat(batch_size, 1)

    def forward(self, noisy, person_mask, person_lat, pose_lat, cloth, cloth_lat, timesteps, captions=None, cloth_captions=None):
        del captions, cloth_captions
        encoder_hidden_states, pooled = self.prompt_embeds(noisy.shape[0], noisy.device, noisy.dtype)
        cloth_hidden, _ = self.prompt_embeds(noisy.shape[0], noisy.device, noisy.dtype)
        add_time_ids = self.add_time_ids(noisy.shape[0], noisy.shape[-2] * 8, noisy.shape[-1] * 8, noisy.device)

        # Official IDM gets IP tokens from CLIPVision cloth image embeddings.
        cloth_01 = (cloth.clamp(-1, 1) + 1) / 2
        clip_pixels = self.clip_processor(images=list(cloth_01.detach().cpu()), return_tensors="pt").pixel_values
        clip_pixels = clip_pixels.to(noisy.device, dtype=next(self.image_encoder.parameters()).dtype)
        image_tokens = self.image_encoder(clip_pixels, output_hidden_states=True).hidden_states[-2]
        ip_tokens = self.image_proj_model(image_tokens.to(dtype=noisy.dtype))

        # Official garment UNet returns multi-level reference features. Diffusers' base
        # UNet exposes sample output, so we use its forward pass as the local garment branch.
        garment_residual = self.unet_encoder(cloth_lat, timesteps, cloth_hidden).sample
        model_in = torch.cat([noisy + 0.05 * garment_residual, person_mask, person_lat, pose_lat], dim=1)
        added = {
            "text_embeds": pooled,
            "time_ids": add_time_ids,
            "image_embeds": ip_tokens,
        }
        return self.unet(model_in, timesteps, encoder_hidden_states, added_cond_kwargs=added).sample


def train(args):
    dist_info = setup_dist()
    model = IDMVTONModel(args).to(dist_info.device)
    model.unet = wrap_ddp(model.unet, dist_info)
    loader, sampler = build_curvton_loader(args, dist_info)
    optimizer = AdamW(model.unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(dist_info.device.type == "cuda"))
    wb_run = _maybe_init_wandb(args, dist_info.is_main)

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_to_load = args.resume
    if ckpt_to_load is None and not args.no_resume:
        ckpt_to_load = latest_checkpoint(run_dir)

    step = 0
    if ckpt_to_load:
        ckpt = torch.load(ckpt_to_load, map_location=dist_info.device)
        raw_module(model.unet).load_state_dict(ckpt["unet_state_dict"])
        if "image_proj_state_dict" in ckpt:
            model.image_proj_model.load_state_dict(ckpt["image_proj_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            person, cloth, gt = batch_images(batch, dist_info.device)
            with torch.no_grad():
                target_lat = model.encode(gt)
                person_lat = model.encode(person)
                pose_lat = model.encode(person)
                cloth_lat = model.encode(cloth)
                person_mask = torch.zeros(
                    person.shape[0], 1, person.shape[2], person.shape[3],
                    device=person.device, dtype=person.dtype
                )
                person_mask = F.interpolate(person_mask, size=target_lat.shape[-2:], mode="bilinear", align_corners=False)

            noise = torch.randn_like(target_lat)
            timesteps = torch.randint(
                0,
                model.scheduler.config.num_train_timesteps,
                (target_lat.shape[0],),
                device=target_lat.device,
            ).long()
            noisy = model.scheduler.add_noise(target_lat, noise, timesteps)
            with autocast(enabled=(dist_info.device.type == "cuda")):
                pred = model(noisy, person_mask, person_lat, pose_lat, cloth, cloth_lat, timesteps)
                target = noise if model.scheduler.config.prediction_type == "epsilon" else model.scheduler.get_velocity(target_lat, noise, timesteps)
                loss = F.mse_loss(pred.float(), target.float())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if dist_info.is_main and step % args.log_interval == 0:
                print(f"[step {step:>6}/{args.max_steps}] loss={loss.item():.6f}", flush=True)
            if dist_info.is_main and wb_run is not None:
                wb_run.log({"train/loss": float(loss.item()), "train/step": step}, step=step)
            if dist_info.is_main and wb_run is not None and step % args.image_log_interval == 0:
                with torch.no_grad():
                    pred_x0 = model.scheduler.step(pred, timesteps, noisy).pred_original_sample
                    pred_img = model.vae.decode(pred_x0 / model.vae.config.scaling_factor).sample
                payload = {
                    "train/step": step,
                    "images/pred_tryon": _to_wandb_image(pred_img[:8].detach().cpu(), f"IDMVTON pred step {step}"),
                    "images/gt_tryon": _to_wandb_image(gt[:8].detach().cpu(), f"IDMVTON gt step {step}"),
                    "images/person": _to_wandb_image(person[:8].detach().cpu(), f"IDMVTON person step {step}"),
                    "images/cloth": _to_wandb_image(cloth[:8].detach().cpu(), f"IDMVTON cloth step {step}"),
                }
                wb_run.log({k: v for k, v in payload.items() if v is not None}, step=step)
            if dist_info.is_main and step % args.save_interval == 0:
                os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
                torch.save({
                    "step": step,
                    "architecture": "IDM-VTON SDXL inpaint + garment UNet + IP adapter tokens",
                    "unet_state_dict": raw_module(model.unet).state_dict(),
                    "image_proj_state_dict": model.image_proj_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "args": vars(args),
                }, os.path.join(args.output_dir, args.run_name, f"ckpt_step_{step}.pt"))
            if step >= args.max_steps:
                break

    if dist_info.is_main:
        os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
        torch.save({
            "step": step,
            "architecture": "IDM-VTON SDXL inpaint + garment UNet + IP adapter tokens",
            "unet_state_dict": raw_module(model.unet).state_dict(),
            "image_proj_state_dict": model.image_proj_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "args": vars(args),
        }, os.path.join(args.output_dir, args.run_name, "ckpt_final.pt"))
        if wb_run is not None:
            wb_run.finish()
    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local IDM-VTON architecture trainer")
    add_common_args(parser)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    parser.add_argument("--pretrained_garmentnet_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--image_encoder_path", type=str, default="ckpt/image_encoder")
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no_resume", action="store_true", default=False)
    args = parser.parse_args()
    args.run_name = args.run_name or "train_idm_vton"
    train(args)
