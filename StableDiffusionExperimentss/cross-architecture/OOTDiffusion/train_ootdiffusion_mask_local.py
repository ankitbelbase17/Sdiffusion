"""OOTDiffusion trainer with real mask conditioning (pose disabled) from stratified-category dataset."""

import argparse
import os
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid

from common import add_common_args, cleanup_dist, latest_checkpoint, setup_dist, wrap_ddp
from utils import _local_load_image

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


_FC_MC_RE = re.compile(r"_(?:fc|mc)_")


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


class OOTCategoryMaskPoseDataset(Dataset):
    """root/category/gender/{cloth,initial_person,mask,pose,tryon}."""

    def __init__(self, root_dir: str, category: str = "all", gender: str = "all", size: int = 0):
        self.root_dir = root_dir
        self.category = category
        self.gender = gender
        self.size = size
        self.samples = []

        if size and size > 0:
            self.img_tf = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.mask_tf = transforms.Compose([
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        else:
            self.img_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.mask_tf = transforms.Compose([transforms.ToTensor()])

        categories = ("dresses", "upper_body", "lower_body", "uncertain") if category == "all" else (category,)
        genders = ("female", "male") if gender == "all" else (gender,)

        for cat in categories:
            for g in genders:
                self._collect(cat, g)

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {root_dir} for category={category}, gender={gender}")
        print(f"[OOTMaskPoseDataset] Loaded {len(self.samples)} samples")

    def _collect(self, category: str, gender: str):
        leaf = os.path.join(self.root_dir, category, gender)
        cloth_dir = os.path.join(leaf, "cloth_image")
        person_dir = os.path.join(leaf, "initial_person_image")
        mask_dir = os.path.join(leaf, "mask_image")
        pose_dir = os.path.join(leaf, "pose_image")
        tryon_dir = os.path.join(leaf, "tryon_image")
        for d in (cloth_dir, person_dir, mask_dir, pose_dir, tryon_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing directory: {d}")

        cloth_files = sorted([f for f in os.listdir(cloth_dir) if f.lower().endswith(".png")])
        person_stems = {os.path.splitext(f)[0] for f in os.listdir(person_dir) if f.lower().endswith(".png")}
        mask_set = set([f for f in os.listdir(mask_dir) if f.lower().endswith(".png")])
        pose_set = set([f for f in os.listdir(pose_dir) if f.lower().endswith(".png")])
        tryon_set = set([f for f in os.listdir(tryon_dir) if f.lower().endswith(".png")])

        for fname in cloth_files:
            if fname not in tryon_set or fname not in mask_set or fname not in pose_set:
                continue
            stem = os.path.splitext(fname)[0]
            m = _FC_MC_RE.search(stem)
            if m is None:
                continue
            person_stem = stem[:m.start()]
            if person_stem not in person_stems:
                continue
            self.samples.append((
                os.path.join(person_dir, person_stem + ".png"),
                os.path.join(cloth_dir, fname),
                os.path.join(mask_dir, fname),
                os.path.join(pose_dir, fname),
                os.path.join(tryon_dir, fname),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        person_p, cloth_p, mask_p, pose_p, tryon_p = self.samples[idx]
        person = self.img_tf(_local_load_image(person_p))
        cloth = self.img_tf(_local_load_image(cloth_p))
        pose = self.img_tf(_local_load_image(pose_p))
        tryon = self.img_tf(_local_load_image(tryon_p))
        mask = self.mask_tf(_local_load_image(mask_p).convert("L"))
        mask = (mask > 0.5).float()
        return {
            "person": person,
            "cloth": cloth,
            "pose": pose,
            "mask": mask,
            "ground_truth": tryon,
        }


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


class OOTAttentionProcessor:
    def __init__(self):
        self.store = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        is_cross_attention = encoder_hidden_states is not None
        
        if getattr(attn, "is_outfitting", False):
            if not is_cross_attention: 
                self.store[attn._oot_name] = hidden_states
            return self._forward(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            
        if not getattr(attn, "is_outfitting", False) and not is_cross_attention:
            g_n = self.store.get(attn._oot_name)
            if g_n is not None:
                orig_len = hidden_states.shape[1]
                joined_hidden_states = torch.cat([hidden_states, g_n], dim=1)
                
                joined_mask = attention_mask
                if attention_mask is not None:
                    joined_mask = torch.cat([attention_mask, attention_mask], dim=-1)
                
                out = self._forward(attn, joined_hidden_states, encoder_hidden_states, joined_mask, **kwargs)
                return out[:, :orig_len]
                
        return self._forward(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

    def _forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape to (batch, heads, seq_len, head_dim) for flash attention
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Use flash attention / memory-efficient attention (O(n) memory, not O(n²))
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class OOTDiffusionModel(nn.Module):
    def __init__(self, model_name, outfitting_dropout=0.1):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.denoising_unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.outfitting_unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.vae.requires_grad_(False)
        self.outfitting_dropout = outfitting_dropout
        
        orig_conv_in = self.denoising_unet.conv_in
        self.denoising_unet.conv_in = nn.Conv2d(8, orig_conv_in.out_channels, orig_conv_in.kernel_size, orig_conv_in.stride, orig_conv_in.padding)
        with torch.no_grad():
            self.denoising_unet.conv_in.weight[:, :4] = orig_conv_in.weight
            self.denoising_unet.conv_in.weight[:, 4:] = torch.zeros_like(orig_conv_in.weight)
            self.denoising_unet.conv_in.bias = orig_conv_in.bias
            
        self.processor = OOTAttentionProcessor()
        self._setup_attention(self.denoising_unet, is_outfitting=False)
        self._setup_attention(self.outfitting_unet, is_outfitting=True)

        # Cache cross_attention_dim as a plain int so it survives DDP wrapping
        _dim = self.denoising_unet.config.cross_attention_dim
        self._cross_attention_dim = int(_dim[0] if isinstance(_dim, (tuple, list)) else _dim)

    @property
    def cross_attention_dim(self):
        return self._cross_attention_dim
        
    def _setup_attention(self, unet, is_outfitting):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            attn_procs[name] = self.processor
        unet.set_attn_processor(attn_procs)
        
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "Attention":
                module.is_outfitting = is_outfitting
                module._oot_name = name

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def empty_text(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 77, self.cross_attention_dim, device=device, dtype=dtype)

    def forward(self, noisy_target, person_lat, cloth_lat, timesteps):
        text = self.empty_text(noisy_target.shape[0], noisy_target.device, noisy_target.dtype)

        self.outfitting_unet(cloth_lat, timesteps, text)
        model_input = torch.cat([noisy_target, person_lat], dim=1)
        return self.denoising_unet(model_input, timesteps, text).sample


def train(args):
    dist_info = setup_dist()
    model = OOTDiffusionModel(args.model_name, args.outfitting_dropout).to(dist_info.device)
    # Enable gradient checkpointing to reduce activation memory (essential with 2 UNets).
    model.denoising_unet.enable_gradient_checkpointing()
    model.outfitting_unet.enable_gradient_checkpointing()
    model.denoising_unet = wrap_ddp(model.denoising_unet, dist_info)
    # NOTE: outfitting_unet is NOT DDP-wrapped because its output is discarded
    # (only intermediate attention features in processor.store are used).
    # DDP can't trace gradients through a Python dict, so we manually all_reduce
    # outfitting_unet gradients after backward instead.

    ds = OOTCategoryMaskPoseDataset(
        root_dir=args.curvton_data_path,
        category=args.category,
        gender=args.gender,
        size=0,  # full-resolution
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        ds, num_replicas=dist_info.world_size, rank=dist_info.rank, shuffle=True, drop_last=True
    ) if dist_info.world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    params = list(model.denoising_unet.parameters()) + list(model.outfitting_unet.parameters())
    optimizer = AdamW(params, lr=args.lr)
    scaler = GradScaler(enabled=(dist_info.device.type == "cuda"))
    wb_run = _maybe_init_wandb(args, dist_info.is_main)

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_to_load = args.resume

    step = 0
    if ckpt_to_load:
        ckpt = torch.load(ckpt_to_load, map_location=dist_info.device)
        model.denoising_unet.module.load_state_dict(ckpt["denoising_unet_state_dict"]) if hasattr(model.denoising_unet, "module") else model.denoising_unet.load_state_dict(ckpt["denoising_unet_state_dict"])
        model.outfitting_unet.load_state_dict(ckpt["outfitting_unet_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))

    @torch.no_grad()
    def _sample_tryon(person_lat, cloth_lat, n_steps, mask_lat=None, image_ori_lat=None):
        latents = torch.randn_like(person_lat)
        noise = latents.clone()
        model.scheduler.set_timesteps(n_steps, device=latents.device)
        text = model.empty_text(1, latents.device, latents.dtype)

        timesteps = model.scheduler.timesteps
        for i, t in enumerate(timesteps):
            t_batch = torch.full((latents.shape[0],), int(t), device=latents.device, dtype=torch.long)
            noise_pred = model(latents, person_lat, cloth_lat, t_batch)
            latents = model.scheduler.step(noise_pred, t, latents).prev_sample
            if mask_lat is not None and image_ori_lat is not None:
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents = model.scheduler.add_noise(
                        image_ori_lat,
                        noise,
                        torch.tensor([noise_timestep], device=latents.device),
                    )
                else:
                    init_latents = image_ori_lat
                latents = (1.0 - mask_lat) * init_latents + mask_lat * latents
        return model.vae.decode(latents / model.vae.config.scaling_factor).sample

    optimizer.zero_grad(set_to_none=True)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            person = batch["person"].to(dist_info.device, non_blocking=True)
            cloth = batch["cloth"].to(dist_info.device, non_blocking=True)
            mask = batch["mask"].to(dist_info.device, non_blocking=True)
            gt = batch["ground_truth"].to(dist_info.device, non_blocking=True)

            # Step 1: Apply mask — zero out masked region
            # Step 2: Fill masked region with mid-grey (0.0 in [-1, 1] ≈ 127/255)
            grey_fill = torch.zeros_like(person)
            masked_person = torch.where(mask > 0.5, grey_fill, person)
            person_for_model = masked_person

            with torch.no_grad():
                target_lat = model.encode(gt)
                person_lat = model.encode(person_for_model)
                cloth_lat = model.encode(cloth)
                image_ori_lat = model.encode(person)
                mask_lat = F.interpolate(mask, size=person_lat.shape[-2:]).to(
                    device=person_lat.device, dtype=person_lat.dtype
                )

            noise = torch.randn_like(target_lat)
            timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (target_lat.shape[0],), device=target_lat.device).long()
            noisy = model.scheduler.add_noise(target_lat, noise, timesteps)
            with autocast(device_type=dist_info.device.type, enabled=(dist_info.device.type == "cuda")):
                pred = model(noisy, person_lat, cloth_lat, timesteps)
                loss = F.mse_loss(pred.float(), noise.float())

            scaler.scale(loss).backward()
            # Manually sync outfitting_unet gradients (not DDP-wrapped).
            if dist_info.world_size > 1:
                for p in model.outfitting_unet.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if dist_info.is_main and step % args.log_interval == 0:
                print(f"[step {step:>6}/{args.max_steps}] loss={loss.item():.6f}", flush=True)
            if dist_info.is_main and wb_run is not None:
                wb_run.log({"train/loss": float(loss.item()), "train/step": step}, step=step)
            if dist_info.is_main and wb_run is not None and step % args.image_log_interval == 0:
                with torch.no_grad():
                    k = min(8, person_lat.shape[0])
                    pred_img = _sample_tryon(
                        person_lat[:k],
                        cloth_lat[:k],
                        args.num_inference_steps,
                        mask_lat=mask_lat[:k],
                        image_ori_lat=image_ori_lat[:k],
                    )
                payload = {
                    "train/step": step,
                    "images/pred_tryon": _to_wandb_image(pred_img[:8].detach().cpu(), f"OOT pred step {step}"),
                    "images/gt_tryon": _to_wandb_image(gt[:8].detach().cpu(), f"OOT gt step {step}"),
                    "images/person": _to_wandb_image(person_for_model[:8].detach().cpu(), f"OOT person step {step}"),
                    "images/cloth": _to_wandb_image(cloth[:8].detach().cpu(), f"OOT cloth step {step}"),
                    "images/mask": _to_wandb_image(mask[:8].repeat(1, 3, 1, 1).detach().cpu() * 2 - 1, f"OOT mask step {step}"),
                }
                wb_run.log({k: v for k, v in payload.items() if v is not None}, step=step)

            if dist_info.is_main and step % args.save_interval == 0:
                torch.save(
                    {
                        "step": step,
                        "architecture": "OOTDiffusion + real mask (pose disabled)",
                        "denoising_unet_state_dict": model.denoising_unet.module.state_dict() if hasattr(model.denoising_unet, "module") else model.denoising_unet.state_dict(),
                        "outfitting_unet_state_dict": model.outfitting_unet.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "args": vars(args),
                    },
                    os.path.join(run_dir, f"ckpt_step_{step}.pt"),
                )
            if step >= args.max_steps:
                break

    if dist_info.is_main:
        torch.save(
            {
                "step": step,
                "architecture": "OOTDiffusion + real mask (pose disabled)",
                "denoising_unet_state_dict": model.denoising_unet.module.state_dict() if hasattr(model.denoising_unet, "module") else model.denoising_unet.state_dict(),
                "outfitting_unet_state_dict": model.outfitting_unet.module.state_dict() if hasattr(model.outfitting_unet, "module") else model.outfitting_unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            },
            os.path.join(run_dir, "ckpt_final.pt"),
        )
        if wb_run is not None:
            wb_run.finish()
    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOTDiffusion trainer with real mask (pose disabled)")
    add_common_args(parser)
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--outfitting_dropout", type=float, default=0.1)
    parser.add_argument("--category", type=str, default="all", choices=["all", "dresses", "upper_body", "lower_body", "uncertain"])
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    args.run_name = args.run_name or "train_ootdiffusion_mask"
    args.image_size = 0
    train(args)
