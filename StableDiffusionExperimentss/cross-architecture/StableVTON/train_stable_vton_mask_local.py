"""Official StableVITON trainer wrapper (Masked version)."""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common import add_common_args, cleanup_dist, latest_checkpoint, setup_dist, wrap_ddp
from train_stable_vton_mask_local import StableCategoryMaskPoseDataset, _collate, _maybe_init_wandb, _to_wandb_image

# Dynamically resolve StableVITON path for both SLURM and Local Windows
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SLURM_VITON_DIR = "/iopsstor/scratch/cscs/dbartaula/StableVITON"
LOCAL_VITON_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "StableVITON"))

STABLE_VITON_DIR = SLURM_VITON_DIR if os.path.exists(SLURM_VITON_DIR) else LOCAL_VITON_DIR
if STABLE_VITON_DIR not in sys.path:
    sys.path.insert(0, STABLE_VITON_DIR)

from omegaconf import OmegaConf
from cldm.model import create_model


def train(args):
    dist_info = setup_dist()
    
    # Load official PyTorch Lightning config
    config_path = os.path.join(STABLE_VITON_DIR, "configs", "VITONHD.yaml")
    config = OmegaConf.load(config_path)
    
    # Enforce ATV loss logic if requested by user
    if hasattr(args, "use_atv_loss") and args.use_atv_loss:
        config.model.params.use_attn_mask = True
        config.model.params.unet_config.params.use_atv_loss = True
    
    model = create_model(config_path, config=config).to(dist_info.device)
    model.train()
    
    # Wrap underlying PyTorch model with DDP (ignoring Lightning wrappers)
    model = wrap_ddp(model, dist_info)

    ds = StableCategoryMaskPoseDataset(
        root_dir=args.curvton_data_path,
        category=args.category,
        gender=args.gender,
        size=args.image_size if args.image_size > 0 else 0,
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

    # Official optimizer configuration targets ControlNet and specific layers
    trainable_params = []
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            
    optimizer = AdamW(trainable_params, lr=args.lr)
    scaler = GradScaler(enabled=(dist_info.device.type == "cuda"))
    wb_run = _maybe_init_wandb(args, dist_info.is_main)

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    ckpt_to_load = args.resume if args.resume else latest_checkpoint(run_dir)
    step = 0
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        ckpt = torch.load(ckpt_to_load, map_location=dist_info.device)
        model.module.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
        step = int(ckpt.get("step", 0))

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
            
        for batch in loader:
            person = batch["person"].to(dist_info.device, non_blocking=True)
            cloth = batch["cloth"].to(dist_info.device, non_blocking=True)
            pose = batch["pose"].to(dist_info.device, non_blocking=True)
            mask = batch["mask"].to(dist_info.device, non_blocking=True)
            gt = batch["ground_truth"].to(dist_info.device, non_blocking=True)

            # Masked processing: Generate grey-filled agnostic image
            if mask.shape[-2:] != person.shape[-2:]:
                mask = F.interpolate(mask, size=person.shape[-2:], mode="nearest")
            grey_fill = torch.full_like(person, 0.5)
            agnostic = torch.where(mask > 0.5, grey_fill, person)

            official_batch = {
                "image": gt,
                "cloth": cloth,
                "agn": agnostic,
                "agn_mask": mask,
                "image_densepose": pose,
                "txt": [""] * person.shape[0],
                "gt_cloth_warped_mask": mask, # Passed for internal ATV loss computation
            }

            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(dist_info.device.type == "cuda")):
                # Extract pre-processed latents based on official config keys
                x, c = model.module.get_input(official_batch, model.module.first_stage_key, force_c_encode=True)
                
                # The official forward pass processes ControlNet + UNet internally
                loss, loss_dict = model.module(x, c)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1

            if dist_info.is_main and step % args.log_interval == 0:
                print(f"[step {step:>6}/{args.max_steps}] loss={loss.item():.6f}", flush=True)
                if wb_run is not None:
                    wb_run.log({"train/loss": float(loss.item()), "train/step": step}, step=step)

            if dist_info.is_main and step % args.image_log_interval == 0 and wb_run is not None:
                with torch.no_grad():
                    log_dict = model.module.log_images(
                        official_batch,
                        N=min(8, official_batch["image"].shape[0]),
                        sample=True,
                        unconditional_guidance_scale=5.0,
                        ddim_steps=30
                    )
                payload = {"train/step": step}
                if "samples_cfg_scale_5.00" in log_dict:
                    payload["images/pred_tryon"] = _to_wandb_image(log_dict["samples_cfg_scale_5.00"], f"Pred step {step}")
                if "input" in log_dict:
                    payload["images/gt_tryon"] = _to_wandb_image(log_dict["input"], f"GT step {step}")
                if "agn" in official_batch:
                    payload["images/person"] = _to_wandb_image(official_batch["agn"][:8].cpu(), f"Person step {step}")
                if "cloth" in official_batch:
                    payload["images/cloth"] = _to_wandb_image(official_batch["cloth"][:8].cpu(), f"Cloth step {step}")
                wb_run.log(payload, step=step)

            if dist_info.is_main and step % args.save_interval == 0:
                torch.save(
                    {
                        "step": step,
                        "state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                    },
                    os.path.join(run_dir, f"ckpt_step_{step}.pt"),
                )
            if step >= args.max_steps:
                break

    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official StableVITON Trainer (Masked)")
    add_common_args(parser)
    parser.add_argument("--use_atv_loss", action="store_true")
    args = parser.parse_args()
    train(args)
