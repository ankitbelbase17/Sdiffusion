"""Inference-faithful OOT training (masked): real mask-conditioned path."""

import argparse
import os
import re
import sys
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler

from common import add_common_args, cleanup_dist, latest_checkpoint, setup_dist

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OOT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "OOTDiffusion"))
if OOT_ROOT not in sys.path:
    sys.path.insert(0, OOT_ROOT)

from ootd.pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel  # noqa: E402
from ootd.pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel  # noqa: E402

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


_FC_MC_RE = re.compile(r"_(?:fc|mc)_")


class OOTMaskedDataset(Dataset):
    """root/category/gender/{cloth_image,initial_person_image,mask_image,tryon_image}."""

    def __init__(self, root_dir: str, gender: str = "all", category: str = "all"):
        self.root_dir = root_dir
        self.gender = gender
        self.category = category.lower()
        self.samples = []
        self.img_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.mask_tf = transforms.Compose([transforms.ToTensor()])
        categories = ("dresses", "upper_body", "lower_body", "uncertain") if category == "all" else (category,)
        genders = ("female", "male") if gender == "all" else (gender,)
        for cat in categories:
            for g in genders:
                self._collect(cat, g)
        if not self.samples:
            raise RuntimeError(f"No samples found under {root_dir}")

    def _collect(self, category: str, gender: str):
        leaf = os.path.join(self.root_dir, category, gender)
        cloth_dir = os.path.join(leaf, "cloth_image")
        person_dir = os.path.join(leaf, "initial_person_image")
        mask_dir = os.path.join(leaf, "mask_image")
        tryon_dir = os.path.join(leaf, "tryon_image")
        for d in (cloth_dir, person_dir, mask_dir, tryon_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing directory: {d}")
        cloth_files = sorted([f for f in os.listdir(cloth_dir) if f.lower().endswith(".png")])
        person_stems = {os.path.splitext(f)[0] for f in os.listdir(person_dir) if f.lower().endswith(".png")}
        mask_set = {f for f in os.listdir(mask_dir) if f.lower().endswith(".png")}
        tryon_set = {f for f in os.listdir(tryon_dir) if f.lower().endswith(".png")}
        for fname in cloth_files:
            if fname not in tryon_set or fname not in mask_set:
                continue
            stem = os.path.splitext(fname)[0]
            m = _FC_MC_RE.search(stem)
            if m is None:
                continue
            person_stem = stem[: m.start()]
            if person_stem not in person_stems:
                continue
            self.samples.append(
                (
                    os.path.join(cloth_dir, fname),
                    os.path.join(person_dir, person_stem + ".png"),
                    os.path.join(mask_dir, fname),
                    os.path.join(tryon_dir, fname),
                    category,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cloth_p, person_p, mask_p, tryon_p, cat = self.samples[idx]
        cloth = self.img_tf(Image.open(cloth_p).convert("RGB"))
        person = self.img_tf(Image.open(person_p).convert("RGB"))
        mask = (self.mask_tf(Image.open(mask_p).convert("L")) > 0.5).float()
        target = self.img_tf(Image.open(tryon_p).convert("RGB"))
        return {"cloth": cloth, "person": person, "mask": mask, "target": target, "category": cat}


def _collate(rows):
    return {
        "cloth": torch.stack([r["cloth"] for r in rows], dim=0),
        "person": torch.stack([r["person"] for r in rows], dim=0),
        "mask": torch.stack([r["mask"] for r in rows], dim=0),
        "target": torch.stack([r["target"] for r in rows], dim=0),
        "category": [r["category"] for r in rows],
    }


def _tokenize(tokenizer: CLIPTokenizer, captions: List[str], max_length: int):
    return tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids


def _to_wandb_image(batch: torch.Tensor, caption: str):
    if wandb is None:
        return None
    vis = (batch.detach().cpu().clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(vis, nrow=min(4, vis.shape[0]))
    return wandb.Image(grid, caption=caption)


def train(args):
    dist = setup_dist()
    device = dist.device

    vit_path = os.path.join(OOT_ROOT, "checkpoints", "clip-vit-large-patch14")
    vae_path = os.path.join(OOT_ROOT, "checkpoints", "ootd")
    model_path = os.path.join(OOT_ROOT, "checkpoints", "ootd")
    unet_path = os.path.join(OOT_ROOT, "checkpoints", "ootd", "ootd_hd", "checkpoint-36000")

    dataset = OOTMaskedDataset(args.curvton_data_path, gender=args.gender, category=args.category)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=dist.world_size, rank=dist.rank, shuffle=True, drop_last=True
    ) if dist.world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=_collate,
        persistent_workers=(args.num_workers > 0),
    )

    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    unet_garm = UNetGarm2DConditionModel.from_pretrained(unet_path, subfolder="unet_garm").to(device)
    unet_vton = UNetVton2DConditionModel.from_pretrained(unet_path, subfolder="unet_vton").to(device)
    auto_processor = AutoProcessor.from_pretrained(vit_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(vit_path).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)

    vae.eval().requires_grad_(False)
    image_encoder.eval().requires_grad_(False)
    text_encoder.eval().requires_grad_(False)
    unet_garm.train()
    unet_vton.train()

    if dist.world_size > 1:
        if any(p.requires_grad for p in unet_garm.parameters()):
            unet_garm = torch.nn.parallel.DistributedDataParallel(
                unet_garm, device_ids=[dist.local_rank], output_device=dist.local_rank
            )
        if any(p.requires_grad for p in unet_vton.parameters()):
            unet_vton = torch.nn.parallel.DistributedDataParallel(
                unet_vton, device_ids=[dist.local_rank], output_device=dist.local_rank
            )

    optimizer = AdamW(list(unet_garm.parameters()) + list(unet_vton.parameters()), lr=args.lr)
    run_dir = os.path.join(args.output_dir, args.run_name or "train_ootdiffusion_mask")
    os.makedirs(run_dir, exist_ok=True)
    image_dir = os.path.join(run_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    wb = None
    if dist.is_main and not args.disable_wandb and wandb is not None:
        try:
            wb = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except Exception:
            wb = None

    save_interval = 1000
    image_log_interval = 250

    ckpt_to_load = args.resume if args.resume else latest_checkpoint(run_dir)
    step = 0
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        ckpt = torch.load(ckpt_to_load, map_location=device)
        raw_g = unet_garm.module if hasattr(unet_garm, "module") else unet_garm
        raw_v = unet_vton.module if hasattr(unet_vton, "module") else unet_vton
        if "unet_garm_state_dict" in ckpt:
            raw_g.load_state_dict(ckpt["unet_garm_state_dict"], strict=False)
        if "unet_vton_state_dict" in ckpt:
            raw_v.load_state_dict(ckpt["unet_vton_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))
        if dist.is_main:
            print(f"Resumed from checkpoint: {ckpt_to_load} (step={step})", flush=True)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            cloth = batch["cloth"].to(device, non_blocking=True)
            person = batch["person"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            bs = cloth.shape[0]

            # Masked requirement: masked person image for vton latent path.
            if mask.shape[-2:] != person.shape[-2:]:
                mask = F.interpolate(mask, size=person.shape[-2:], mode="nearest")
            gray = torch.zeros_like(person)
            person_masked = torch.where(mask > 0.5, gray, person)

            with torch.no_grad():
                cloth_pil = [transforms.ToPILImage()(((img * 0.5) + 0.5).cpu().clamp(0, 1)) for img in cloth]
                prompt_image = auto_processor(images=cloth_pil, return_tensors="pt").to(device)
                prompt_img_emb = image_encoder(prompt_image.data["pixel_values"]).image_embeds.unsqueeze(1)
                prompt_embeds = text_encoder(_tokenize(tokenizer, [""] * bs, 2).to(device))[0]
                prompt_embeds[:, 1:] = prompt_img_emb

                garm_latents = vae.encode(cloth).latent_dist.mode()
                vton_latents = vae.encode(person_masked).latent_dist.mode()
                target_latents = vae.encode(target).latent_dist.mode()

            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device, dtype=torch.long)
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

            _, spatial_attn_outputs = unet_garm(
                garm_latents,
                0,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            noise_pred = unet_vton(
                torch.cat([noisy_latents, vton_latents], dim=1),
                spatial_attn_outputs.copy(),
                timesteps,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # Single training objective: noise MSE.
            loss = F.mse_loss(noise_pred.float(), noise.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1

            if dist.is_main and step % args.log_interval == 0:
                print(f"[step {step:>6}/{args.max_steps}] loss={loss.item():.6f}", flush=True)
            if dist.is_main and wb is not None:
                wb.log({"train/loss": float(loss.item()), "train/step": step}, step=step)

            if dist.is_main and step % image_log_interval == 0:
                with torch.no_grad():
                    k = min(4, bs)
                    sample_latents = torch.randn_like(vton_latents[:k])
                    sample_noise = sample_latents.clone()
                    scheduler.set_timesteps(args.num_inference_steps, device=device)
                    mask_lat = F.interpolate(mask[:k], size=vton_latents.shape[-2:]).to(device=device, dtype=vton_latents.dtype)
                    image_ori_lat = vae.encode(person[:k]).latent_dist.mode()
                    for i, t in enumerate(scheduler.timesteps):
                        t_batch = torch.full((k,), int(t), device=device, dtype=torch.long)
                        _, sp = unet_garm(
                            garm_latents[:k],
                            0,
                            encoder_hidden_states=prompt_embeds[:k],
                            return_dict=False,
                        )
                        n_pred = unet_vton(
                            torch.cat([sample_latents, vton_latents[:k]], dim=1),
                            sp.copy(),
                            t_batch,
                            encoder_hidden_states=prompt_embeds[:k],
                            return_dict=False,
                        )[0]
                        sample_latents = scheduler.step(n_pred, t, sample_latents).prev_sample
                        if i < len(scheduler.timesteps) - 1:
                            nt = scheduler.timesteps[i + 1]
                            init_lat = scheduler.add_noise(image_ori_lat, sample_noise, torch.tensor([nt], device=device))
                        else:
                            init_lat = image_ori_lat
                        sample_latents = (1.0 - mask_lat) * init_lat + mask_lat * sample_latents
                    pred = vae.decode(sample_latents).sample
                if wb is not None:
                    wb.log(
                        {
                            "train/step": step,
                            "images/pred_tryon": _to_wandb_image(pred, f"pred {step}"),
                            "images/gt_tryon": _to_wandb_image(target[:k], f"gt {step}"),
                            "images/person": _to_wandb_image(person_masked[:k], f"person {step}"),
                            "images/cloth": _to_wandb_image(cloth[:k], f"cloth {step}"),
                            "images/mask": _to_wandb_image(mask[:k].repeat(1, 3, 1, 1) * 2 - 1, f"mask {step}"),
                        },
                        step=step,
                    )

            if dist.is_main and step % save_interval == 0:
                raw_g = unet_garm.module if hasattr(unet_garm, "module") else unet_garm
                raw_v = unet_vton.module if hasattr(unet_vton, "module") else unet_vton
                torch.save(
                    {
                        "step": step,
                        "unet_garm_state_dict": raw_g.state_dict(),
                        "unet_vton_state_dict": raw_v.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                    },
                    os.path.join(run_dir, f"ckpt_step_{step}.pt"),
                )
            if step >= args.max_steps:
                break

    if dist.is_main:
        raw_g = unet_garm.module if hasattr(unet_garm, "module") else unet_garm
        raw_v = unet_vton.module if hasattr(unet_vton, "module") else unet_vton
        torch.save(
            {
                "step": step,
                "unet_garm_state_dict": raw_g.state_dict(),
                "unet_vton_state_dict": raw_v.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            os.path.join(run_dir, "ckpt_final.pt"),
        )
        if wb is not None:
            wb.finish()
    cleanup_dist()


@record
def _main():
    parser = argparse.ArgumentParser(description="OOT masked trainer with inference-faithful forward")
    add_common_args(parser)
    parser.add_argument("--category", type=str, default="all", choices=["all", "dresses", "upper_body", "lower_body", "uncertain"])
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--resume", type=str, default=None, help="Optional explicit checkpoint path. If unset, latest checkpoint in run_dir is used.")
    args = parser.parse_args()
    args.run_name = args.run_name or "train_ootdiffusion_mask"
    train(args)


if __name__ == "__main__":
    _main()
