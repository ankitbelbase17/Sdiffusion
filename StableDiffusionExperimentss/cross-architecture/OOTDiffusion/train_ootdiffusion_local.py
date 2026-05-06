"""Inference-faithful OOT training (unmasked): initial person + black mask."""

import argparse
import os
import re
import sys
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
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
_CAT_RE = re.compile(r"_(dresses|upper_body|lower_body|uncertain)\.png$", re.IGNORECASE)


class OOTUnmaskedDataset(Dataset):
    """root/gender/{cloth_image,initial_person_image,tryon_image}."""

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
        genders = ("female", "male") if gender == "all" else (gender,)
        for g in genders:
            self._collect(g)
        if not self.samples:
            raise RuntimeError(f"No samples found under {root_dir}")

    def _collect(self, gender: str):
        leaf = os.path.join(self.root_dir, gender)
        cloth_dir = os.path.join(leaf, "cloth_image")
        person_dir = os.path.join(leaf, "initial_person_image")
        tryon_dir = os.path.join(leaf, "tryon_image")
        for d in (cloth_dir, person_dir, tryon_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Missing directory: {d}")
        cloth_files = sorted([f for f in os.listdir(cloth_dir) if f.lower().endswith(".png")])
        person_stems = {os.path.splitext(f)[0] for f in os.listdir(person_dir) if f.lower().endswith(".png")}
        tryon_set = {f for f in os.listdir(tryon_dir) if f.lower().endswith(".png")}
        for fname in cloth_files:
            m_cat = _CAT_RE.search(fname)
            if m_cat is None:
                continue
            cat = m_cat.group(1).lower()
            if self.category != "all" and cat != self.category:
                continue
            if fname not in tryon_set:
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
                    os.path.join(tryon_dir, fname),
                    cat,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cloth_p, person_p, tryon_p, cat = self.samples[idx]
        cloth = self.img_tf(Image.open(cloth_p).convert("RGB"))
        person = self.img_tf(Image.open(person_p).convert("RGB"))
        target = self.img_tf(Image.open(tryon_p).convert("RGB"))
        return {"cloth": cloth, "person": person, "target": target, "category": cat}


def _collate(rows):
    return {
        "cloth": torch.stack([r["cloth"] for r in rows], dim=0),
        "person": torch.stack([r["person"] for r in rows], dim=0),
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


def _is_valid_diffusers_root(path: str) -> bool:
    required = [
        os.path.join(path, "vae", "config.json"),
        os.path.join(path, "scheduler", "scheduler_config.json"),
        os.path.join(path, "tokenizer"),
        os.path.join(path, "text_encoder"),
    ]
    return all(os.path.exists(p) for p in required)


def _find_diffusers_root(search_root: str) -> str | None:
    if not os.path.isdir(search_root):
        return None
    # Fast checks for common direct locations first.
    direct_candidates = [
        search_root,
        os.path.join(search_root, "ootd"),
        os.path.join(search_root, "stable-diffusion-v1-5"),
        os.path.join(search_root, "sd15"),
    ]
    for cand in direct_candidates:
        if _is_valid_diffusers_root(cand):
            return cand

    for root, _, _ in os.walk(search_root):
        if _is_valid_diffusers_root(root):
            return root
    return None


def _resolve_diffusers_root(preferred: str, fallback_search_root: str) -> str:
    if _is_valid_diffusers_root(preferred):
        return preferred
    found = _find_diffusers_root(fallback_search_root)
    if found is not None:
        return found
    raise FileNotFoundError(
        "Could not find a valid diffusers model root. "
        f"Tried: {preferred} and searched under: {fallback_search_root}. "
        "Expected subfolders/files: vae/config.json, scheduler/scheduler_config.json, tokenizer/, text_encoder/."
    )


def train(args):
    dist = setup_dist()
    device = dist.device

    default_ckpt_root = os.path.join(OOT_ROOT, "checkpoints")
    model_path = _resolve_diffusers_root(args.pretrained_model_path, default_ckpt_root)
    vae_path = model_path
    vit_path = args.clip_model_path
    unet_path = args.unet_checkpoint_path
    if dist.is_main:
        print(f"[paths] pretrained_model_path={model_path}", flush=True)
        print(f"[paths] clip_model_path={vit_path}", flush=True)
        print(f"[paths] unet_checkpoint_path={unet_path}", flush=True)

    dataset = OOTUnmaskedDataset(args.curvton_data_path, gender=args.gender, category=args.category)
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
    scaler = GradScaler(enabled=(device.type == "cuda"))
    run_dir = os.path.join(args.output_dir, args.run_name or "train_ootdiffusion")
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
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = int(ckpt.get("step", 0))
        if dist.is_main:
            print(f"Resumed from checkpoint: {ckpt_to_load} (step={step})", flush=True)
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            cloth = batch["cloth"].to(device, non_blocking=True)
            # Unmasked requirement: use initial person image directly.
            person = batch["person"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            bs = cloth.shape[0]

            with torch.no_grad():
                cloth_pil = [transforms.ToPILImage()(((img * 0.5) + 0.5).cpu().clamp(0, 1)) for img in cloth]
                prompt_image = auto_processor(images=cloth_pil, return_tensors="pt").to(device)
                prompt_img_emb = image_encoder(prompt_image.data["pixel_values"]).image_embeds.unsqueeze(1)
                prompt_embeds = text_encoder(_tokenize(tokenizer, [""] * bs, 2).to(device))[0]
                prompt_embeds[:, 1:] = prompt_img_emb

                garm_latents = vae.encode(cloth).latent_dist.mode()
                vton_latents = vae.encode(person).latent_dist.mode()
                target_latents = vae.encode(target).latent_dist.mode()

            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bs,), device=device, dtype=torch.long)
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

            with autocast(enabled=(device.type == "cuda")):
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1

            if dist.is_main and step % args.log_interval == 0:
                print(f"[step {step:>6}/{args.max_steps}] loss={loss.item():.6f}", flush=True)
            if dist.is_main and wb is not None:
                wb.log({"train/loss": float(loss.item()), "train/step": step}, step=step)

            if dist.is_main and step % image_log_interval == 0:
                with torch.no_grad():
                    sample_latents = torch.randn_like(vton_latents[: min(4, bs)])
                    sample_noise = sample_latents.clone()
                    scheduler.set_timesteps(args.num_inference_steps, device=device)
                    for i, t in enumerate(scheduler.timesteps):
                        t_batch = torch.full((sample_latents.shape[0],), int(t), device=device, dtype=torch.long)
                        _, sp = unet_garm(
                            garm_latents[: sample_latents.shape[0]],
                            0,
                            encoder_hidden_states=prompt_embeds[: sample_latents.shape[0]],
                            return_dict=False,
                        )
                        n_pred = unet_vton(
                            torch.cat([sample_latents, vton_latents[: sample_latents.shape[0]]], dim=1),
                            sp.copy(),
                            t_batch,
                            encoder_hidden_states=prompt_embeds[: sample_latents.shape[0]],
                            return_dict=False,
                        )[0]
                        sample_latents = scheduler.step(n_pred, t, sample_latents).prev_sample
                    pred = vae.decode(sample_latents).sample
                if wb is not None:
                    wb.log(
                        {
                            "train/step": step,
                            "images/pred_tryon": _to_wandb_image(pred, f"pred {step}"),
                            "images/gt_tryon": _to_wandb_image(target[: pred.shape[0]], f"gt {step}"),
                            "images/person": _to_wandb_image(person[: pred.shape[0]], f"person {step}"),
                            "images/cloth": _to_wandb_image(cloth[: pred.shape[0]], f"cloth {step}"),
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
                        "scaler_state_dict": scaler.state_dict(),
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
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            },
            os.path.join(run_dir, "ckpt_final.pt"),
        )
        if wb is not None:
            wb.finish()
    cleanup_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OOT unmasked trainer with inference-faithful forward")
    add_common_args(parser)
    parser.add_argument("--category", type=str, default="all", choices=["all", "dresses", "upper_body", "lower_body", "uncertain"])
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--resume", type=str, default=None, help="Optional explicit checkpoint path. If unset, latest checkpoint in run_dir is used.")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=os.path.join(OOT_ROOT, "checkpoints", "ootd"),
        help="Diffusers SD model root containing vae/scheduler/tokenizer/text_encoder.",
    )
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default=os.path.join(OOT_ROOT, "checkpoints", "clip-vit-large-patch14"),
        help="Path or HF id for CLIP vision encoder.",
    )
    parser.add_argument(
        "--unet_checkpoint_path",
        type=str,
        default=os.path.join(OOT_ROOT, "checkpoints", "ootd", "ootd_hd", "checkpoint-36000"),
        help="Path to OOT UNet checkpoint root containing unet_garm/ and unet_vton/.",
    )
    args = parser.parse_args()
    args.run_name = args.run_name or "train_ootdiffusion"
    train(args)
