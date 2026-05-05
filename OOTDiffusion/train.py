import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler

from ootd.pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from ootd.pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel


@dataclass
class TrainBatch:
    cloth: torch.Tensor
    vton_masked: torch.Tensor
    image_ori: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    category: List[str]


class OOTTrainDataset(Dataset):
    """
    Expected tree:
      root/
        category/
          gender/
            cloth_image/
            initial_person_image/
            tryon_image/
            mask_image/   (optional; falls back to black mask)
    """

    def __init__(self, root_dir: str, image_size: Tuple[int, int] = (1024, 768)):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.img_tf = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )
        self._collect()
        if not self.samples:
            raise RuntimeError(f"No samples found under {root_dir}")

    def _collect(self):
        for cat_dir in self.root_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            category = cat_dir.name
            for gender_dir in cat_dir.iterdir():
                if not gender_dir.is_dir():
                    continue
                cloth_dir = gender_dir / "cloth_image"
                person_dir = gender_dir / "initial_person_image"
                tryon_dir = gender_dir / "tryon_image"
                mask_dir = gender_dir / "mask_image"
                if not (cloth_dir.exists() and person_dir.exists() and tryon_dir.exists()):
                    continue
                cloth_files = sorted([p for p in cloth_dir.glob("*.png")])
                person_stems = {p.stem for p in person_dir.glob("*.png")}
                tryon_files = {p.name for p in tryon_dir.glob("*.png")}
                mask_files = {p.name for p in mask_dir.glob("*.png")} if mask_dir.exists() else set()
                for cloth_path in cloth_files:
                    name = cloth_path.name
                    if name not in tryon_files:
                        continue
                    stem = cloth_path.stem
                    pivot = stem.find("_fc_")
                    if pivot < 0:
                        pivot = stem.find("_mc_")
                    if pivot < 0:
                        continue
                    person_stem = stem[:pivot]
                    if person_stem not in person_stems:
                        continue
                    self.samples.append(
                        (
                            str(cloth_path),
                            str(person_dir / f"{person_stem}.png"),
                            str(tryon_dir / name),
                            str(mask_dir / name) if name in mask_files else None,
                            category,
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cloth_p, person_p, tryon_p, mask_p, category = self.samples[idx]
        cloth = self.img_tf(Image.open(cloth_p).convert("RGB"))
        person = self.img_tf(Image.open(person_p).convert("RGB"))
        target = self.img_tf(Image.open(tryon_p).convert("RGB"))
        if mask_p is None:
            mask = torch.zeros((1, cloth.shape[1], cloth.shape[2]), dtype=cloth.dtype)
        else:
            mask = self.mask_tf(Image.open(mask_p).convert("L"))
            mask = (mask > 0.5).float()

        # Match original masked inference behavior: keep outside mask, fill masked area with gray image.
        gray = torch.zeros_like(person)
        vton_masked = torch.where(mask > 0.5, gray, person)

        return {
            "cloth": cloth,
            "vton_masked": vton_masked,
            "image_ori": person,
            "mask": mask,
            "target": target,
            "category": category,
        }


def _collate(rows):
    return TrainBatch(
        cloth=torch.stack([r["cloth"] for r in rows], dim=0),
        vton_masked=torch.stack([r["vton_masked"] for r in rows], dim=0),
        image_ori=torch.stack([r["image_ori"] for r in rows], dim=0),
        mask=torch.stack([r["mask"] for r in rows], dim=0),
        target=torch.stack([r["target"] for r in rows], dim=0),
        category=[r["category"] for r in rows],
    )


def tokenize_captions(tokenizer: CLIPTokenizer, captions: List[str], max_length: int):
    inputs = tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    return inputs.input_ids


def main():
    parser = argparse.ArgumentParser("Train OOTDiffusion with inference-faithful forward + MSE noise loss")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./train_outputs")
    parser.add_argument("--model_type", type=str, default="hd", choices=["hd", "dc"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vit_path", type=str, default="./checkpoints/clip-vit-large-patch14")
    parser.add_argument("--vae_path", type=str, default="./checkpoints/ootd")
    parser.add_argument("--model_path", type=str, default="./checkpoints/ootd")
    parser.add_argument("--unet_path", type=str, default="./checkpoints/ootd/ootd_hd/checkpoint-36000")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = OOTTrainDataset(args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=_collate,
    )

    vae = AutoencoderKL.from_pretrained(args.vae_path, subfolder="vae").to(device)
    scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    unet_garm = UNetGarm2DConditionModel.from_pretrained(args.unet_path, subfolder="unet_garm").to(device)
    unet_vton = UNetVton2DConditionModel.from_pretrained(args.unet_path, subfolder="unet_vton").to(device)
    auto_processor = AutoProcessor.from_pretrained(args.vit_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.vit_path).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    image_encoder.eval()
    text_encoder.eval()
    unet_garm.train()
    unet_vton.train()

    optim = AdamW(list(unet_garm.parameters()) + list(unet_vton.parameters()), lr=args.lr)

    step = 0
    while step < args.max_steps:
        for batch in loader:
            cloth = batch.cloth.to(device, non_blocking=True)
            vton_masked = batch.vton_masked.to(device, non_blocking=True)
            image_ori = batch.image_ori.to(device, non_blocking=True)
            mask = batch.mask.to(device, non_blocking=True)
            target = batch.target.to(device, non_blocking=True)
            bs = cloth.shape[0]

            with torch.no_grad():
                # Same conditioning construction style as original inference code.
                prompt_image = auto_processor(
                    images=[transforms.ToPILImage()(((img * 0.5) + 0.5).cpu().clamp(0, 1)) for img in cloth],
                    return_tensors="pt",
                ).to(device)
                prompt_img_emb = image_encoder(prompt_image.data["pixel_values"]).image_embeds.unsqueeze(1)
                if args.model_type == "hd":
                    prompt_embeds = text_encoder(tokenize_captions(tokenizer, [""] * bs, 2).to(device))[0]
                    prompt_embeds[:, 1:] = prompt_img_emb
                else:
                    prompt_embeds = text_encoder(tokenize_captions(tokenizer, batch.category, 3).to(device))[0]
                    prompt_embeds = torch.cat([prompt_embeds, prompt_img_emb], dim=1)

                garm_latents = vae.encode(cloth).latent_dist.mode()
                vton_latents = vae.encode(vton_masked).latent_dist.mode()
                image_ori_latents = vae.encode(image_ori).latent_dist.mode()
                target_latents = vae.encode(target).latent_dist.mode()
                mask_latents = F.interpolate(mask, size=(vton_latents.size(-2), vton_latents.size(-1)))

            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (bs,),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

            # Inference-faithful forward path:
            # 1) garment UNet generates spatial attention features
            # 2) vton UNet predicts noise from concat([noisy_latent, vton_latent])
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

            loss = F.mse_loss(noise_pred.float(), noise.float())
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            step += 1

            if step % args.log_interval == 0:
                print(f"[step {step}/{args.max_steps}] loss={loss.item():.6f}", flush=True)

            if step % args.save_interval == 0 or step == args.max_steps:
                ckpt = {
                    "step": step,
                    "unet_garm_state_dict": unet_garm.state_dict(),
                    "unet_vton_state_dict": unet_vton.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt, os.path.join(args.output_dir, f"ckpt_step_{step}.pt"))

            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()

