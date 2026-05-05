import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler

THIS_DIR = os.path.dirname(__file__)
CROSS_ARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
OOT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "OOTDiffusion"))
for p in (CROSS_ARCH_DIR, OOT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import DistInfo, batch_images, build_curvton_loader, latest_checkpoint  # noqa: E402
from ootd.pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel  # noqa: E402
from ootd.pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel  # noqa: E402


def _load_image(path: str, size: int) -> torch.Tensor:
    tf_list = []
    if size and size > 0:
        tf_list.append(transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
    tf_list.extend([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    return transforms.Compose(tf_list)(Image.open(path).convert("RGB")).unsqueeze(0)


def _load_mask(path: str, size: int) -> torch.Tensor:
    tf_list = []
    if size and size > 0:
        tf_list.append(transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST))
    tf_list.append(transforms.ToTensor())
    return (transforms.Compose(tf_list)(Image.open(path).convert("L")).unsqueeze(0) > 0.5).float()


def _save_tensor_image(tensor, path):
    x = ((tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    Image.fromarray(x.permute(1, 2, 0).numpy()).save(path)


def _tokenize(tokenizer, captions, max_length):
    return tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids


def _build_modules(device):
    vit_path = os.path.join(OOT_ROOT, "checkpoints", "clip-vit-large-patch14")
    vae_path = os.path.join(OOT_ROOT, "checkpoints", "ootd")
    model_path = os.path.join(OOT_ROOT, "checkpoints", "ootd")
    unet_path = os.path.join(OOT_ROOT, "checkpoints", "ootd", "ootd_hd", "checkpoint-36000")
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    unet_garm = UNetGarm2DConditionModel.from_pretrained(unet_path, subfolder="unet_garm").to(device)
    unet_vton = UNetVton2DConditionModel.from_pretrained(unet_path, subfolder="unet_vton").to(device)
    auto_processor = AutoProcessor.from_pretrained(vit_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(vit_path).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device).eval()
    vae.eval()
    return vae, scheduler, unet_garm, unet_vton, auto_processor, image_encoder, tokenizer, text_encoder


@torch.no_grad()
def _predict(vae, scheduler, unet_garm, unet_vton, auto_processor, image_encoder, tokenizer, text_encoder, person, cloth, mask, steps, device):
    bs = cloth.shape[0]
    cloth_pil = [transforms.ToPILImage()(((img * 0.5) + 0.5).cpu().clamp(0, 1)) for img in cloth]
    prompt_image = auto_processor(images=cloth_pil, return_tensors="pt").to(device)
    prompt_img_emb = image_encoder(prompt_image.data["pixel_values"]).image_embeds.unsqueeze(1)
    prompt_embeds = text_encoder(_tokenize(tokenizer, [""] * bs, 2).to(device))[0]
    prompt_embeds[:, 1:] = prompt_img_emb

    grey = torch.zeros_like(person)
    person_masked = torch.where(mask > 0.5, grey, person) if mask is not None else person
    garm_latents = vae.encode(cloth).latent_dist.mode()
    vton_latents = vae.encode(person_masked).latent_dist.mode()
    image_ori_lat = vae.encode(person).latent_dist.mode()
    mask_lat = None if mask is None else F.interpolate(mask, size=vton_latents.shape[-2:]).to(device=device, dtype=vton_latents.dtype)

    latents = torch.randn_like(vton_latents)
    noise = latents.clone()
    scheduler.set_timesteps(steps, device=device)
    for i, t in enumerate(scheduler.timesteps):
        t_batch = torch.full((bs,), int(t), device=device, dtype=torch.long)
        _, sp = unet_garm(garm_latents, 0, encoder_hidden_states=prompt_embeds, return_dict=False)
        n_pred = unet_vton(torch.cat([latents, vton_latents], dim=1), sp.copy(), t_batch, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        latents = scheduler.step(n_pred, t, latents).prev_sample
        if mask_lat is not None:
            if i < len(scheduler.timesteps) - 1:
                nt = scheduler.timesteps[i + 1]
                init_lat = scheduler.add_noise(image_ori_lat, noise, torch.tensor([nt], device=device))
            else:
                init_lat = image_ori_lat
            latents = (1.0 - mask_lat) * init_lat + mask_lat * latents
    return vae.decode(latents).sample


def infer(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    vae, scheduler, unet_garm, unet_vton, auto_processor, image_encoder, tokenizer, text_encoder = _build_modules(device)
    ckpt_path = args.checkpoint or latest_checkpoint(os.path.join(args.output_dir, args.run_name))
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    unet_garm.load_state_dict(ckpt["unet_garm_state_dict"], strict=False)
    unet_vton.load_state_dict(ckpt["unet_vton_state_dict"], strict=False)
    os.makedirs(args.save_dir, exist_ok=True)

    written = 0
    if args.person and args.cloth:
        person = _load_image(args.person, args.size).to(device)
        cloth = _load_image(args.cloth, args.size).to(device)
        mask = _load_mask(args.mask, args.size).to(device) if args.is_masked and args.mask else None
        out = _predict(vae, scheduler, unet_garm, unet_vton, auto_processor, image_encoder, tokenizer, text_encoder, person, cloth, mask, args.num_inference_steps, device)
        out_path = args.output if args.output else os.path.join(args.save_dir, "oot_single.png")
        _save_tensor_image(out[0], out_path)
        print(f"Inference complete. Saved 1 image to {out_path}")
        return

    loader_args = argparse.Namespace(curvton_data_path=args.curvton_data_path, difficulty=args.difficulty, gender=args.gender, batch_size=1, num_workers=args.num_workers)
    dist_info = DistInfo(rank=0, local_rank=0, world_size=1, device=device, is_main=True)
    loader, _ = build_curvton_loader(loader_args, dist_info)
    for batch in loader:
        person, cloth, _ = batch_images(batch, device)
        mask = batch["mask"].to(device) if args.is_masked and "mask" in batch else None
        out = _predict(vae, scheduler, unet_garm, unet_vton, auto_processor, image_encoder, tokenizer, text_encoder, person, cloth, mask, args.num_inference_steps, device)
        _save_tensor_image(out[0], os.path.join(args.save_dir, f"oot_{written:05d}.png"))
        written += 1
        if written >= args.num_samples:
            break
    print(f"Inference complete. Saved {written} images to {args.save_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OOT inference with train-consistent forward")
    p.add_argument("--curvton_data_path", type=str, default=None)
    p.add_argument("--difficulty", type=str, default="all", choices=["easy", "medium", "hard", "all", "easy_hard", "medium_hard"])
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--output_dir", type=str, default="runs/cross_architecture")
    p.add_argument("--run_name", type=str, default="train_ootdiffusion")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="runs/cross_architecture/oot_infer")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--person", type=str, default=None)
    p.add_argument("--cloth", type=str, default=None)
    p.add_argument("--mask", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--size", type=int, default=0)
    p.add_argument("--is_masked", action="store_true")
    a = p.parse_args()
    if not (a.person and a.cloth) and not a.curvton_data_path:
        p.error("Provide either --person and --cloth, or --curvton_data_path.")
    infer(a)

