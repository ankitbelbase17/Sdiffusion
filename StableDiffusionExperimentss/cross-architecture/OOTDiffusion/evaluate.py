import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import AutoencoderKL, DDPMScheduler

THIS_DIR = os.path.dirname(__file__)
STABLE_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
CROSS_ARCH_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
OOT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "OOTDiffusion"))
for p in (STABLE_DIR, CROSS_ARCH_DIR, OOT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from config import CURVTON_TEST_PATH, STREET_TRYON_PATH, TRIPLET_TEST_PATH  # noqa: E402
except Exception:
    CURVTON_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test"
    TRIPLET_TEST_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/triplet_dataset_backup_1"
    STREET_TRYON_PATH = "/iopsstor/scratch/cscs/dbartaula/human_gen/benchmark_datasets/street_tryon"
from eval_common import build_eval_loaders, evaluate_all_splits  # noqa: E402
from ootd.pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel  # noqa: E402
from ootd.pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel  # noqa: E402


DEFAULT_CURVTON_SPLITS = ["easy", "medium", "hard", "overall", "traditional", "non_traditional", "dresses", "upper_body", "lower_body"]
_CURVTON_SPLIT_TO_LOADER = {
    "easy": "curvton_easy",
    "medium": "curvton_medium",
    "hard": "curvton_hard",
    "overall": "curvton_overall",
    "traditional": "curvton_traditional",
    "non_traditional": "curvton_non_traditional",
    "dresses": "curvton_dresses",
    "upper_body": "curvton_upper_body",
    "lower_body": "curvton_lower_body",
}


def _build_modules(device):
    vit_path = os.path.join(OOT_ROOT, "checkpoints", "clip-vit-large-patch14")
    vae_path = os.path.join(OOT_ROOT, "checkpoints", "ootd")
    model_path = os.path.join(OOT_ROOT, "checkpoints", "ootd")
    unet_path = os.path.join(OOT_ROOT, "checkpoints", "ootd", "ootd_hd", "checkpoint-36000")
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to(device).eval()
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    unet_garm = UNetGarm2DConditionModel.from_pretrained(unet_path, subfolder="unet_garm").to(device).eval()
    unet_vton = UNetVton2DConditionModel.from_pretrained(unet_path, subfolder="unet_vton").to(device).eval()
    auto_processor = AutoProcessor.from_pretrained(vit_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(vit_path).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device).eval()
    return vae, scheduler, unet_garm, unet_vton, auto_processor, image_encoder, tokenizer, text_encoder


def _tokenize(tokenizer, captions, max_length):
    return tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids


def build_predict_fn(mods, num_inference_steps: int, is_masked: bool):
    vae, scheduler, unet_garm, unet_vton, auto_processor, image_encoder, tokenizer, text_encoder = mods

    @torch.no_grad()
    def _predict(batch, device):
        person = batch["person"].to(device)
        cloth = batch["cloth"].to(device)
        bs = cloth.shape[0]
        cloth_pil = [transforms.ToPILImage()(((img * 0.5) + 0.5).cpu().clamp(0, 1)) for img in cloth]
        prompt_image = auto_processor(images=cloth_pil, return_tensors="pt").to(device)
        prompt_img_emb = image_encoder(prompt_image.data["pixel_values"]).image_embeds.unsqueeze(1)
        prompt_embeds = text_encoder(_tokenize(tokenizer, [""] * bs, 2).to(device))[0]
        prompt_embeds[:, 1:] = prompt_img_emb

        if is_masked and "mask" in batch:
            mask = batch["mask"].to(device)
            person_masked = torch.where(mask > 0.5, torch.zeros_like(person), person)
        else:
            mask = None
            person_masked = person

        garm_latents = vae.encode(cloth).latent_dist.mode()
        vton_latents = vae.encode(person_masked).latent_dist.mode()
        image_ori_lat = vae.encode(person).latent_dist.mode()
        mask_lat = None if mask is None else F.interpolate(mask, size=vton_latents.shape[-2:]).to(device=device, dtype=vton_latents.dtype)

        latents = torch.randn_like(vton_latents)
        noise = latents.clone()
        scheduler.set_timesteps(num_inference_steps, device=device)
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

    return _predict


def _resolve_feature_cache_root(args):
    if args.feature_cache_dir:
        return args.feature_cache_dir
    run_name = "init_weights"
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        run_name = ckpt.parent.parent.name if ckpt.parent.name == "checkpoints" else ckpt.parent.name
    elif getattr(args, "run_name", None):
        run_name = args.run_name
    return str(Path(args.feature_cache_root) / run_name / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def _resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if args.cuda_device is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{args.cuda_device}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    device = _resolve_device(args)
    mods = _build_modules(device)
    _, _, unet_garm, unet_vton, _, _, _, _ = mods
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        unet_garm.load_state_dict(ckpt["unet_garm_state_dict"], strict=False)
        unet_vton.load_state_dict(ckpt["unet_vton_state_dict"], strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using initial weights (no checkpoint load).")

    loaders = build_eval_loaders(
        curvton_test_data_path=args.curvton_test_data_path,
        triplet_test_data_path=args.triplet_test_data_path,
        street_tryon_data_path=args.street_tryon_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gender=args.gender,
        street_split=args.street_split,
    )
    if args.curvton_splits:
        requested = [s.strip().lower() for s in args.curvton_splits.split(",") if s.strip()]
        keys = [_CURVTON_SPLIT_TO_LOADER[s] for s in requested if s in _CURVTON_SPLIT_TO_LOADER]
        loaders.curvton = {k: v for k, v in loaders.curvton.items() if k in keys}

    results = evaluate_all_splits(
        loaders=loaders,
        predict_fn=build_predict_fn(mods, args.num_inference_steps, args.is_masked),
        device=device,
        max_batches=args.max_batches,
        eval_frac_curvton=args.eval_frac_curvton,
        eval_frac_curvton_overall=args.eval_frac_curvton_overall,
        eval_frac_curvton_extra=args.eval_frac_curvton_extra,
        eval_frac_triplet=args.eval_frac_triplet,
        eval_frac_street=args.eval_frac_street,
        feature_cache_root=_resolve_feature_cache_root(args),
    )
    print("\nEvaluation metrics:\n" + json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics JSON to: {args.output_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser("Evaluate OOT with train-consistent forward")
    p.add_argument("--run_name", type=str, default="train_ootdiffusion")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--curvton_test_data_path", type=str, default=CURVTON_TEST_PATH)
    p.add_argument("--triplet_test_data_path", type=str, default=TRIPLET_TEST_PATH)
    p.add_argument("--street_tryon_data_path", type=str, default=STREET_TRYON_PATH)
    p.add_argument("--street_split", type=str, default="validation", choices=["train", "validation"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gender", type=str, default="all", choices=["female", "male", "all"])
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--eval_frac_curvton", type=float, default=0.02)
    p.add_argument("--eval_frac_curvton_overall", type=float, default=None)
    p.add_argument("--eval_frac_curvton_extra", type=float, default=0.02)
    p.add_argument("--eval_frac_triplet", type=float, default=0.02)
    p.add_argument("--eval_frac_street", type=float, default=0.02)
    p.add_argument("--curvton_splits", type=str, default=",".join(DEFAULT_CURVTON_SPLITS))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cuda_device", type=int, default=None)
    p.add_argument("--feature_cache_root", type=str, default="/iopsstor/scratch/cscs/dbartaula/featurecache")
    p.add_argument("--feature_cache_dir", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--is_masked", action="store_true")
    main(p.parse_args())

