"""Official StableVITON Inference Wrapper."""

import argparse
import os
import sys
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from common import DistInfo
from train_stable_vton_mask_local import stableviton_preprocess as stableviton_preprocess_mask
from train_stable_vton_local import stableviton_preprocess as stableviton_preprocess_unmask

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SLURM_VITON_DIR = "/iopsstor/scratch/cscs/dbartaula/StableVITON"
LOCAL_VITON_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", "StableVITON"))

STABLE_VITON_DIR = SLURM_VITON_DIR if os.path.exists(SLURM_VITON_DIR) else LOCAL_VITON_DIR
if STABLE_VITON_DIR not in sys.path:
    sys.path.insert(0, STABLE_VITON_DIR)

from omegaconf import OmegaConf
from cldm.model import create_model


def _load_image(path: str, size: int) -> torch.Tensor:
    tf_list = []
    if size and size > 0:
        tf_list.append(transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
    tf_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tf = transforms.Compose(tf_list)
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


def _load_mask(path: str, size: int) -> torch.Tensor:
    tf_list = []
    if size and size > 0:
        tf_list.append(transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST))
    tf_list.append(transforms.ToTensor())
    tf = transforms.Compose(tf_list)
    mask = tf(Image.open(path).convert("L")).unsqueeze(0)
    return (mask > 0.5).float()


def save_tensor_image(tensor, path):
    x = ((tensor.detach().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    Image.fromarray(x).save(path)


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    config_path = os.path.join(STABLE_VITON_DIR, "configs", "VITONHD.yaml")
    config = OmegaConf.load(config_path)
    
    model = create_model(config_path, config=config).to(device)
    model.eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    else:
        raise ValueError("Must provide --checkpoint for inference.")

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        person = _load_image(args.person, args.size).to(device)
        cloth = _load_image(args.cloth, args.size).to(device)
        gt = person # Dummy GT
        
        if args.is_masked:
            if not args.mask:
                raise ValueError("Must provide --mask when --is_masked is True")
            mask = _load_mask(args.mask, args.size).to(device)
            if args.pose:
                pose = _load_image(args.pose, args.size).to(device)
            else:
                pose = person # Surrogate
            grey_fill = torch.full_like(person, 0.5)
            agnostic = torch.where(mask > 0.5, grey_fill, person)
            agn_mask = mask
            image_densepose = pose
        else:
            # Unmasked logic
            agnostic = person
            agn_mask = torch.zeros(1, 1, person.shape[2], person.shape[3], device=device, dtype=person.dtype)
            image_densepose = torch.zeros_like(person)
            mask = agn_mask

        official_batch = {
            "image": gt,
            "cloth": cloth,
            "agn": agnostic,
            "agn_mask": agn_mask,
            "image_densepose": image_densepose,
            "txt": [""] * person.shape[0],
            "gt_cloth_warped_mask": mask,
        }

        log_dict = model.log_images(
            official_batch,
            N=1,
            sample=True,
            unconditional_guidance_scale=args.guidance_scale,
            ddim_steps=args.num_inference_steps,
        )
        
        pred_key = f"samples_cfg_scale_{args.guidance_scale:.2f}"
        if pred_key in log_dict:
            preds = log_dict[pred_key]
        else:
            preds = log_dict["samples"]

        out_path = args.output if args.output else os.path.join(args.save_dir, "stable_official_single.png")
        save_tensor_image(preds[0], out_path)
        print(f"Inference complete. Saved 1 image to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official StableVITON Single Image Inference")
    parser.add_argument("--person", type=str, required=True, help="Person image path")
    parser.add_argument("--cloth", type=str, required=True, help="Cloth image path")
    parser.add_argument("--mask", type=str, default=None, help="Mask image path (required if is_masked)")
    parser.add_argument("--pose", type=str, default=None, help="Pose image path (optional)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="runs/cross_architecture/stable_infer")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--is_masked", action="store_true", help="Set to use the masked pipeline")
    args = parser.parse_args()
    main(args)
