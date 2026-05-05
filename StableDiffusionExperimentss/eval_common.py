import math
import os
import random
from pathlib import Path
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


def _configure_quiet_eval_logging() -> None:
    # Keep eval output focused on checkpoint used + split progress + metrics.
    noisy = [
        "httpcore",
        "httpx",
        "urllib3",
        "huggingface_hub",
        "transformers",
        "PIL",
        "PIL.PngImagePlugin",
        "matplotlib",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.ERROR)

    # If root is set to DEBUG by environment, raise it to INFO.
    root = logging.getLogger()
    if root.level <= logging.DEBUG:
        root.setLevel(logging.INFO)

    try:
        from transformers.utils import logging as hf_logging  # type: ignore
        hf_logging.set_verbosity_error()
    except Exception:
        pass


_configure_quiet_eval_logging()

try:
    import lpips as lpips_lib
except ImportError as exc:
    raise ImportError("Install lpips: pip install lpips") from exc

try:
    from torchmetrics.image import (
        FrechetInceptionDistance,
        KernelInceptionDistance,
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
except ImportError as exc:
    raise ImportError("Install torchmetrics with image extras: pip install torchmetrics[image]") from exc

try:
    from config import IMAGE_SIZE, MODEL_NAME
except Exception:
    IMAGE_SIZE = 512
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
from utils import (
    collate_fn,
    get_curvton_test_dataloaders,
    get_triplet_test_dataloaders,
)


_VAE_CACHE = {}


def _safe_loader_kwargs(num_workers: int) -> dict:
    # Frontend/login nodes can crash with forked workers + pinned memory.
    if num_workers <= 0:
        return {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
    return {
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": False,
        "multiprocessing_context": "spawn",
    }


def _to_01(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32 and x.dtype != torch.float64 and x.dtype != torch.float16 and x.dtype != torch.bfloat16:
        x = x.float()
    if x.min() < -0.01 or x.max() > 1.01:
        x = (x + 1.0) * 0.5
    return x.clamp(0.0, 1.0)


def _to_u8(x01: torch.Tensor) -> torch.Tensor:
    return (x01 * 255.0).round().clamp(0, 255).to(torch.uint8)


def _get_metric_gt_vae(device: torch.device):
    key = str(device)
    if key in _VAE_CACHE:
        return _VAE_CACHE[key]
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    vae.eval()
    vae.requires_grad_(False)
    _VAE_CACHE[key] = vae
    return vae


@torch.no_grad()
def _vae_roundtrip_gt(gt_01: torch.Tensor, device: torch.device) -> torch.Tensor:
    vae = _get_metric_gt_vae(device)
    gt_in = gt_01 * 2.0 - 1.0
    latent_dist = vae.encode(gt_in).latent_dist
    latents = latent_dist.mean * vae.config.scaling_factor
    gt_recon = vae.decode(latents / vae.config.scaling_factor).sample
    return _to_01(gt_recon)


def _safe_compute(metric):
    try:
        out = metric.compute()
        if isinstance(out, tuple):
            return tuple(float(v.item()) for v in out)
        return float(out.item())
    except Exception:
        return None


class StreetTryOnEvalDataset(Dataset):
    """Eval-only loader for StreetTryOn image split.

    For unpaired metrics we only need image distributions.
    We still expose person/cloth/ground_truth keys for model forward compatibility.
    """

    def __init__(self, root_dir: str, split: str = "validation", size: int = 512):
        self.root_dir = root_dir
        self.split = split
        self.size = size
        self.image_dir = os.path.join(root_dir, split, "image")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"StreetTryOn image directory not found: {self.image_dir}")
        self.files = sorted(
            f for f in os.listdir(self.image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        )
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self._zero_mask = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = os.path.join(self.image_dir, self.files[idx])
        img_raw = Image.open(p).convert("RGB")
        img = self.tf(img_raw)
        cloth = self.tf(img_raw)
        if self._zero_mask is None or self._zero_mask.shape[-2:] != img.shape[-2:]:
            self._zero_mask = torch.zeros(1, img.shape[-2], img.shape[-1])
        return {
            "ground_truth": img,
            "person": img,
            "cloth": cloth,
            "mask": self._zero_mask,
        }


def get_street_tryon_loader(root_dir: str, split: str, batch_size: int, num_workers: int, size: int = 512):
    ds = StreetTryOnEvalDataset(root_dir=root_dir, split=split, size=size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        **_safe_loader_kwargs(num_workers),
    )


@dataclass
class EvalLoaders:
    curvton: Dict[str, DataLoader]
    triplet: Dict[str, DataLoader]
    street: Dict[str, DataLoader]


def build_eval_loaders(
    curvton_test_data_path: Optional[str],
    triplet_test_data_path: Optional[str],
    street_tryon_data_path: Optional[str],
    batch_size: int,
    num_workers: int,
    size: int = IMAGE_SIZE,
    gender: str = "all",
    street_split: str = "validation",
) -> EvalLoaders:
    curvton_loaders: Dict[str, DataLoader] = OrderedDict()
    triplet_loaders: Dict[str, DataLoader] = OrderedDict()
    street_loaders: Dict[str, DataLoader] = OrderedDict()

    if curvton_test_data_path:
        genders = ("female", "male") if gender == "all" else (gender,)
        c = get_curvton_test_dataloaders(
            root_dir=curvton_test_data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            size=size,
            genders=genders,
        )
        for k in ("easy", "medium", "hard"):
            if k in c:
                curvton_loaders[f"curvton_{k}"] = c[k]
        if "all" in c:
            curvton_loaders["curvton_overall"] = c["all"]

        # Additional CurvTON semantic splits built from cloth filename labels.
        # Expected stem examples:
        #   *_fc_012345_dresses
        #   *_mc_045678_upper_body
        #   *_mc_098765_kimono
        all_loader = c.get("all")
        if all_loader is not None and hasattr(all_loader, "dataset"):
            all_ds = all_loader.dataset
            if hasattr(all_ds, "datasets"):
                flat_triplets = []
                for sub_ds in all_ds.datasets:
                    if hasattr(sub_ds, "triplets"):
                        flat_triplets.extend(list(sub_ds.triplets))

                def _extract_label_from_triplet(tri):
                    cloth_path = tri[1]
                    stem = Path(cloth_path).stem
                    # label after _fc_<id>_ / _mc_<id>_
                    parts = stem.split("_")
                    if len(parts) >= 2:
                        # robust parse: find fc/mc token then skip id token
                        for i in range(len(parts) - 2):
                            if parts[i] in ("fc", "mc"):
                                return "_".join(parts[i + 2 :]).lower()
                    return stem.lower()

                # Traditional clothing labels observed in this repo's datasets.
                traditional_labels = {
                    "kimono", "yukata", "hanbok", "hanfu", "thobe", "kandura", "dishdasha",
                    "bisht", "djellaba", "boubou", "agbada", "sherwani", "kurta_pajama",
                    "salwar_kameez", "chapan", "changshan", "shenyi", "samue", "jinbei",
                    "daura_suruwal", "shuka", "kazakh_shapan", "uyghur_doppa_outfit",
                    "korean_dopo", "montsuki", "pashtun_dress", "ewe_kente_cloth",
                    "zhuang_ethnic_dress", "ainu_attus", "chakma_dress", "chut_thai",
                    "ao_ba_ba", "ryusou", "sherpa_bakhu",
                }

                def _build_subset_loader(name, pred):
                    idxs = [i for i, tri in enumerate(flat_triplets) if pred(_extract_label_from_triplet(tri))]
                    if not idxs:
                        return
                    subset = torch.utils.data.Subset(all_ds, idxs)
                    curvton_loaders[name] = DataLoader(
                        subset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        collate_fn=collate_fn,
                        **_safe_loader_kwargs(num_workers),
                    )

                _build_subset_loader("curvton_dresses", lambda lbl: lbl == "dresses")
                _build_subset_loader("curvton_upper_body", lambda lbl: lbl == "upper_body")
                _build_subset_loader("curvton_lower_body", lambda lbl: lbl == "lower_body")
                _build_subset_loader("curvton_traditional", lambda lbl: lbl in traditional_labels)
                _build_subset_loader("curvton_non_traditional", lambda lbl: lbl not in traditional_labels)

    if triplet_test_data_path:
        t = get_triplet_test_dataloaders(
            root_dir=triplet_test_data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            size=0,
        )
        for k in ("dresscode_dresses", "dresscode_lower", "dresscode_upper", "viton_hd"):
            if k in t:
                triplet_loaders[k] = t[k]
        if t:
            merged = ConcatDataset([loader.dataset for loader in t.values()])
            triplet_loaders["triplet_overall"] = DataLoader(
                merged,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
                **_safe_loader_kwargs(num_workers),
            )

    if street_tryon_data_path:
        street_loaders["street_tryon"] = get_street_tryon_loader(
            root_dir=street_tryon_data_path,
            split=street_split,
            batch_size=batch_size,
            num_workers=num_workers,
            size=0,
        )

    return EvalLoaders(curvton=curvton_loaders, triplet=triplet_loaders, street=street_loaders)


def evaluate_loader(
    loader: DataLoader,
    predict_fn: Callable[[dict, torch.device], torch.Tensor],
    device: torch.device,
    progress_name: str = "eval",
    max_batches: int = 0,
    eval_frac: float = 1.0,
    paired_metrics: bool = True,
    unpaired_metrics: bool = True,
    apply_vae_gt_roundtrip: bool = False,
    feature_cache_dir: Optional[str] = None,
):
    lpips_fn = lpips_lib.LPIPS(net="alex").to(device) if paired_metrics else None
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device) if paired_metrics else None
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device) if paired_metrics else None

    est_n = len(loader.dataset) if hasattr(loader, "dataset") else 256
    kid_subset = max(2, min(50, int(est_n)))

    # Compute FID in both input conventions:
    # - normalize=False expects uint8 [0,255]
    # - normalize=True expects float [0,1]
    fid_p = FrechetInceptionDistance(feature=64, reset_real_features=True, normalize=False).to(device) if paired_metrics else None
    fid_p_norm_true = FrechetInceptionDistance(feature=64, reset_real_features=True, normalize=True).to(device) if paired_metrics else None
    kid_p = KernelInceptionDistance(feature=64, reset_real_features=True, normalize=False, subset_size=kid_subset).to(device) if paired_metrics else None
    fid_u = FrechetInceptionDistance(feature=64, reset_real_features=True, normalize=False).to(device) if unpaired_metrics else None
    fid_u_norm_true = FrechetInceptionDistance(feature=64, reset_real_features=True, normalize=True).to(device) if unpaired_metrics else None
    kid_u = KernelInceptionDistance(feature=64, reset_real_features=True, normalize=False, subset_size=kid_subset).to(device) if unpaired_metrics else None

    lpips_sum = 0.0
    ssim_sum = 0.0
    psnr_sum = 0.0
    n_img = 0

    total_batches = len(loader)
    frac_batches = total_batches
    if eval_frac > 0 and eval_frac < 1.0:
        frac_batches = max(1, int(math.ceil(total_batches * eval_frac)))
    effective_batches = frac_batches if max_batches <= 0 else min(frac_batches, max_batches)

    # Random batch sampling: pick which batches to evaluate instead of
    # always using the first N (sequential).  This ensures each evaluation
    # run covers a random subset while preserving triplet correspondence
    # (person/cloth/tryon stay matched within each batch).
    if effective_batches < total_batches:
        selected_indices = set(random.sample(range(total_batches), effective_batches))
    else:
        selected_indices = None  # evaluate all

    print(
        f"[{progress_name}] starting: total_batches={total_batches}, "
        f"effective_batches={effective_batches}, eval_frac={eval_frac}, "
        f"sampling={'random' if selected_indices is not None else 'all'}"
    )

    for bidx, batch in enumerate(tqdm(loader, total=effective_batches, desc=f"[{progress_name}] Generation")):
        if selected_indices is not None and bidx not in selected_indices:
            continue
        if batch is None:
            continue

        with torch.no_grad():
            pred = predict_fn(batch, device)

        gt = _to_01(batch["ground_truth"].to(device))
        if apply_vae_gt_roundtrip:
            gt = _vae_roundtrip_gt(gt, device)
        pred = _to_01(pred.to(device))
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        if pred.shape[0] != gt.shape[0]:
            bs = min(pred.shape[0], gt.shape[0])
            pred = pred[:bs]
            gt = gt[:bs]

        bs = gt.shape[0]
        if bs == 0:
            continue
        n_img += bs

        gt_u8 = _to_u8(gt)
        pred_u8 = _to_u8(pred)

        if paired_metrics:
            lp = lpips_fn(pred * 2 - 1, gt * 2 - 1).mean().item()
            lpips_sum += lp * bs
            ssim_sum += float(ssim_metric(pred, gt).item()) * bs
            psnr_sum += float(psnr_metric(pred, gt).item()) * bs
            fid_p.update(gt_u8, real=True)
            fid_p.update(pred_u8, real=False)
            fid_p_norm_true.update(gt, real=True)
            fid_p_norm_true.update(pred, real=False)
            kid_p.update(gt_u8, real=True)
            kid_p.update(pred_u8, real=False)

        if unpaired_metrics:
            perm = torch.randperm(bs, device=gt_u8.device)
            gt_shuf = gt_u8[perm]
            gt_shuf_f = gt[perm]
            fid_u.update(gt_shuf, real=True)
            fid_u.update(pred_u8, real=False)
            fid_u_norm_true.update(gt_shuf_f, real=True)
            fid_u_norm_true.update(pred, real=False)
            kid_u.update(gt_shuf, real=True)
            kid_u.update(pred_u8, real=False)

        if feature_cache_dir:
            cache_dir = Path(feature_cache_dir)
            gen_dir = cache_dir / "generated"
            gt_dir = cache_dir / "ground_truth"
            diff_dir = cache_dir / "diff_abs"
            gen_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
            diff_dir.mkdir(parents=True, exist_ok=True)

            pred_cpu = pred_u8.detach().cpu()
            gt_cpu = gt_u8.detach().cpu()
            for i in range(bs):
                sample_idx = n_img - bs + i
                pred_img = pred_cpu[i].permute(1, 2, 0).numpy()
                gt_img = gt_cpu[i].permute(1, 2, 0).numpy()
                diff_img = np.abs(pred_img.astype(np.int16) - gt_img.astype(np.int16)).astype(np.uint8)
                stem = f"sample_{sample_idx:08d}"
                Image.fromarray(pred_img).save(gen_dir / f"{stem}__generated.png")
                Image.fromarray(gt_img).save(gt_dir / f"{stem}__target.png")
                Image.fromarray(diff_img).save(diff_dir / f"{stem}__generated_vs_target_diff_abs.png")

    out = {"n_images": int(n_img)}
    if paired_metrics and n_img > 0:
        out["lpips"] = lpips_sum / n_img
        out["ssim"] = ssim_sum / n_img
        out["psnr"] = psnr_sum / n_img
        fidp = _safe_compute(fid_p)
        fidp_true = _safe_compute(fid_p_norm_true)
        kidp = _safe_compute(kid_p)
        out["fid_paired"] = fidp if fidp is not None else float("nan")
        out["fid_paired_norm_false"] = fidp if fidp is not None else float("nan")
        out["fid_paired_norm_true"] = fidp_true if fidp_true is not None else float("nan")
        if isinstance(kidp, tuple):
            out["kid_paired_mean"], out["kid_paired_std"] = kidp
        else:
            out["kid_paired_mean"], out["kid_paired_std"] = float("nan"), float("nan")

    if unpaired_metrics and n_img > 0:
        fidu = _safe_compute(fid_u)
        fidu_true = _safe_compute(fid_u_norm_true)
        kidu = _safe_compute(kid_u)
        out["fid_unpaired"] = fidu if fidu is not None else float("nan")
        out["fid_unpaired_norm_false"] = fidu if fidu is not None else float("nan")
        out["fid_unpaired_norm_true"] = fidu_true if fidu_true is not None else float("nan")
        if isinstance(kidu, tuple):
            out["kid_unpaired_mean"], out["kid_unpaired_std"] = kidu
        else:
            out["kid_unpaired_mean"], out["kid_unpaired_std"] = float("nan"), float("nan")

    return out


def summarize_group(title: str, results: Dict[str, dict]):
    print(f"\n=== {title} ===")
    if not results:
        print("No splits evaluated.")
        return
    for split, m in results.items():
        print(f"\n[{split}] n={m.get('n_images', 0)}")
        if "lpips" in m:
            print(f"LPIPS={m['lpips']:.4f}  SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f}")
            print(f"FID(paired)={m.get('fid_paired', float('nan')):.4f}  KID(paired)={m.get('kid_paired_mean', float('nan')):.6f}")
            print(
                f"FID(paired,norm_false)={m.get('fid_paired_norm_false', float('nan')):.4f}  "
                f"FID(paired,norm_true)={m.get('fid_paired_norm_true', float('nan')):.4f}"
            )
        print(f"FID(unpaired)={m.get('fid_unpaired', float('nan')):.4f}  KID(unpaired)={m.get('kid_unpaired_mean', float('nan')):.6f}")
        print(
            f"FID(unpaired,norm_false)={m.get('fid_unpaired_norm_false', float('nan')):.4f}  "
            f"FID(unpaired,norm_true)={m.get('fid_unpaired_norm_true', float('nan')):.4f}"
        )


def summarize_single(split: str, m: dict):
    print(f"\n[{split}] completed. n={m.get('n_images', 0)}")
    if "lpips" in m:
        print(f"LPIPS={m['lpips']:.4f}  SSIM={m['ssim']:.4f}  PSNR={m['psnr']:.2f}")
        print(f"FID(paired)={m.get('fid_paired', float('nan')):.4f}  KID(paired)={m.get('kid_paired_mean', float('nan')):.6f}")
        print(
            f"FID(paired,norm_false)={m.get('fid_paired_norm_false', float('nan')):.4f}  "
            f"FID(paired,norm_true)={m.get('fid_paired_norm_true', float('nan')):.4f}"
        )
    print(f"FID(unpaired)={m.get('fid_unpaired', float('nan')):.4f}  KID(unpaired)={m.get('kid_unpaired_mean', float('nan')):.6f}")
    print(
        f"FID(unpaired,norm_false)={m.get('fid_unpaired_norm_false', float('nan')):.4f}  "
        f"FID(unpaired,norm_true)={m.get('fid_unpaired_norm_true', float('nan')):.4f}"
    )


def evaluate_all_splits(
    loaders: EvalLoaders,
    predict_fn: Callable[[dict, torch.device], torch.Tensor],
    device: torch.device,
    max_batches: int = 0,
    eval_frac_curvton: float = 0.02,
    eval_frac_curvton_overall: Optional[float] = None,
    eval_frac_triplet: float = 0.02,
    eval_frac_street: float = 0.02,
    eval_frac_curvton_extra: Optional[float] = None,
    feature_cache_root: Optional[str] = None,
):
    curvton_results: Dict[str, dict] = OrderedDict()
    triplet_results: Dict[str, dict] = OrderedDict()
    street_results: Dict[str, dict] = OrderedDict()

    for name, loader in loaders.triplet.items():
        triplet_results[name] = evaluate_loader(
            loader=loader,
            predict_fn=predict_fn,
            device=device,
            progress_name=name,
            max_batches=max_batches,
            eval_frac=eval_frac_triplet,
            paired_metrics=True,
            unpaired_metrics=True,
            apply_vae_gt_roundtrip=False,
            feature_cache_dir=(str(Path(feature_cache_root) / "triplet" / name) if feature_cache_root else None),
        )
        summarize_single(name, triplet_results[name])

    for name, loader in loaders.street.items():
        street_results[name] = evaluate_loader(
            loader=loader,
            predict_fn=predict_fn,
            device=device,
            progress_name=name,
            max_batches=max_batches,
            eval_frac=eval_frac_street,
            paired_metrics=False,
            unpaired_metrics=True,
            apply_vae_gt_roundtrip=False,
            feature_cache_dir=(str(Path(feature_cache_root) / "street_tryon" / name) if feature_cache_root else None),
        )
        summarize_single(name, street_results[name])

    for name, loader in loaders.curvton.items():
        frac = eval_frac_curvton
        # Use separate fraction for curvton_overall if provided.
        if eval_frac_curvton_overall is not None and name == "curvton_overall":
            frac = eval_frac_curvton_overall
        elif eval_frac_curvton_extra is not None and name in {
            "curvton_traditional",
            "curvton_non_traditional",
            "curvton_dresses",
            "curvton_upper_body",
            "curvton_lower_body",
        }:
            frac = eval_frac_curvton_extra
        curvton_results[name] = evaluate_loader(
            loader=loader,
            predict_fn=predict_fn,
            device=device,
            progress_name=name,
            max_batches=max_batches,
            eval_frac=frac,
            paired_metrics=True,
            unpaired_metrics=True,
            apply_vae_gt_roundtrip=True,
            feature_cache_dir=(str(Path(feature_cache_root) / "curvton" / name) if feature_cache_root else None),
        )
        summarize_single(name, curvton_results[name])

    summarize_group("TRIPLET (DRESSCODE + VITONHD + OVERALL)", triplet_results)
    summarize_group("STREET TRYON (UNPAIRED ONLY)", street_results)
    summarize_group("CURVTON SPLITS + OVERALL", curvton_results)

    return {
        "curvton": curvton_results,
        "triplet": triplet_results,
        "street_tryon": street_results,
    }

