# Cross-Architecture Local Training

This directory now contains local implementations for each cross-architecture experiment. The shell scripts do not depend on cloning external official repositories.

## OOTDiffusion

- Script: `OOTDiffusion/train_ootdiffusion.sh`
- Local trainer: `OOTDiffusion/train_ootdiffusion_local.py`
- Architecture: latent diffusion with outfitting-fusion conditioning from person and garment latents.
- Input tensors: `person [B,3,512,512]`, `cloth [B,3,512,512]`, `ground_truth [B,3,512,512]`.
- Model input: `cat([noisy_target_latent, person_latent, cloth_latent], channel) -> [B,12,H,W]`.
- Output tensor: predicted noise latent `[B,4,H,W]`.
- Loss: MSE between predicted and sampled diffusion noise.

## IDMVTON

- Script: `IDMVTON/train_idm_vton.sh`
- Local trainer: `IDMVTON/train_idm_vton_local.py`
- Architecture: SDXL inpainting-style denoising UNet with garment latent conditioning.
- Input tensors: `person`, `cloth`, `ground_truth`.
- Model input: `cat([noisy_target_latent, mask, masked_person_latent, cloth_latent], channel) -> [B,13,H,W]`.
- Output tensor: predicted noise latent `[B,4,H,W]`.
- Loss: diffusion noise-prediction MSE.

## StableVTON

- Script: `StableVTON/train_stable_vton.sh`
- Local trainer: `StableVTON/train_stable_vton_local.py`
- Architecture: StableVITON-style modified latent diffusion UNet with 13-channel try-on conditioning.
- Input tensors: `person`, `cloth`, `ground_truth`.
- Preprocessing keys (constructed locally): `agnostic`, `agnostic-mask`, `densepose`(surrogate), `cloth_mask`, `gt_cloth_warped_mask`.
- Dataset adaptation used:
  - `agnostic` uses initial person image.
  - `agnostic-mask` is a black 1-channel image.
  - `densepose` surrogate uses initial person image.
- Model input: `cat([noisy_target_latent, mask, agnostic_latent, pose_latent], channel) -> [B,13,H,W]`.
- Output tensor: predicted noise latent `[B,4,H,W]`.
- Loss: diffusion noise-prediction MSE, plus optional ATV-style smoothness term over warped-mask proxy.

## CPVTON

- Script: `CPVTON/train_cpvton.sh`
- Local trainer: `CPVTON/train_cpvton_local.py`
- Architecture: two-stage CP-VTON pipeline.
- `CPVTON_STAGE=GMM`: geometric matching module predicts TPS control points and warps cloth.
- `CPVTON_STAGE=TOM`: try-on module predicts rendered image and composition mask, then blends with warped cloth.
- GMM output tensors: warped cloth `[B,3,512,512]`, TPS grid `[B,512,512,2]`.
- TOM output tensors: final try-on `[B,3,512,512]`, rendered person `[B,3,512,512]`, composition mask `[B,1,512,512]`.
- Losses: GMM warp L1 plus TPS regularization; TOM reconstruction L1, render L1, and mask smoothness.
