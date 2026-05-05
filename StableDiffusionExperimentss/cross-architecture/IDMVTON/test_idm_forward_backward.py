import importlib.util
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Cfg(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class FakeLatentDist:
    def __init__(self, lat):
        self._lat = lat

    def sample(self):
        return self._lat


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=0.18215)
        self.proj = nn.Conv2d(3, 4, 1)

    def encode(self, image):
        lat = F.avg_pool2d(self.proj(image), kernel_size=8, stride=8)
        return SimpleNamespace(latent_dist=FakeLatentDist(lat))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000, prediction_type="epsilon")

    def add_noise(self, target, noise, timesteps):
        alpha = 0.1 + timesteps.float().view(-1, 1, 1, 1) / 1000.0
        return target + alpha * noise

    def get_velocity(self, target, noise, timesteps):
        _ = target, timesteps
        return noise

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeTokenizer:
    model_max_length = 8

    def __call__(self, prompts, max_length, padding, truncation, return_tensors):
        _ = max_length, padding, truncation, return_tensors
        b = len(prompts)
        return SimpleNamespace(input_ids=torch.ones(b, self.model_max_length, dtype=torch.long))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeTextOut:
    def __init__(self, hidden, pooled):
        self.hidden_states = [hidden, hidden, hidden]
        self._pooled = pooled

    def __getitem__(self, idx):
        if idx == 0:
            return self._pooled
        raise IndexError(idx)


class FakeTextEncoder(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(1024, hidden_dim)

    def forward(self, ids, output_hidden_states=False):
        x = self.emb(ids)
        pooled = x.mean(dim=1)
        if output_hidden_states:
            return FakeTextOut(x, pooled)
        return (x,)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeImageProcessor:
    def __call__(self, images, return_tensors):
        _ = return_tensors
        if isinstance(images, list):
            pix = torch.stack(images, dim=0)
        else:
            pix = images
        pix = F.interpolate(pix, size=(32, 32), mode="bilinear", align_corners=False)
        return SimpleNamespace(pixel_values=pix)


class FakeVisionEncoder(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.conv = nn.Conv2d(3, hidden_size, 1)

    def forward(self, x, output_hidden_states=False):
        feat = self.conv(x).flatten(2).transpose(1, 2)
        if output_hidden_states:
            return SimpleNamespace(hidden_states=[feat, feat, feat])
        return SimpleNamespace(last_hidden_state=feat)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeUNet(nn.Module):
    def __init__(self, in_channels=9):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.config = _Cfg(cross_attention_dim=64, in_channels=in_channels, addition_embed_type="text")
        self.encoder_hid_proj = nn.Identity()

    def forward(self, x, timesteps, encoder_hidden_states, added_cond_kwargs=None):
        _ = timesteps, encoder_hidden_states, added_cond_kwargs
        return SimpleNamespace(sample=self.conv_in(x))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(in_channels=9)


def main():
    mod_path = os.path.join(os.path.dirname(__file__), "train_idm_vton_local.py")
    idm = load_module(mod_path, "idm_train_local")

    idm.AutoencoderKL = FakeVAE
    idm.DDPMScheduler = FakeScheduler
    idm.UNet2DConditionModel = FakeUNet
    idm.CLIPTokenizer = FakeTokenizer
    idm.CLIPTextModel = FakeTextEncoder
    idm.CLIPTextModelWithProjection = FakeTextEncoder
    idm.CLIPImageProcessor = FakeImageProcessor
    idm.CLIPVisionModelWithProjection = FakeVisionEncoder

    args = SimpleNamespace(
        pretrained_model_name_or_path="dummy",
        pretrained_garmentnet_path="dummy",
        image_encoder_path="dummy",
        num_tokens=8,
    )
    device = torch.device("cpu")
    model = idm.IDMVTONModel(args).to(device)
    model.train()

    b, h, w = 2, 64, 64
    person = torch.randn(b, 3, h, w, device=device)
    cloth = torch.randn(b, 3, h, w, device=device)
    gt = torch.randn(b, 3, h, w, device=device)

    target_lat = model.encode(gt)
    person_lat = model.encode(person)
    pose_lat = model.encode(person)
    cloth_lat = model.encode(cloth)
    person_mask = torch.zeros(
        person.shape[0], 1, person.shape[2], person.shape[3],
        device=person.device, dtype=person.dtype
    )
    person_mask = F.interpolate(person_mask, size=target_lat.shape[-2:], mode="bilinear", align_corners=False)

    noise = torch.randn_like(target_lat)
    timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (b,), device=device).long()
    noisy = model.scheduler.add_noise(target_lat, noise, timesteps)
    captions = ["model is wearing a garment"] * b
    cloth_captions = ["a photo of a garment"] * b

    pred = model(noisy, person_mask, person_lat, pose_lat, cloth, cloth_lat, timesteps, captions, cloth_captions)
    assert pred.shape == noise.shape, f"Pred shape {pred.shape} != noise shape {noise.shape}"

    target = noise if model.scheduler.config.prediction_type == "epsilon" else model.scheduler.get_velocity(target_lat, noise, timesteps)
    loss = F.mse_loss(pred, target)
    loss.backward()

    assert model.unet.conv_in.weight.grad is not None, "No grad on IDM main UNet conv"
    assert model.image_proj_model.proj_in.weight.grad is not None, "No grad through IP-Adapter resampler"
    print("IDM forward/backward PASS")


if __name__ == "__main__":
    main()
