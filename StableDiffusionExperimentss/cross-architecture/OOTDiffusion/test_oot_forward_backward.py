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
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def add_noise(self, target, noise, timesteps):
        alpha = 0.1 + (timesteps.float().view(-1, 1, 1, 1) / 1000.0)
        return target + alpha * noise

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 4, 3, padding=1)
        self.config = _Cfg(cross_attention_dim=64, in_channels=4)

    def forward(self, x, timesteps, encoder_hidden_states, added_cond_kwargs=None):
        _ = timesteps, encoder_hidden_states, added_cond_kwargs
        return SimpleNamespace(sample=self.conv_in(x))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def main():
    mod_path = os.path.join(os.path.dirname(__file__), "train_ootdiffusion_local.py")
    oot = load_module(mod_path, "oot_train_local")

    oot.AutoencoderKL = FakeVAE
    oot.DDPMScheduler = FakeScheduler
    oot.UNet2DConditionModel = FakeUNet

    device = torch.device("cpu")
    model = oot.OOTDiffusionModel(model_name="dummy", outfitting_dropout=0.0).to(device)
    model.train()

    b, h, w = 2, 64, 64
    person = torch.randn(b, 3, h, w, device=device)
    cloth = torch.randn(b, 3, h, w, device=device)
    gt = torch.randn(b, 3, h, w, device=device)

    target_lat = model.encode(gt)
    person_lat = model.encode(person)
    cloth_lat = model.encode(cloth)

    noise = torch.randn_like(target_lat)
    timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (b,), device=device).long()
    noisy = model.scheduler.add_noise(target_lat, noise, timesteps)

    pred = model(noisy, person_lat, cloth_lat, timesteps)
    assert pred.shape == noise.shape, f"Pred shape {pred.shape} != noise shape {noise.shape}"

    loss = F.mse_loss(pred, noise)
    loss.backward()

    denoise_grad = model.denoising_unet.conv_in.weight.grad
    outfit_grad = model.outfit_adapter.weight.grad
    assert denoise_grad is not None, "No gradient on denoising UNet"
    assert outfit_grad is not None, "No gradient on outfit adapter"
    print("OOT forward/backward PASS")


if __name__ == "__main__":
    main()
