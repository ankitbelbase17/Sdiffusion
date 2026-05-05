import re

NEW_CODE = """class OOTAttentionProcessor:
    def __init__(self):
        self.store = {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Fallback to the default inner mechanism (Diffusers uses scaled dot product)
        is_cross_attention = encoder_hidden_states is not None
        
        # OOTDiffusion Paper: "we concatenate them in the spatial domain as: x_gn = x_n Ⓢ c Ⓢ g_n.
        # And we replace x_n with the concatenated feature map x_gn as the input to the self-attention layer"
        # Since spatial domain in Diffusers attention is dim=1 (B, SeqLen, Dim), we concat along dim=1
        
        # Outfitting branch: save state
        if getattr(attn, "is_outfitting", False):
            if not is_cross_attention: # Only for self-attention
                self.store[attn._oot_name] = hidden_states
            # return normal forward
            return self._forward(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            
        # Denoising branch: apply fusion
        if not getattr(attn, "is_outfitting", False) and not is_cross_attention:
            g_n = self.store.get(attn._oot_name)
            if g_n is not None:
                # Concatenate sequence length / spatial domain
                orig_len = hidden_states.shape[1]
                joined_hidden_states = torch.cat([hidden_states, g_n], dim=1)
                
                # Expand attention mask if provided
                joined_mask = attention_mask
                if attention_mask is not None:
                    joined_mask = torch.cat([attention_mask, attention_mask], dim=-1)
                
                out = self._forward(attn, joined_hidden_states, encoder_hidden_states, joined_mask, **kwargs)
                return out[:, :orig_len] # Crop first half
                
        return self._forward(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

    def _forward(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class OOTDiffusionModel(nn.Module):
    def __init__(self, model_name, outfitting_dropout=0.1):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.denoising_unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.outfitting_unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.vae.requires_grad_(False)
        self.outfitting_dropout = outfitting_dropout
        
        # Expand denoising UNet conv_in to 8 channels (noisy_latent + person_latent)
        # We initialize the extra 4 channels with zeros
        orig_conv_in = self.denoising_unet.conv_in
        self.denoising_unet.conv_in = nn.Conv2d(8, orig_conv_in.out_channels, orig_conv_in.kernel_size, orig_conv_in.stride, orig_conv_in.padding)
        with torch.no_grad():
            self.denoising_unet.conv_in.weight[:, :4] = orig_conv_in.weight
            self.denoising_unet.conv_in.weight[:, 4:] = torch.zeros_like(orig_conv_in.weight)
            self.denoising_unet.conv_in.bias = orig_conv_in.bias
            
        dim = self.denoising_unet.config.cross_attention_dim
        self._cross_attention_dim = int(dim[0] if isinstance(dim, (tuple, list)) else dim)
        
        self.processor = OOTAttentionProcessor()
        self._setup_attention(self.denoising_unet, is_outfitting=False)
        self._setup_attention(self.outfitting_unet, is_outfitting=True)
        
    def _setup_attention(self, unet, is_outfitting):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            attn_procs[name] = self.processor
        unet.set_attn_processor(attn_procs)
        
        # Tag attention modules so processor knows their role
        for name, module in unet.named_modules():
            if module.__class__.__name__ == "Attention":
                module.is_outfitting = is_outfitting
                module._oot_name = name

    def encode(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def empty_text(self, batch_size, device, dtype):
        return torch.zeros(batch_size, 77, self._cross_attention_dim, device=device, dtype=dtype)

    def forward(self, noisy_target, person_lat, cloth_lat, timesteps):
        text = self.empty_text(noisy_target.shape[0], noisy_target.device, noisy_target.dtype)

        if self.training:
            import torch.distributed as _dist
            skip_flag = torch.zeros(1, device=noisy_target.device)
            if not _dist.is_available() or not _dist.is_initialized() or _dist.get_rank() == 0:
                skip_flag.fill_(1.0 if random.random() < self.outfitting_dropout else 0.0)
            if _dist.is_available() and _dist.is_initialized():
                _dist.broadcast(skip_flag, src=0)
            use_dropout = skip_flag.item() > 0.5
        else:
            use_dropout = False

        if use_dropout:
            # Drop input cloth latent entirely
            cloth_lat = torch.zeros_like(cloth_lat)
            
        # 1. Run outfitting UNet (populates OOTAttentionProcessor.store)
        self.outfitting_unet(cloth_lat, timesteps, text)
        
        # 2. Run denoising UNet (fetches from store automatically)
        # Note: We replaced additive fusion with concatenating person_lat in conv_in
        model_input = torch.cat([noisy_target, person_lat], dim=1)
        return self.denoising_unet(model_input, timesteps, text).sample
"""

TARGETS = [
    "c:/Users/acer/OneDrive/Desktop/sdiffusion/StableDiffusionExperimentss/cross-architecture/OOTDiffusion/train_ootdiffusion_local.py",
    "c:/Users/acer/OneDrive/Desktop/sdiffusion/StableDiffusionExperimentss/cross-architecture/OOTDiffusion/train_ootdiffusion_mask_local.py",
]

for target in TARGETS:
    with open(target, 'r', encoding='utf-8') as f:
        src = f.read()

    # Replace OOTDiffusionModel class
    start = src.find("class OOTDiffusionModel")
    if start == -1:
        print("Model class not found in", target)
        continue
    
    end = src.find("_FC_MC_RE =" if "train_ootdiffusion_local" in target else "def train(args):")
    
    # Needs to end just after the forward method
    method_end = src.find("return self.denoising_unet(", start)
    method_block_end = src.find("\n", method_end) + 1
    
    new_src = src[:start] + NEW_CODE + "\n\n" + src[end:]
    
    # Fix sample_tryon
    sample_regex = r"(def _sample_tryon.*?:)(.*?)(?=\s*optimizer\.zero_grad)"
    
    sample_tryon_replacement = r'''\1
        latents = torch.randn_like(person_lat)
        raw_model.scheduler.set_timesteps(n_steps, device=latents.device)
        text = raw_model.empty_text(1, latents.device, latents.dtype)

        for t in raw_model.scheduler.timesteps:
            t_batch = torch.full((latents.shape[0],), int(t), device=latents.device, dtype=torch.long)
            
            # Classifier-free guidance (sg = 2.0)
            # 1. Unconditional pass (cloth = zeros)
            noise_pred_uncond = raw_model(latents, person_lat, torch.zeros_like(cloth_lat), t_batch)
            
            # 2. Conditional pass (cloth = clothed)
            noise_pred_cond = raw_model(latents, person_lat, cloth_lat, t_batch)
            
            # Combine
            noise_pred = noise_pred_uncond + 2.0 * (noise_pred_cond - noise_pred_uncond)
            
            latents = raw_model.scheduler.step(noise_pred, t, latents).prev_sample
            
        return raw_model.vae.decode(latents / raw_model.vae.config.scaling_factor).sample
'''
    new_src = re.sub(sample_regex, sample_tryon_replacement, new_src, flags=re.DOTALL)
    
    # Remove outfit_adapter parameter references
    new_src = new_src.replace(' + list(raw_model.outfit_adapter.parameters())', '')
    new_src = re.sub(r' +"?outfit_adapter_state_dict"?.*?\n', '', new_src)

    with open(target, 'w', encoding='utf-8') as f:
        f.write(new_src)
    
    print(f"Updated {target}")
