"""
SD3 Pure Per-Layer CFG — no standard CFG at all.

Applies per-layer CFG at every denoising step, with the SCALE varying
both across layers (schedule) AND across timesteps (timestep_schedule).

This tests whether we can match standard CFG quality by tuning both
dimensions — which layers and which timesteps get strong guidance.
"""

import os
import time
import math
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline


def make_layer_schedule(schedule, base_scale, num_layers=24):
    if schedule == "uniform":
        return [base_scale] * num_layers
    elif schedule == "linear_up":
        return [base_scale * i / (num_layers - 1) for i in range(num_layers)]
    elif schedule == "peak_mid":
        mid = (num_layers - 1) / 2
        sigma = num_layers / 4
        return [base_scale * math.exp(-0.5 * ((i - mid) / sigma) ** 2) for i in range(num_layers)]
    elif schedule == "peak_mid_narrow":
        mid = (num_layers - 1) / 2
        sigma = num_layers / 6
        return [base_scale * math.exp(-0.5 * ((i - mid) / sigma) ** 2) for i in range(num_layers)]
    else:
        return [base_scale] * num_layers


def make_timestep_multiplier(t_schedule, step_idx, total_steps):
    """Returns a multiplier for the layer scales at this timestep."""
    frac = step_idx / max(total_steps - 1, 1)  # 0=first, 1=last step
    if t_schedule == "constant":
        return 1.0
    elif t_schedule == "linear_up":
        # Stronger guidance in later steps
        return 0.5 + 0.5 * frac
    elif t_schedule == "linear_down":
        # Stronger guidance in early steps
        return 1.5 - frac
    elif t_schedule == "strong_early":
        # Strong at start, tapering off
        return max(0.2, 2.0 * (1.0 - frac))
    elif t_schedule == "peak_mid":
        # Bell curve over timesteps
        return 0.3 + 1.4 * math.exp(-0.5 * ((frac - 0.5) / 0.2) ** 2)
    elif t_schedule == "strong_early_mid":
        # Strong 0-60%, then taper
        if frac < 0.6:
            return 1.5
        else:
            return 1.5 * (1.0 - (frac - 0.6) / 0.4)
    elif t_schedule == "ramp_up":
        # 0→max over the trajectory
        return 2.0 * frac
    else:
        return 1.0


class SD3PureLayerCFGEditor:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda", task="edit"):
        self.device = torch.device(device)
        self.task = task
        print(f"Loading SD3 pipeline ({task})...")
        if task == "edit":
            self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        else:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.num_layers = len(self.pipe.transformer.transformer_blocks)
        print(f"SD3 loaded. {self.num_layers} layers.")

    def _run_layercfg_step(self, latents, timestep, cond_embeds, neg_embeds,
                            cond_pooled, layer_scales):
        transformer = self.pipe.transformer
        hidden_states = latents
        height, width = hidden_states.shape[-2:]
        hidden_states = transformer.pos_embed(hidden_states)
        temb = transformer.time_text_embed(timestep, cond_pooled)
        enc_cond = transformer.context_embedder(cond_embeds)
        enc_uncond = transformer.context_embedder(neg_embeds)

        for idx, block in enumerate(transformer.transformer_blocks):
            scale_i = layer_scales[idx]
            if block.context_pre_only:
                _, hidden_states = block(hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond, temb=temb)
            elif scale_i == 1.0:
                enc_cond, hidden_states = block(hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond, temb=temb)
            else:
                enc_cond_out, hidden_cond = block(hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond, temb=temb)
                enc_uncond_out, hidden_uncond = block(hidden_states=hidden_states,
                    encoder_hidden_states=enc_uncond, temb=temb)
                hidden_states = hidden_uncond + scale_i * (hidden_cond - hidden_uncond)
                enc_cond = enc_cond_out
                enc_uncond = enc_uncond_out

        hidden_states = transformer.norm_out(hidden_states, temb)
        hidden_states = transformer.proj_out(hidden_states)
        patch_size = transformer.config.patch_size
        h_out = height // patch_size
        w_out = width // patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], h_out, w_out,
            patch_size, patch_size, transformer.out_channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            hidden_states.shape[0], transformer.out_channels,
            h_out * patch_size, w_out * patch_size)

    @torch.no_grad()
    def run(self, source_image=None, target_prompt="",
            num_steps=25, strength=0.7,
            base_scale=5.0, layer_schedule="peak_mid",
            t_schedule="constant", seed=42):
        t0 = time.time()
        device = self.pipe._execution_device

        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True, device=device)

        base_layer_scales = make_layer_schedule(layer_schedule, base_scale, self.num_layers)

        if self.task == "edit":
            w, h = source_image.size
            w, h = w - w % 16, h - h % 16
            source_image = source_image.resize((w, h))
            image_tensor = self.pipe.image_processor.preprocess(source_image)
            image_tensor = image_tensor.to(device=device, dtype=self.pipe.vae.dtype)
            latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
            latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            latents = latents.to(dtype=cond_embeds.dtype)
            self.pipe.scheduler.set_timesteps(num_steps, device=device)
            timesteps, _ = self.pipe.get_timesteps(num_steps, strength, device)
            noise = torch.randn_like(latents)
            latents = self.pipe.scheduler.scale_noise(latents, timesteps[:1], noise)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
            num_channels = self.pipe.transformer.config.in_channels
            latents = torch.randn(1, num_channels, 512 // self.pipe.vae_scale_factor,
                512 // self.pipe.vae_scale_factor, generator=generator, device=device,
                dtype=torch.bfloat16)
            self.pipe.scheduler.set_timesteps(num_steps, device=device)
            timesteps = self.pipe.scheduler.timesteps

        total_steps = len(timesteps)

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            t_mult = make_timestep_multiplier(t_schedule, i, total_steps)
            step_scales = [s * t_mult for s in base_layer_scales]

            noise_pred = self._run_layercfg_step(
                latents, timestep, cond_embeds, neg_embeds,
                cond_pooled, step_scales)

            latents_dtype = latents.dtype
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        elapsed = time.time() - t0
        return image, elapsed
