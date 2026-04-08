"""
SD3 Three-Stage Text-to-Image Generation with Normalized Embedding Guidance.

Same as sd3_gen_3stage but uses L2-normalized embedding guidance to allow
higher alpha values without out-of-distribution collapse.
"""

import os
import time
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline


def norm_emb_guidance(cond, uncond, alpha):
    """L2-normalized embedding guidance."""
    cond_f = cond.float()
    uncond_f = uncond.float()
    v = (1.0 + alpha) * cond_f - alpha * uncond_f
    cond_norm = cond_f.norm(dim=-1, keepdim=True)
    v_norm = v.norm(dim=-1, keepdim=True) + 1e-8
    return (v * (cond_norm / v_norm)).to(cond.dtype)


class SD3ThreeStageNormGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda"):
        self.device = torch.device(device)
        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD3 loaded.")

    @torch.no_grad()
    def generate(self, prompt, num_steps=25, cfg_scale=7.0,
                 cfg_range=(0.3, 0.8), emb_range=(0.8, 1.0),
                 emb_alpha=0.5, emb_early_range=None, emb_early_alpha=None,
                 height=512, width=512, seed=42):
        t0 = time.time()
        device = self.pipe._execution_device

        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=device,
        )

        # Normalized guided embeddings (late)
        guided_embeds = norm_emb_guidance(cond_embeds, neg_embeds, emb_alpha)
        guided_pooled = norm_emb_guidance(cond_pooled, neg_pooled, emb_alpha)

        # Early emb (separate alpha)
        ea = emb_early_alpha if emb_early_alpha is not None else emb_alpha
        guided_early_embeds = norm_emb_guidance(cond_embeds, neg_embeds, ea)
        guided_early_pooled = norm_emb_guidance(cond_pooled, neg_pooled, ea)

        # Initial noise
        generator = torch.Generator(device=device).manual_seed(seed)
        num_channels = self.pipe.transformer.config.in_channels
        latents = torch.randn(
            1, num_channels, height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor,
            generator=generator, device=device, dtype=torch.bfloat16,
        )

        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        total_steps = len(timesteps)

        if hasattr(self.pipe.scheduler, 'init_noise_sigma'):
            latents = latents * self.pipe.scheduler.init_noise_sigma

        cfg_start = int(total_steps * cfg_range[0])
        cfg_end = int(total_steps * cfg_range[1])
        emb_start = int(total_steps * emb_range[0])
        emb_end = int(total_steps * emb_range[1])

        if emb_early_range is not None:
            emb_early_start = int(total_steps * emb_early_range[0])
            emb_early_end = int(total_steps * emb_early_range[1])
        else:
            emb_early_start, emb_early_end = 0, 0

        step_modes = []
        for i in range(total_steps):
            if cfg_start <= i < cfg_end:
                step_modes.append("cfg")
            elif emb_early_start <= i < emb_early_end:
                step_modes.append("emb_early")
            elif emb_start <= i < emb_end:
                step_modes.append("emb")
            else:
                step_modes.append("none")

        cfg_count = step_modes.count("cfg")
        emb_count = step_modes.count("emb")
        emb_early_count = step_modes.count("emb_early")
        none_count = step_modes.count("none")

        for i, t in enumerate(timesteps):
            mode = step_modes[i]
            timestep = t.expand(latents.shape[0])

            if mode == "cfg":
                latent_input = torch.cat([latents] * 2)
                timestep_input = t.expand(latent_input.shape[0])
                prompt_input = torch.cat([neg_embeds, cond_embeds], dim=0)
                pooled_input = torch.cat([neg_pooled, cond_pooled], dim=0)
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_input, timestep=timestep_input,
                    encoder_hidden_states=prompt_input,
                    pooled_projections=pooled_input,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            elif mode == "emb_early":
                noise_pred = self.pipe.transformer(
                    hidden_states=latents, timestep=timestep,
                    encoder_hidden_states=guided_early_embeds,
                    pooled_projections=guided_early_pooled,
                    return_dict=False,
                )[0]

            elif mode == "emb":
                noise_pred = self.pipe.transformer(
                    hidden_states=latents, timestep=timestep,
                    encoder_hidden_states=guided_embeds,
                    pooled_projections=guided_pooled,
                    return_dict=False,
                )[0]

            else:
                noise_pred = self.pipe.transformer(
                    hidden_states=latents, timestep=timestep,
                    encoder_hidden_states=cond_embeds,
                    pooled_projections=cond_pooled,
                    return_dict=False,
                )[0]

            latents_dtype = latents.dtype
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        elapsed = time.time() - t0
        return image, elapsed, {"cfg_steps": cfg_count, "emb_steps": emb_count,
                                "emb_early_steps": emb_early_count,
                                "none_steps": none_count, "total_steps": total_steps}
