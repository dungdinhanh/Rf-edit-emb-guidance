"""
SD3 Three-Stage Text-to-Image Generation (single trajectory).

Same three-stage approach as editing but for generation from noise.
Used to verify interval guidance findings: CFG in middle should be
most effective for generation (unlike editing where early CFG wins).
"""

import os
import time

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline


class SD3ThreeStageGenerator:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda", offload=False):
        self.device = torch.device(device)

        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if offload:
            self.pipe.enable_sequential_cpu_offload(gpu_id=0)
        else:
            self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD3 loaded.")

    @torch.no_grad()
    def generate(self, prompt, num_steps=25, cfg_scale=7.0,
                 cfg_range=(0.3, 0.7), emb_range=(0.7, 1.0),
                 emb_alpha=0.3, emb_early_range=None, emb_early_alpha=None,
                 height=512, width=512, seed=42):
        """Three-stage generation from pure noise.

        Supports two separate emb ranges:
        - emb_range + emb_alpha: primary emb range (typically late)
        - emb_early_range + emb_early_alpha: optional early emb range
        """
        t0 = time.time()
        device = self.pipe._execution_device

        # Encode prompts
        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=device,
        )

        # Prepare guided embeddings (primary / late)
        guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * neg_embeds
        guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * neg_pooled

        # Prepare early guided embeddings (separate alpha)
        ea = emb_early_alpha if emb_early_alpha is not None else emb_alpha
        guided_early_embeds = (1.0 + ea) * cond_embeds - ea * neg_embeds
        guided_early_pooled = (1.0 + ea) * cond_pooled - ea * neg_pooled

        # Generate initial noise
        generator = torch.Generator(device=device).manual_seed(seed)
        num_channels = self.pipe.transformer.config.in_channels
        latents = torch.randn(
            1, num_channels, height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor,
            generator=generator, device=device, dtype=torch.bfloat16,
        )

        # Setup timesteps
        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        total_steps = len(timesteps)

        # Scale initial noise (FlowMatch doesn't need scaling, but check)
        if hasattr(self.pipe.scheduler, 'init_noise_sigma'):
            latents = latents * self.pipe.scheduler.init_noise_sigma

        # Convert ranges to step indices
        cfg_start = int(total_steps * cfg_range[0])
        cfg_end = int(total_steps * cfg_range[1])
        emb_start = int(total_steps * emb_range[0])
        emb_end = int(total_steps * emb_range[1])

        # Early emb range
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
        none_count = step_modes.count("none")

        # Denoising loop
        for i, t in enumerate(timesteps):
            mode = step_modes[i]
            timestep = t.expand(latents.shape[0])

            if mode == "cfg":
                latent_input = torch.cat([latents] * 2)
                timestep_input = t.expand(latent_input.shape[0])
                prompt_input = torch.cat([neg_embeds, cond_embeds], dim=0)
                pooled_input = torch.cat([neg_pooled, cond_pooled], dim=0)

                noise_pred = self.pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep_input,
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

        # Decode
        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        elapsed = time.time() - t0
        return image, elapsed, {"cfg_steps": cfg_count, "emb_steps": emb_count,
                                "none_steps": none_count, "total_steps": total_steps}
