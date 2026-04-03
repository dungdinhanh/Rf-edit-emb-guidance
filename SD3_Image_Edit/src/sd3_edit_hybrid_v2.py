"""
SD3 Hybrid Editing v2: Single-trajectory switching.

Uses enable_model_cpu_offload() for memory management.
Switches between emb guidance and CFG within a single denoising trajectory.
"""

import os
import time
import gc

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


class SD3HybridEditorV2:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda", offload=True):
        self.device = torch.device(device)

        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        # Always use model_cpu_offload for memory safety
        self.pipe.enable_model_cpu_offload(gpu_id=0)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD3 loaded.")

    @torch.no_grad()
    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7,
             emb_alpha=0.3, cfg_scale=7.0, cfg_steps=3):
        """
        Single-trajectory hybrid: emb guidance for early steps, CFG for last k.
        """
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        device = self.pipe._execution_device

        # Encode prompts
        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=device,
        )

        # Prepare guided embeddings
        guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * neg_embeds
        guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * neg_pooled

        # Encode image
        image_tensor = self.pipe.image_processor.preprocess(source_image)
        image_tensor = image_tensor.to(device=device, dtype=self.pipe.vae.dtype)
        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = latents.to(dtype=cond_embeds.dtype)

        # Free VAE from GPU
        self.pipe.vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # Setup timesteps
        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps, num_inference_steps = self.pipe.get_timesteps(num_steps, strength, device)

        # Add noise
        noise = torch.randn_like(latents)
        latents = self.pipe.scheduler.scale_noise(latents, timesteps[:1], noise)

        total_steps = len(timesteps)
        cfg_steps = min(cfg_steps, total_steps)

        # Last cfg_steps use CFG, rest use emb guidance
        cfg_start = total_steps - cfg_steps

        # Move transformer to GPU
        self.pipe.transformer.to(device)

        for i, t in enumerate(timesteps):
            use_cfg = (i >= cfg_start)

            if use_cfg:
                # Sequential CFG: uncond pass then cond pass
                timestep = t.expand(latents.shape[0])

                noise_pred_uncond = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=neg_embeds.to(device),
                    pooled_projections=neg_pooled.to(device),
                    return_dict=False,
                )[0]

                noise_pred_cond = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=cond_embeds.to(device),
                    pooled_projections=cond_pooled.to(device),
                    return_dict=False,
                )[0]

                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Emb guidance: single pass
                timestep = t.expand(latents.shape[0])

                noise_pred = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=guided_embeds.to(device),
                    pooled_projections=guided_pooled.to(device),
                    return_dict=False,
                )[0]

            latents_dtype = latents.dtype
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        # Move transformer off, VAE on for decode
        self.pipe.transformer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        self.pipe.vae.to(device)
        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        self.pipe.vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        return image, elapsed
