"""
SD3 Hybrid Editing: Embedding Guidance + Selective CFG.

Most timesteps use embedding guidance (1 forward pass, fast).
k consecutive timesteps use standard CFG (2 forward passes, sharp).

This combines the speed of embedding guidance with the sharpening of CFG.
"""

import os
import time
import json

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


class SD3HybridEditor:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda", offload=False):
        self.device = torch.device(device)
        self.offload = offload

        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if offload:
            self.pipe.enable_sequential_cpu_offload(gpu_id=0)
        else:
            # Only move transformer and VAE to GPU, keep text encoders on CPU
            self.pipe.transformer.to(self.device)
            self.pipe.vae.to(self.device)
            # Text encoders stay on CPU - we'll move them temporarily for encode_prompt
        self.pipe.set_progress_bar_config(disable=True)
        self._text_encoders_offloaded = True  # Start offloaded
        print("SD3 loaded.")

    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7,
             emb_alpha=0.3, cfg_scale=7.0, cfg_steps=3, cfg_position="last"):
        """
        Hybrid editing with embedding guidance + selective CFG.

        Args:
            emb_alpha: Embedding guidance strength for non-CFG steps
            cfg_scale: CFG scale for CFG steps
            cfg_steps: Number of steps to use CFG (k)
            cfg_position: Where to apply CFG steps - "last", "first", or "middle"
        """
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        device = self.pipe._execution_device

        # Move text encoders to GPU for encoding, then back to CPU
        if not self.offload:
            # Temporarily move text encoders to GPU
            self.pipe.text_encoder.to(self.device)
            self.pipe.text_encoder_2.to(self.device)
            if self.pipe.text_encoder_3 is not None:
                self.pipe.text_encoder_3.to(self.device)

        with torch.no_grad():
            (cond_embeds, neg_embeds,
             cond_pooled, neg_pooled) = self.pipe.encode_prompt(
                prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
                negative_prompt="", negative_prompt_2="", negative_prompt_3="",
                do_classifier_free_guidance=True,
                device=device,
            )

        if not self.offload:
            self.pipe.text_encoder.cpu()
            self.pipe.text_encoder_2.cpu()
            if self.pipe.text_encoder_3 is not None:
                self.pipe.text_encoder_3.cpu()
            import gc; gc.collect()
            torch.cuda.empty_cache()

        # Prepare guided embeddings for emb guidance steps
        guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * neg_embeds
        guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * neg_pooled

        # Encode image to latents
        image_tensor = self.pipe.image_processor.preprocess(source_image)
        image_tensor = image_tensor.to(device=device, dtype=self.pipe.vae.dtype)
        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = latents.to(dtype=cond_embeds.dtype)

        # Setup scheduler and timesteps
        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps, num_inference_steps = self.pipe.get_timesteps(num_steps, strength, device)
        latent_timestep = timesteps[:1]

        # Add noise
        noise = torch.randn_like(latents)
        latents = self.pipe.scheduler.scale_noise(latents, latent_timestep, noise)

        total_steps = len(timesteps)

        # Determine which steps use CFG
        cfg_steps = min(cfg_steps, total_steps)
        if cfg_position == "last":
            cfg_step_indices = set(range(total_steps - cfg_steps, total_steps))
        elif cfg_position == "first":
            cfg_step_indices = set(range(cfg_steps))
        elif cfg_position == "middle":
            start = (total_steps - cfg_steps) // 2
            cfg_step_indices = set(range(start, start + cfg_steps))
        else:
            raise ValueError(f"Unknown cfg_position: {cfg_position}")

        # Custom denoising loop
        for i, t in enumerate(timesteps):
            use_cfg = i in cfg_step_indices

            if use_cfg:
                # CFG: 2 sequential forward passes (saves memory vs batch=2)
                timestep = t.expand(latents.shape[0])

                noise_pred_uncond = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=neg_embeds,
                    pooled_projections=neg_pooled,
                    return_dict=False,
                )[0]

                noise_pred_cond = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=cond_embeds,
                    pooled_projections=cond_pooled,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # Embedding guidance: 1 forward pass
                timestep = t.expand(latents.shape[0])

                noise_pred = self.pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=guided_embeds,
                    pooled_projections=guided_pooled,
                    return_dict=False,
                )[0]

            # Scheduler step
            latents_dtype = latents.dtype
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        # Decode
        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        # Note: text encoders stay on CPU. Caller must pre-encode prompts
        # or call edit() which handles encoding before offload.

        elapsed = time.time() - t0
        return image, elapsed
