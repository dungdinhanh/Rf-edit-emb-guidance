"""
SD3 Three-Stage Hybrid Editing (single trajectory).

Three guidance modes applied at different timestep ranges:
1. No guidance (cond only) — preserves source structure
2. CFG (cond + uncond, prediction-space) — strong editing signal
3. Embedding guidance (blended embeddings) — smooth refinement

Each step's mode is determined by which range it falls in.
Ranges are specified as fractions of total denoising steps.
"""

import os
import time
import gc

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


class SD3ThreeStageEditor:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda"):
        self.device = torch.device(device)

        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        # Load only transformer + VAE to GPU. Text encoders stay on CPU.
        # Same memory footprint as normal pipeline during denoising.
        self.pipe.transformer.to(self.device)
        self.pipe.vae.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD3 loaded.")

    @torch.no_grad()
    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7,
             emb_alpha=0.3, cfg_scale=7.0,
             cfg_range=(0.3, 0.7), emb_range=(0.7, 1.0)):
        """
        Three-stage editing within a single denoising trajectory.

        Args:
            cfg_range: (start_frac, end_frac) — fraction of total steps for CFG.
                       e.g., (0.3, 0.7) means CFG from step 30% to 70%.
            emb_range: (start_frac, end_frac) — fraction for emb guidance.
                       e.g., (0.7, 1.0) means emb guidance from 70% to end.
            Steps outside both ranges use no guidance (cond only).

        Stage assignment priority: if ranges overlap, CFG takes precedence.
        """
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        device = self.device

        # Move text encoders to GPU, encode, move back
        self.pipe.text_encoder.to(device)
        self.pipe.text_encoder_2.to(device)
        if self.pipe.text_encoder_3 is not None:
            self.pipe.text_encoder_3.to(device)

        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=device,
        )
        # Detach to break computation graph, then free text encoders
        cond_embeds, neg_embeds = cond_embeds.detach(), neg_embeds.detach()
        cond_pooled, neg_pooled = cond_pooled.detach(), neg_pooled.detach()

        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder_2.to("cpu")
        if self.pipe.text_encoder_3 is not None:
            self.pipe.text_encoder_3.to("cpu")
        del self.pipe.text_encoder._hf_hook  # Remove accelerate hooks that hold references
        del self.pipe.text_encoder_2._hf_hook
        if self.pipe.text_encoder_3 is not None and hasattr(self.pipe.text_encoder_3, '_hf_hook'):
            del self.pipe.text_encoder_3._hf_hook
        gc.collect()
        torch.cuda.empty_cache()

        # Prepare guided embeddings
        guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * neg_embeds
        guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * neg_pooled

        # Encode image (VAE already on GPU)
        image_tensor = self.pipe.image_processor.preprocess(source_image)
        image_tensor = image_tensor.to(device=device, dtype=self.pipe.vae.dtype)
        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = latents.to(dtype=cond_embeds.dtype)

        # Setup timesteps
        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps, num_inference_steps = self.pipe.get_timesteps(num_steps, strength, device)

        # Add noise
        noise = torch.randn_like(latents)
        latents = self.pipe.scheduler.scale_noise(latents, timesteps[:1], noise)

        total_steps = len(timesteps)

        # Convert fractional ranges to step indices
        cfg_start = int(total_steps * cfg_range[0])
        cfg_end = int(total_steps * cfg_range[1])
        emb_start = int(total_steps * emb_range[0])
        emb_end = int(total_steps * emb_range[1])

        # Build per-step mode assignment
        step_modes = []
        for i in range(total_steps):
            if cfg_start <= i < cfg_end:
                step_modes.append("cfg")
            elif emb_start <= i < emb_end:
                step_modes.append("emb")
            else:
                step_modes.append("none")

        cfg_count = step_modes.count("cfg")
        emb_count = step_modes.count("emb")
        none_count = step_modes.count("none")

        # Transformer and VAE already on GPU
        for i, t in enumerate(timesteps):
            mode = step_modes[i]
            timestep = t.expand(latents.shape[0])

            if mode == "cfg":
                # Sequential CFG
                noise_pred_uncond = self.pipe.transformer(
                    hidden_states=latents, timestep=timestep,
                    encoder_hidden_states=neg_embeds,
                    pooled_projections=neg_pooled,
                    return_dict=False,
                )[0]

                noise_pred_cond = self.pipe.transformer(
                    hidden_states=latents, timestep=timestep,
                    encoder_hidden_states=cond_embeds,
                    pooled_projections=cond_pooled,
                    return_dict=False,
                )[0]

                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            elif mode == "emb":
                # Embedding guidance
                noise_pred = self.pipe.transformer(
                    hidden_states=latents, timestep=timestep,
                    encoder_hidden_states=guided_embeds,
                    pooled_projections=guided_pooled,
                    return_dict=False,
                )[0]

            else:
                # No guidance — cond only
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

        # Decode (VAE already on GPU)
        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        elapsed = time.time() - t0
        return image, elapsed, {"cfg_steps": cfg_count, "emb_steps": emb_count,
                                "none_steps": none_count, "total_steps": total_steps,
                                "step_modes": step_modes}
