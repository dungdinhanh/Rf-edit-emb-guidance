"""
SD3 Three-Stage Editor with Conditional Orthogonal Embedding Guidance.

Same three-stage approach but uses conditional orthogonal guidance for the
emb stages:

  For each token:
    direction = cond - uncond
    dot = direction · cond
    if dot >= 0:  # aligned, no conflict
        eff_dir = direction
    else:         # conflict
        eff_dir = direction - (dot / ||cond||²) * cond  # orthogonal only
    guided = cond + α * eff_dir
    guided = guided * (||cond|| / ||guided||)  # normalize magnitude
"""

import os
import time
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def orth_emb_guidance(cond, uncond, alpha):
    """Conditional orthogonal embedding guidance with magnitude normalization."""
    cond_f = cond.float()
    uncond_f = uncond.float()
    direction = cond_f - uncond_f

    # Per-token dot product (sign indicates alignment)
    dot = (direction * cond_f).sum(dim=-1, keepdim=True)

    # Parallel component projection
    cond_norm_sq = (cond_f * cond_f).sum(dim=-1, keepdim=True) + 1e-8
    proj_coef = dot / cond_norm_sq
    parallel = proj_coef * cond_f
    orth_dir = direction - parallel

    # Conditional: aligned → use full direction; conflict → use only orthogonal
    aligned_mask = (dot >= 0).float()
    eff_dir = aligned_mask * direction + (1.0 - aligned_mask) * orth_dir

    guided = cond_f + alpha * eff_dir

    # Normalize magnitude back to ||cond||
    guided_norm = guided.norm(dim=-1, keepdim=True) + 1e-8
    cond_norm = cond_f.norm(dim=-1, keepdim=True)
    guided = guided * (cond_norm / guided_norm)

    return guided.to(cond.dtype)


class SD3ThreeStageOrthEditor:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda"):
        self.device = torch.device(device)
        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD3 loaded.")

    @torch.no_grad()
    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7,
             emb_alpha=0.8, emb_early_alpha=None,
             cfg_scale=7.0,
             cfg_range=(0.0, 0.5), emb_range=(0.5, 1.0),
             emb_early_range=None):
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        device = self.pipe._execution_device

        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=device,
        )

        # Late emb (orthogonal)
        guided_embeds = orth_emb_guidance(cond_embeds, neg_embeds, emb_alpha)
        guided_pooled = orth_emb_guidance(cond_pooled, neg_pooled, emb_alpha)

        # Early emb (orthogonal)
        ea = emb_early_alpha if emb_early_alpha is not None else emb_alpha
        guided_early_embeds = orth_emb_guidance(cond_embeds, neg_embeds, ea)
        guided_early_pooled = orth_emb_guidance(cond_pooled, neg_pooled, ea)

        # Encode image
        image_tensor = self.pipe.image_processor.preprocess(source_image)
        image_tensor = image_tensor.to(device=device, dtype=self.pipe.vae.dtype)
        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = latents.to(dtype=cond_embeds.dtype)

        # Setup timesteps
        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps, num_inference_steps = self.pipe.get_timesteps(num_steps, strength, device)

        noise = torch.randn_like(latents)
        latents = self.pipe.scheduler.scale_noise(latents, timesteps[:1], noise)

        total_steps = len(timesteps)
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
