"""
SD3 Hybrid Editing: Embedding Guidance + Selective CFG.

Uses two sequential pipeline calls:
1. Emb guidance for most steps (1 forward pass/step, fast)
2. CFG for last k steps (2 forward passes/step, sharp)

The output of step 1 is fed as input to step 2 with reduced strength.
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
            self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD3 loaded.")

    @torch.no_grad()
    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7,
             emb_alpha=0.3, cfg_scale=7.0, cfg_steps=3):
        """
        Hybrid editing: emb guidance first, then CFG for last k steps.

        Total denoising steps = round(num_steps * strength).
        Of these, (total - cfg_steps) use emb guidance, last cfg_steps use CFG.
        """
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        # Pre-encode all prompts
        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=self.pipe._execution_device,
        )

        # Compute total denoising steps
        total_denoise_steps = max(1, round(num_steps * strength))
        emb_steps = max(0, total_denoise_steps - cfg_steps)

        if cfg_steps == 0 or emb_steps == total_denoise_steps:
            # Pure embedding guidance
            guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * neg_embeds
            guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * neg_pooled

            result = self.pipe(
                prompt_embeds=guided_embeds,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=guided_pooled,
                negative_pooled_prompt_embeds=None,
                image=source_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=1.0,
            )
            elapsed = time.time() - t0
            return result.images[0], elapsed

        if cfg_steps >= total_denoise_steps:
            # Pure CFG
            result = self.pipe(
                prompt_embeds=cond_embeds,
                negative_prompt_embeds=neg_embeds,
                pooled_prompt_embeds=cond_pooled,
                negative_pooled_prompt_embeds=neg_pooled,
                image=source_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=cfg_scale,
            )
            elapsed = time.time() - t0
            return result.images[0], elapsed

        # --- Hybrid: emb guidance first, then CFG ---
        # Step 1: Emb guidance with reduced strength (covers first portion)
        emb_strength = strength * (emb_steps / total_denoise_steps)
        guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * neg_embeds
        guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * neg_pooled

        intermediate = self.pipe(
            prompt_embeds=guided_embeds,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=guided_pooled,
            negative_pooled_prompt_embeds=None,
            image=source_image,
            strength=emb_strength,
            num_inference_steps=num_steps,
            guidance_scale=1.0,
        )

        # Step 2: CFG on intermediate result with remaining strength
        cfg_strength = strength * (cfg_steps / total_denoise_steps)
        # Clamp to valid range
        cfg_strength = max(0.01, min(0.99, cfg_strength))

        result = self.pipe(
            prompt_embeds=cond_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=cond_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            image=intermediate.images[0],
            strength=cfg_strength,
            num_inference_steps=num_steps,
            guidance_scale=cfg_scale,
        )

        elapsed = time.time() - t0
        return result.images[0], elapsed
