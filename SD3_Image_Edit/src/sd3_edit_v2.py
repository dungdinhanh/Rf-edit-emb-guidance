"""
SD3 Image Editing — v2 using diffusers pipeline directly.

Uses SD3's built-in scheduler and CFG for correct flow matching.
Embedding guidance modifies the prompt embeddings before calling the pipeline.
"""

import os
import time
import argparse
import json

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


class SD3EditorV2:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda", dtype=torch.float16, offload=False):
        self.device = torch.device(device)
        self.dtype = dtype
        self.offload = offload

        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if offload:
            # Use the GPU specified by CUDA_VISIBLE_DEVICES
            gpu_id = 0  # After CUDA_VISIBLE_DEVICES remapping, always cuda:0
            self.pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        else:
            self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("SD3 loaded.")

    def edit_cfg(self, source_image, source_prompt, target_prompt,
                 num_steps=25, guidance_scale=7.0, strength=0.7):
        """Standard img2img with CFG (2 forward passes per step)."""
        t0 = time.time()

        # Resize to multiple of 16
        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        # Pre-encode prompts (same as emb_guidance for fair timing)
        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt,
            prompt_2=target_prompt,
            prompt_3=target_prompt,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=self.pipe._execution_device,
        )

        result = self.pipe(
            prompt_embeds=cond_embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=cond_pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            image=source_image,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        elapsed = time.time() - t0
        return result.images[0], elapsed

    def edit_emb_guidance(self, source_image, source_prompt, target_prompt,
                          num_steps=25, guidance_scale=7.0, strength=0.7,
                          emb_alpha=0.5):
        """Embedding guidance: blend cond/uncond embeddings, single forward pass."""
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        # Encode prompts on the pipeline's device
        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt,
            prompt_2=target_prompt,
            prompt_3=target_prompt,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=self.pipe._execution_device,
        )

        # Blend embeddings
        guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * neg_embeds
        guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * neg_pooled

        # Run pipeline with guided embeddings and NO CFG (guidance_scale=1.0 disables CFG)
        result = self.pipe(
            prompt_embeds=guided_embeds,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=guided_pooled,
            negative_pooled_prompt_embeds=None,
            image=source_image,
            strength=strength,
            num_inference_steps=num_steps,
            guidance_scale=1.0,  # 1.0 = no CFG (do_classifier_free_guidance=False)
        )
        elapsed = time.time() - t0
        return result.images[0], elapsed

    def edit(self, source_image, source_prompt, target_prompt,
             mode="cfg", num_steps=25, guidance_scale=7.0,
             emb_alpha=0.5, strength=0.7):
        if mode == "cfg":
            return self.edit_cfg(source_image, source_prompt, target_prompt,
                                num_steps, guidance_scale, strength)
        elif mode == "emb_guidance":
            return self.edit_emb_guidance(source_image, source_prompt, target_prompt,
                                         num_steps, guidance_scale, strength, emb_alpha)
        else:
            raise ValueError(f"Unknown mode: {mode}")
