"""
SD3 Editing with Normalized Embedding Guidance.

After computing guided = (1+α)·cond - α·uncond, normalize the result to match
the statistics of the cond embeddings. This prevents out-of-distribution issues
from large alpha values.

Three normalization modes:
- "none": Standard emb guidance (no normalization)
- "norm": L2 norm matching — rescale guided to have same norm as cond
- "stats": Full statistical normalization (per-channel mean/std matching)
"""

import os
import time
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def normalize_guided(guided, cond, mode="norm"):
    """Normalize guided embedding to stay within cond's statistical distribution."""
    if mode == "none":
        return guided
    elif mode == "norm":
        # L2 norm matching: preserve direction, match magnitude
        cond_norm = cond.float().norm()
        guided_norm = guided.float().norm()
        if guided_norm > 0:
            return (guided.float() * (cond_norm / guided_norm)).to(guided.dtype)
        return guided
    elif mode == "stats":
        # Per-channel mean/std matching
        cond_mean = cond.float().mean(dim=-1, keepdim=True)
        cond_std = cond.float().std(dim=-1, keepdim=True)
        guided_mean = guided.float().mean(dim=-1, keepdim=True)
        guided_std = guided.float().std(dim=-1, keepdim=True)
        normalized = (guided.float() - guided_mean) / (guided_std + 1e-6) * cond_std + cond_mean
        return normalized.to(guided.dtype)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


class SD3NormEmbEditor:
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
             num_steps=25, strength=0.7, emb_alpha=0.5, norm_mode="norm"):
        """Edit with normalized embedding guidance."""
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        # Encode prompts
        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=self.pipe._execution_device,
        )

        # Compute raw guided embeddings
        guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * neg_embeds
        guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * neg_pooled

        # Normalize to stay in distribution
        guided_embeds = normalize_guided(guided_embeds, cond_embeds, mode=norm_mode)
        guided_pooled = normalize_guided(guided_pooled, cond_pooled, mode=norm_mode)

        # Run pipeline with normalized guided embeddings, no CFG
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
