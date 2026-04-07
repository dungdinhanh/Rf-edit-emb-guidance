"""
SD3 Editing with Orthogonal Embedding Guidance.

Decompose the guidance direction (cond - uncond) into parallel and
orthogonal components relative to cond. Use only the orthogonal part:

  direction = cond - uncond
  parallel  = (direction · cond / ||cond||²) · cond     # along cond
  orth_dir  = direction - parallel                       # perpendicular
  guided    = cond + α · orth_dir

The parallel component just scales cond (no new information).
The orthogonal component is the genuinely new direction the guidance brings.
By adding only α times the orthogonal direction to cond, we get pure rotation
guidance with no magnitude blowup and no conflict with cond.
"""

import os
import time
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def orth_guidance(cond, uncond, alpha, normalize=True):
    """
    Conditional orthogonal guidance:
    - If cos(direction, cond) >= 0 (no conflict): use standard guidance
        guided = cond + alpha * (cond - uncond)
        then normalize magnitude back to ||cond||
    - If cos(direction, cond) < 0 (conflict): project out the conflicting
        parallel component, use only orthogonal
        guided = cond + alpha * orth_dir

    Per-token decision (each token's vector has its own dot product).
    """
    cond_f = cond.float()
    uncond_f = uncond.float()

    direction = cond_f - uncond_f

    # Per-token dot product (sign indicates alignment)
    dot = (direction * cond_f).sum(dim=-1, keepdim=True)  # [..., 1]

    # Parallel component
    cond_norm_sq = (cond_f * cond_f).sum(dim=-1, keepdim=True) + 1e-8
    proj_coef = dot / cond_norm_sq
    parallel = proj_coef * cond_f
    orth_dir = direction - parallel

    # Two paths:
    # (a) dot >= 0: direction aligns with cond → use full direction
    # (b) dot < 0: direction conflicts → use only orthogonal
    aligned_mask = (dot >= 0).float()  # 1 where aligned, 0 where conflict

    # Effective guidance direction:
    #   aligned tokens → direction (full)
    #   conflicting tokens → orth_dir (perpendicular only)
    eff_dir = aligned_mask * direction + (1.0 - aligned_mask) * orth_dir

    # Apply guidance
    guided = cond_f + alpha * eff_dir

    if normalize:
        # Normalize per-token magnitude back to ||cond||
        guided_norm = guided.norm(dim=-1, keepdim=True) + 1e-8
        cond_norm = cond_f.norm(dim=-1, keepdim=True)
        guided = guided * (cond_norm / guided_norm)

    return guided.to(cond.dtype)


class SD3OrthEmbEditor:
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
             num_steps=25, strength=0.7, emb_alpha=0.5):
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=self.pipe._execution_device,
        )

        # Apply orthogonal guidance
        guided_embeds = orth_guidance(cond_embeds, neg_embeds, emb_alpha)
        guided_pooled = orth_guidance(cond_pooled, neg_pooled, emb_alpha)

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
