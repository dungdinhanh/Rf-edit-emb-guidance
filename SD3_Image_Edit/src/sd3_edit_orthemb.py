"""
SD3 Editing with Normalized + Orthogonal Embedding Guidance.

Two-step procedure to keep guidance non-conflicting with cond direction:

  1. Compute guided vector: v = cond + α*(cond - uncond)
  2. Normalize v to magnitude ||cond||  (keep magnitude in distribution)
  3. Project v_norm onto plane orthogonal to cond (remove parallel component)
  4. Final = cond + orthogonal_component

The orthogonal component is the part of the guidance signal that doesn't
conflict with the cond direction — it adds perpendicular "rotation" guidance
without canceling or amplifying cond itself.
"""

import os
import time
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def norm_then_orth_guidance(cond, uncond, alpha):
    """
    Compute v = cond + alpha*(cond - uncond)
    Normalize to ||cond||, then project onto plane orthogonal to cond.
    Final = cond + orthogonal component.

    All ops are per-token (last dim treated as the vector dim).
    """
    cond_f = cond.float()
    uncond_f = uncond.float()

    # Step 1: standard guided vector
    v = cond_f + alpha * (cond_f - uncond_f)

    # Step 2: normalize v to per-token magnitude of cond
    cond_norm = cond_f.norm(dim=-1, keepdim=True)
    v_norm = v.norm(dim=-1, keepdim=True) + 1e-8
    v_normalized = v * (cond_norm / v_norm)

    # Step 3: project v_normalized onto plane orthogonal to cond
    # parallel component = (v · cond / ||cond||²) * cond
    cond_norm_sq = (cond_f * cond_f).sum(dim=-1, keepdim=True) + 1e-8
    proj_coef = (v_normalized * cond_f).sum(dim=-1, keepdim=True) / cond_norm_sq
    parallel = proj_coef * cond_f
    orthogonal = v_normalized - parallel

    # Step 4: final = cond + orthogonal guidance signal
    final = cond_f + orthogonal

    return final.to(cond.dtype)


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

        # Apply norm-then-orthogonal guidance
        guided_embeds = norm_then_orth_guidance(cond_embeds, neg_embeds, emb_alpha)
        guided_pooled = norm_then_orth_guidance(cond_pooled, neg_pooled, emb_alpha)

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
