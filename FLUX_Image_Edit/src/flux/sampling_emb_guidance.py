"""
Embedding-level guidance for FLUX image editing.

Instead of relying solely on the model's internal guidance distillation,
we blend conditional and unconditional text embeddings before each forward pass:
    guided_txt = (1 + alpha) * cond_txt - alpha * uncond_txt
    guided_vec = (1 + alpha) * cond_vec - alpha * uncond_vec

This provides an additional guidance mechanism on top of the distilled guidance.
"""

import math
from typing import Callable

import torch
from einops import rearrange
from torch import Tensor

from .model import Flux


def denoise_emb_guidance(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    cond_txt: Tensor,
    uncond_txt: Tensor,
    txt_ids: Tensor,
    cond_vec: Tensor,
    uncond_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse: bool,
    info: dict,
    guidance: float = 4.0,
    emb_guidance_alpha: float = 1.0,
):
    """Denoise with embedding-level guidance.

    Args:
        cond_txt: Conditional T5 text embeddings [B, seq, 4096]
        uncond_txt: Unconditional T5 text embeddings [B, seq, 4096]
        cond_vec: Conditional CLIP pooled embeddings [B, 768]
        uncond_vec: Unconditional CLIP pooled embeddings [B, 768]
        emb_guidance_alpha: Guidance strength. 0 = use cond only, 1 = standard blend.
    """
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # Apply embedding-level guidance (blend cond and uncond)
    alpha = emb_guidance_alpha
    if not inverse and alpha > 0:
        guided_txt = (1.0 + alpha) * cond_txt - alpha * uncond_txt
        guided_vec = (1.0 + alpha) * cond_vec - alpha * uncond_vec
    else:
        # During inversion, use cond embeddings only (no guidance)
        guided_txt = cond_txt
        guided_vec = cond_vec

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=guided_txt,
            txt_ids=txt_ids,
            y=guided_vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=guided_txt,
            txt_ids=txt_ids,
            y=guided_vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info
