"""
Attention-level guidance for FLUX image editing.

Instead of pre-blending embeddings (v1), this operates at each transformer layer:
1. Compute text token importance scores from image queries
2. Find top-k overlap between cond and uncond important tokens
3. Zero overlap in uncond path to amplify editing signal
4. Blend: guided_k = (1 + alpha) * cond_k - alpha * masked_uncond_k
"""

import torch
from torch import Tensor


def compute_text_importance_scores(img_q: Tensor, txt_k: Tensor) -> Tensor:
    """Score each text token's importance to image queries.

    Args:
        img_q: Image queries [B, H, img_len, D]
        txt_k: Text keys [B, H, txt_len, D]

    Returns:
        scores: [B, txt_len] importance scores (averaged over heads and image tokens)
    """
    # Average image queries across spatial dimension: [B, H, 1, D]
    img_q_avg = img_q.mean(dim=2, keepdim=True)
    # Dot product: [B, H, 1, txt_len]
    scores = torch.matmul(img_q_avg, txt_k.transpose(-2, -1)) / (img_q.shape[-1] ** 0.5)
    # Average over heads: [B, txt_len]
    scores = scores.squeeze(2).mean(dim=1)
    return scores


def find_overlap_mask(cond_scores: Tensor, uncond_scores: Tensor, k: int) -> Tensor:
    """Find tokens that are in top-k for both cond and uncond.

    Args:
        cond_scores: [B, txt_len]
        uncond_scores: [B, txt_len]
        k: Number of top tokens to consider

    Returns:
        overlap_mask: [B, txt_len] boolean mask of overlapping tokens
    """
    txt_len = cond_scores.shape[-1]
    k = min(k, txt_len)

    _, cond_topk = torch.topk(cond_scores, k, dim=-1)
    _, uncond_topk = torch.topk(uncond_scores, k, dim=-1)

    cond_mask = torch.zeros_like(cond_scores, dtype=torch.bool)
    uncond_mask = torch.zeros_like(uncond_scores, dtype=torch.bool)

    cond_mask.scatter_(1, cond_topk, True)
    uncond_mask.scatter_(1, uncond_topk, True)

    overlap = cond_mask & uncond_mask
    return overlap


def apply_attn_guidance_to_kv(
    cond_k: Tensor, cond_v: Tensor,
    uncond_k: Tensor, uncond_v: Tensor,
    img_q: Tensor,
    k: int = 32,
    alpha: float = 1.0,
    overlap_target: str = "uncond",
) -> tuple[Tensor, Tensor]:
    """Apply attention-level guidance to text K/V.

    Args:
        cond_k: Conditional text keys [B, H, txt_len, D]
        cond_v: Conditional text values [B, H, txt_len, D]
        uncond_k: Unconditional text keys [B, H, txt_len, D]
        uncond_v: Unconditional text values [B, H, txt_len, D]
        img_q: Image queries [B, H, img_len, D]
        k: Top-k for overlap detection
        alpha: Guidance strength
        overlap_target: Which path to zero overlap ("uncond" or "cond")

    Returns:
        guided_k, guided_v: Guided text K/V [B, H, txt_len, D]
    """
    # Compute importance scores
    cond_scores = compute_text_importance_scores(img_q, cond_k)
    uncond_scores = compute_text_importance_scores(img_q, uncond_k)

    # Find overlap
    overlap = find_overlap_mask(cond_scores, uncond_scores, k)
    # Expand mask for heads and dim: [B, 1, txt_len, 1]
    mask = overlap.unsqueeze(1).unsqueeze(-1)

    if overlap_target == "uncond":
        # Zero overlapping tokens in uncond path
        masked_uncond_k = uncond_k * (~mask)
        masked_uncond_v = uncond_v * (~mask)
        guided_k = (1.0 + alpha) * cond_k - alpha * masked_uncond_k
        guided_v = (1.0 + alpha) * cond_v - alpha * masked_uncond_v
    else:
        masked_cond_k = cond_k * (~mask)
        masked_cond_v = cond_v * (~mask)
        guided_k = (1.0 + alpha) * masked_cond_k - alpha * uncond_k
        guided_v = (1.0 + alpha) * masked_cond_v - alpha * uncond_v

    return guided_k, guided_v
