"""
Attention-level Embedding Guidance (v2) for HunyuanVideo.

Instead of pre-blending cond/uncond embeddings before the transformer (v1),
this applies guidance inside each attention layer by:
1. Scoring text tokens against image queries for both cond and uncond
2. Identifying top-k important tokens for each
3. Finding overlapping tokens (common to both — not editing-relevant)
4. Zeroing out overlap in the uncond path
5. Blending cond/uncond K/V with the overlap mask applied

This preserves editing-relevant tokens while suppressing shared background tokens.
"""

import torch
from typing import Tuple, Optional


def compute_text_importance_scores(
    img_q: torch.Tensor,
    txt_k: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token importance of text keys w.r.t. image queries.

    Uses averaged image query to keep memory low:
      score[t] = mean_img_q . txt_k[t] / sqrt(d)

    Args:
        img_q: [B, img_len, H, D] image queries
        txt_k: [B, txt_len, H, D] text keys

    Returns:
        importance: [B, txt_len] per-token importance scores
    """
    d = img_q.shape[-1]
    scale = 1.0 / (d ** 0.5)

    # Average over image tokens: [B, H, D]
    mean_q = img_q.mean(dim=1)

    # Dot product with each text key: [B, H, txt_len]
    scores = torch.einsum('bhd,bthd->bht', mean_q, txt_k) * scale

    # Average over heads: [B, txt_len]
    return scores.mean(dim=1)


def find_overlap_mask(
    cond_scores: torch.Tensor,
    uncond_scores: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Find tokens that appear in top-k of BOTH cond and uncond.

    Args:
        cond_scores: [B, txt_len]
        uncond_scores: [B, txt_len]
        k: number of top tokens to select

    Returns:
        overlap_mask: [B, txt_len] bool — True for overlapping tokens
    """
    txt_len = cond_scores.shape[-1]
    k = min(k, txt_len)

    _, cond_topk_idx = cond_scores.topk(k, dim=-1)
    _, uncond_topk_idx = uncond_scores.topk(k, dim=-1)

    # Convert indices to boolean masks
    cond_mask = torch.zeros_like(cond_scores, dtype=torch.bool)
    uncond_mask = torch.zeros_like(uncond_scores, dtype=torch.bool)
    cond_mask.scatter_(1, cond_topk_idx, True)
    uncond_mask.scatter_(1, uncond_topk_idx, True)

    return cond_mask & uncond_mask


def apply_attn_guidance_to_kv(
    cond_txt_k: torch.Tensor,
    cond_txt_v: torch.Tensor,
    uncond_txt_k: torch.Tensor,
    uncond_txt_v: torch.Tensor,
    img_q: torch.Tensor,
    k: int,
    alpha: float,
    overlap_target: str = "uncond",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply attention-level guidance to text K/V.

    Scores both cond and uncond text tokens, finds overlap in top-k,
    zeros out overlap in the target path, then blends K/V.

    Args:
        cond_txt_k: [B, txt_len, H, D]
        cond_txt_v: [B, txt_len, H, D]
        uncond_txt_k: [B, txt_len, H, D]
        uncond_txt_v: [B, txt_len, H, D]
        img_q: [B, img_len, H, D]
        k: top-k parameter
        alpha: guidance strength
        overlap_target: "uncond" to zero overlap in uncond, "cond" for cond

    Returns:
        (guided_txt_k, guided_txt_v): blended K/V with overlap masking
    """
    # Score both paths
    cond_scores = compute_text_importance_scores(img_q, cond_txt_k)
    uncond_scores = compute_text_importance_scores(img_q, uncond_txt_k)

    # Find overlapping top-k tokens
    overlap = find_overlap_mask(cond_scores, uncond_scores, k)

    # Expand mask for broadcasting: [B, txt_len, 1, 1]
    mask = overlap.unsqueeze(-1).unsqueeze(-1)

    if overlap_target == "uncond":
        masked_uncond_k = uncond_txt_k.masked_fill(mask, 0.0)
        masked_uncond_v = uncond_txt_v.masked_fill(mask, 0.0)
        guided_k = (1.0 + alpha) * cond_txt_k - alpha * masked_uncond_k
        guided_v = (1.0 + alpha) * cond_txt_v - alpha * masked_uncond_v
    else:
        masked_cond_k = cond_txt_k.masked_fill(mask, 0.0)
        masked_cond_v = cond_txt_v.masked_fill(mask, 0.0)
        guided_k = (1.0 + alpha) * masked_cond_k - alpha * uncond_txt_k
        guided_v = (1.0 + alpha) * masked_cond_v - alpha * uncond_txt_v

    return guided_k, guided_v


def make_double_block_hook(
    block,
    cond_txt: torch.Tensor,
    uncond_txt: torch.Tensor,
    cond_vec: torch.Tensor,
    uncond_vec: torch.Tensor,
    guidance_k: int,
    guidance_alpha: float,
    overlap_target: str,
):
    """Create an attn_kv_hook for a double block.

    In double blocks, img and txt have separate QKV projections.
    The hook computes uncond txt Q/K/V using the block's txt projections,
    then applies guidance to the txt portion of the concatenated K/V.
    """
    from einops import rearrange
    from .modulate_layers import modulate

    def hook(q, k, v, img_len, txt_len):
        # Extract img portion of q for scoring
        img_q = q[:, :img_len, :, :]

        # Extract cond txt K/V from the concatenated K/V
        cond_txt_k = k[:, img_len:, :, :]
        cond_txt_v = v[:, img_len:, :, :]

        # Compute uncond txt K/V using the block's projections
        _, _, _, txt_mod1_shift, txt_mod1_scale, _ = block.txt_mod(uncond_vec).chunk(6, dim=-1)
        uncond_txt_mod = modulate(block.txt_norm1(uncond_txt), shift=txt_mod1_shift, scale=txt_mod1_scale)
        uncond_qkv = block.txt_attn_qkv(uncond_txt_mod)
        _, uncond_k, uncond_v = rearrange(uncond_qkv, "B L (K H D) -> K B L H D", K=3, H=block.heads_num)
        uncond_k = block.txt_attn_k_norm(uncond_k).to(uncond_v)

        # Apply guidance to txt K/V
        guided_txt_k, guided_txt_v = apply_attn_guidance_to_kv(
            cond_txt_k, cond_txt_v, uncond_k, uncond_v,
            img_q, guidance_k, guidance_alpha, overlap_target,
        )

        # Replace txt portion in concatenated K/V
        k_out = torch.cat([k[:, :img_len, :, :], guided_txt_k], dim=1)
        v_out = torch.cat([v[:, :img_len, :, :], guided_txt_v], dim=1)

        return q, k_out, v_out

    return hook


def make_single_block_hook(
    block,
    uncond_txt: torch.Tensor,
    cond_vec: torch.Tensor,
    uncond_vec: torch.Tensor,
    guidance_k: int,
    guidance_alpha: float,
    overlap_target: str,
):
    """Create an attn_kv_hook for a single block.

    In single blocks, img and txt are merged. We project uncond_txt
    through the block's linear1 to get uncond K/V for the text portion.
    """
    from einops import rearrange
    from .modulate_layers import modulate

    def hook(q, k, v, img_len, txt_len):
        # Extract img portion of q for scoring
        img_q = q[:, :img_len, :, :]

        # Extract cond txt K/V from merged K/V
        cond_txt_k = k[:, img_len:, :, :]
        cond_txt_v = v[:, img_len:, :, :]

        # Project uncond_txt through this block's linear1
        uncond_mod_shift, uncond_mod_scale, _ = block.modulation(uncond_vec).chunk(3, dim=-1)
        uncond_txt_mod = modulate(block.pre_norm(uncond_txt), shift=uncond_mod_shift, scale=uncond_mod_scale)
        uncond_qkv, _ = torch.split(
            block.linear1(uncond_txt_mod),
            [3 * block.hidden_size, block.mlp_hidden_dim],
            dim=-1,
        )
        _, uncond_k, uncond_v = rearrange(uncond_qkv, "B L (K H D) -> K B L H D", K=3, H=block.heads_num)
        uncond_k = block.k_norm(uncond_k).to(uncond_v)

        # Apply guidance
        guided_txt_k, guided_txt_v = apply_attn_guidance_to_kv(
            cond_txt_k, cond_txt_v, uncond_k, uncond_v,
            img_q, guidance_k, guidance_alpha, overlap_target,
        )

        # Replace txt portion
        k_out = torch.cat([k[:, :img_len, :, :], guided_txt_k], dim=1)
        v_out = torch.cat([v[:, :img_len, :, :], guided_txt_v], dim=1)

        return q, k_out, v_out

    return hook
