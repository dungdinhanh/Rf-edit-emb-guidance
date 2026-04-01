"""
Attention-level guidance sampling for FLUX image editing.

At each double-stream transformer layer:
1. Compute both cond and uncond text QKV (uncond_txt propagated through layers)
2. Score text tokens against image queries
3. Find top-k overlap between cond/uncond important tokens
4. Zero overlap and blend K/V before attention

Single-stream blocks use standard forward (no guidance) since txt/img are merged
and text-only manipulation is unreliable there.
"""

import math
import os

import torch
from einops import rearrange
from torch import Tensor

from .model import Flux
from .modules.layers import timestep_embedding
from .math import attention
from .attn_guidance import apply_attn_guidance_to_kv


def flux_forward_with_attn_guidance(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Tensor,
    info: dict,
    uncond_txt: Tensor,
    uncond_y: Tensor,
    attn_k: int = 32,
    attn_alpha: float = 1.0,
    overlap_target: str = "uncond",
) -> tuple[Tensor, dict]:
    """Forward pass with attention-level guidance in double-stream blocks.

    Key fix: uncond_txt is propagated through each double block (its own
    attention + MLP path) so it stays a valid representation at every layer.
    Single-stream blocks use standard forward without guidance.
    """
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # Standard input processing
    img = model.img_in(img)
    vec = model.time_in(timestep_embedding(timesteps, 256))
    if model.params.guidance_embed:
        if guidance is None:
            raise ValueError("Didn't get guidance strength for guidance distilled model.")
        vec = vec + model.guidance_in(timestep_embedding(guidance, 256))
    vec = vec + model.vector_in(y)
    txt = model.txt_in(txt)

    # Uncond processing
    uncond_vec = model.time_in(timestep_embedding(timesteps, 256))
    if model.params.guidance_embed:
        uncond_vec = uncond_vec + model.guidance_in(timestep_embedding(guidance, 256))
    uncond_vec = uncond_vec + model.vector_in(uncond_y)
    uncond_txt_state = model.txt_in(uncond_txt)

    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = model.pe_embedder(ids)

    # ---- Double Stream Blocks (with guidance) ----
    for block in model.double_blocks:
        img_mod1, img_mod2 = block.img_mod(vec)
        txt_mod1, txt_mod2 = block.txt_mod(vec)

        # Image QKV
        img_modulated = block.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = block.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
        img_q, img_k = block.img_attn.norm(img_q, img_k, img_v)

        # Cond text QKV
        txt_modulated = block.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = block.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
        txt_q, txt_k = block.txt_attn.norm(txt_q, txt_k, txt_v)

        # Uncond text QKV (using uncond_txt_state which evolves through layers)
        uncond_txt_mod1, uncond_txt_mod2 = block.txt_mod(uncond_vec)
        uncond_txt_modulated = block.txt_norm1(uncond_txt_state)
        uncond_txt_modulated = (1 + uncond_txt_mod1.scale) * uncond_txt_modulated + uncond_txt_mod1.shift
        uncond_txt_qkv = block.txt_attn.qkv(uncond_txt_modulated)
        u_txt_q, u_txt_k, u_txt_v = rearrange(uncond_txt_qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
        u_txt_q, u_txt_k = block.txt_attn.norm(u_txt_q, u_txt_k, u_txt_v)

        # Apply attention guidance to text K/V only
        guided_txt_k, guided_txt_v = apply_attn_guidance_to_kv(
            cond_k=txt_k, cond_v=txt_v,
            uncond_k=u_txt_k, uncond_v=u_txt_v,
            img_q=img_q,
            k=attn_k, alpha=attn_alpha, overlap_target=overlap_target,
        )

        # COND attention with guided text K/V
        q = torch.cat((txt_q, img_q), dim=2)
        k_cat = torch.cat((guided_txt_k, img_k), dim=2)
        v_cat = torch.cat((guided_txt_v, img_v), dim=2)
        attn_out = attention(q, k_cat, v_cat, pe=pe)

        txt_attn, img_attn = attn_out[:, :txt.shape[1]], attn_out[:, txt.shape[1]:]

        # Update cond streams (residual connections)
        img = img + img_mod1.gate * block.img_attn.proj(img_attn)
        img = img + img_mod2.gate * block.img_mlp(
            (1 + img_mod2.scale) * block.img_norm2(img) + img_mod2.shift
        )
        txt = txt + txt_mod1.gate * block.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * block.txt_mlp(
            (1 + txt_mod2.scale) * block.txt_norm2(txt) + txt_mod2.shift
        )

        # UNCOND text: run its own attention path to propagate properly
        # Use uncond text Q against concatenated [uncond_txt, img] K/V (same img)
        u_q = torch.cat((u_txt_q, img_q), dim=2)
        u_k = torch.cat((u_txt_k, img_k), dim=2)
        u_v = torch.cat((u_txt_v, img_v), dim=2)
        u_attn_out = attention(u_q, u_k, u_v, pe=pe)
        u_txt_attn = u_attn_out[:, :uncond_txt_state.shape[1]]

        # Update uncond text state (residual connections with uncond modulation)
        uncond_txt_state = uncond_txt_state + uncond_txt_mod1.gate * block.txt_attn.proj(u_txt_attn)
        uncond_txt_state = uncond_txt_state + uncond_txt_mod2.gate * block.txt_mlp(
            (1 + uncond_txt_mod2.scale) * block.txt_norm2(uncond_txt_state) + uncond_txt_mod2.shift
        )

    # ---- Single Stream Blocks (standard forward, no guidance) ----
    cnt = 0
    x = torch.cat((txt, img), 1)
    txt_len = txt.shape[1]
    info['type'] = 'single'

    for block in model.single_blocks:
        info['id'] = cnt
        mod, _ = block.modulation(vec)
        x_mod = (1 + mod.scale) * block.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(block.linear1(x_mod), [3 * block.hidden_size, block.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=block.num_heads)
        q, k = block.norm(q, k, v)

        # Feature injection (same as original)
        if info['inject'] and info['id'] > 19:
            feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_V'
            if info['inverse']:
                info['feature'][feature_name] = v.cpu()
            else:
                v = info['feature'][feature_name].to(v.device)

        attn_out = attention(q, k, v, pe=pe)
        output = block.linear2(torch.cat((attn_out, block.mlp_act(mlp)), 2))
        x = x + mod.gate * output
        cnt += 1

    img = x[:, txt_len:, ...]
    img = model.final_layer(img, vec)
    return img, info


def denoise_attn_guidance(
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
    # attention guidance parameters
    attn_k: int = 32,
    attn_alpha: float = 1.0,
    overlap_target: str = "uncond",
):
    """Denoise with attention-level guidance.

    During inversion: uses standard forward (no guidance).
    During denoising: applies per-layer attention guidance in double blocks.
    """
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        if inverse:
            pred, info = model(
                img=img, img_ids=img_ids, txt=cond_txt, txt_ids=txt_ids,
                y=cond_vec, timesteps=t_vec, guidance=guidance_vec, info=info,
            )
        else:
            pred, info = flux_forward_with_attn_guidance(
                model, img=img, img_ids=img_ids, txt=cond_txt, txt_ids=txt_ids,
                timesteps=t_vec, y=cond_vec, guidance=guidance_vec, info=info,
                uncond_txt=uncond_txt, uncond_y=uncond_vec,
                attn_k=attn_k, attn_alpha=attn_alpha, overlap_target=overlap_target,
            )

        # 2nd order midpoint
        img_mid = img + (t_prev - t_curr) / 2 * pred
        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True

        if inverse:
            pred_mid, info = model(
                img=img_mid, img_ids=img_ids, txt=cond_txt, txt_ids=txt_ids,
                y=cond_vec, timesteps=t_vec_mid, guidance=guidance_vec, info=info,
            )
        else:
            pred_mid, info = flux_forward_with_attn_guidance(
                model, img=img_mid, img_ids=img_ids, txt=cond_txt, txt_ids=txt_ids,
                timesteps=t_vec_mid, y=cond_vec, guidance=guidance_vec, info=info,
                uncond_txt=uncond_txt, uncond_y=uncond_vec,
                attn_k=attn_k, attn_alpha=attn_alpha, overlap_target=overlap_target,
            )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info
