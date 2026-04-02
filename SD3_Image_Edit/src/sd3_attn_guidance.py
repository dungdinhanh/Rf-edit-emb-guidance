"""
SD3 Image Editing with Attention-level Guidance.

Uses register_forward_pre_hook / register_forward_hook on each JointTransformerBlock
to inject guided attention K/V without modifying the transformer forward.
"""

import os
import time
import json
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def compute_text_importance_scores(img_q, txt_k):
    """[B,H,L,D] tensors -> [B, txt_len] scores."""
    img_q_avg = img_q.mean(dim=2, keepdim=True)
    scores = torch.matmul(img_q_avg, txt_k.transpose(-2, -1)) / (img_q.shape[-1] ** 0.5)
    return scores.squeeze(2).mean(dim=1)


def find_overlap_mask(cond_scores, uncond_scores, k):
    txt_len = cond_scores.shape[-1]
    k = min(k, txt_len)
    _, cond_topk = torch.topk(cond_scores, k, dim=-1)
    _, uncond_topk = torch.topk(uncond_scores, k, dim=-1)
    cond_mask = torch.zeros_like(cond_scores, dtype=torch.bool).scatter_(1, cond_topk, True)
    uncond_mask = torch.zeros_like(uncond_scores, dtype=torch.bool).scatter_(1, uncond_topk, True)
    return cond_mask & uncond_mask


class GuidedJointAttnProcessor:
    """Replaces JointAttnProcessor2_0 with attention-level guidance on text K/V."""

    def __init__(self, attn_k=32, attn_alpha=1.0):
        self.attn_k = attn_k
        self.attn_alpha = attn_alpha
        # Set by pre-hook before each block forward
        self.norm_uncond_encoder_hidden_states = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # Image projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Cond text projections
        cond_txt_q = attn.add_q_proj(encoder_hidden_states)
        cond_txt_k = attn.add_k_proj(encoder_hidden_states)
        cond_txt_v = attn.add_v_proj(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape to [B, H, L, D]
        img_q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        img_k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        img_v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_txt_k = cond_txt_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_txt_v = cond_txt_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_txt_q = cond_txt_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply guidance if uncond available and alpha > 0
        if self.norm_uncond_encoder_hidden_states is not None and self.attn_alpha > 0:
            uncond_txt = self.norm_uncond_encoder_hidden_states
            u_txt_k = attn.add_k_proj(uncond_txt).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            u_txt_v = attn.add_v_proj(uncond_txt).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            cond_scores = compute_text_importance_scores(img_q, c_txt_k)
            uncond_scores = compute_text_importance_scores(img_q, u_txt_k)
            overlap = find_overlap_mask(cond_scores, uncond_scores, self.attn_k)
            mask = overlap.unsqueeze(1).unsqueeze(-1)

            alpha = self.attn_alpha
            masked_u_k = u_txt_k * (~mask)
            masked_u_v = u_txt_v * (~mask)
            guided_txt_k = (1.0 + alpha) * c_txt_k - alpha * masked_u_k
            guided_txt_v = (1.0 + alpha) * c_txt_v - alpha * masked_u_v
        else:
            guided_txt_k = c_txt_k
            guided_txt_v = c_txt_v

        # Concatenate and attend
        full_q = torch.cat([img_q, c_txt_q], dim=2)
        full_k = torch.cat([img_k, guided_txt_k], dim=2)
        full_v = torch.cat([img_v, guided_txt_v], dim=2)

        out = F.scaled_dot_product_attention(full_q, full_k, full_v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)

        hidden_states_out = out[:, :residual.shape[1]]
        encoder_hidden_states_out = out[:, residual.shape[1]:]

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        if not attn.context_pre_only:
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

        if input_ndim == 4:
            hidden_states_out = hidden_states_out.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_out = encoder_hidden_states_out.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states_out, encoder_hidden_states_out


class SD3AttnGuidanceEditor:
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

        # Save original processors
        self.original_processors = {}
        for i, block in enumerate(self.pipe.transformer.transformer_blocks):
            self.original_processors[i] = block.attn.processor
        print("SD3 loaded.")

    def _install_guidance(self, uncond_embeds, attn_k, attn_alpha):
        """Install guided processors and hooks on all blocks."""
        transformer = self.pipe.transformer
        self._hooks = []
        self._guided_processors = {}

        # Project uncond through context_embedder once (same as cond path)
        self._uncond_projected = transformer.context_embedder(
            uncond_embeds.to(next(transformer.parameters()).device)
        )
        # Mutable state for uncond propagation through blocks
        self._uncond_state = [self._uncond_projected.clone()]

        for i, block in enumerate(transformer.transformer_blocks):
            proc = GuidedJointAttnProcessor(attn_k=attn_k, attn_alpha=attn_alpha)
            self._guided_processors[i] = proc
            block.attn.processor = proc

            # Pre-hook: normalize uncond and store on processor before attention
            def make_pre_hook(block_ref, proc_ref, idx):
                def pre_hook(module, args):
                    # args = (hidden_states, encoder_hidden_states, temb)
                    temb = args[2] if len(args) > 2 else None
                    if temb is None and 'temb' in (args[1] if isinstance(args[1], dict) else {}):
                        temb = args[1]['temb']

                    uncond = self._uncond_state[0]
                    if block_ref.context_pre_only:
                        norm_uncond = block_ref.norm1_context(uncond, temb)
                    else:
                        norm_uncond, _, _, _, _ = block_ref.norm1_context(uncond, emb=temb)
                    proc_ref.norm_uncond_encoder_hidden_states = norm_uncond
                    return None
                return pre_hook

            # Post-hook: propagate uncond through block's text FF
            def make_post_hook(block_ref, idx):
                def post_hook(module, args, output):
                    if block_ref.context_pre_only:
                        return output

                    temb = args[2] if len(args) > 2 else None
                    uncond = self._uncond_state[0]

                    # Get uncond modulation params
                    _, u_gate_msa, u_shift_mlp, u_scale_mlp, u_gate_mlp = block_ref.norm1_context(
                        uncond, emb=temb
                    )

                    # Run uncond attention output through the text FF
                    # We approximate: skip attention update, just apply FF
                    norm_uncond_ff = block_ref.norm2_context(uncond)
                    norm_uncond_ff = norm_uncond_ff * (1 + u_scale_mlp[:, None]) + u_shift_mlp[:, None]
                    uncond_ff = block_ref.ff_context(norm_uncond_ff)
                    self._uncond_state[0] = uncond + u_gate_mlp.unsqueeze(1) * uncond_ff

                    return output
                return post_hook

            h1 = block.register_forward_pre_hook(make_pre_hook(block, proc, i))
            h2 = block.register_forward_hook(make_post_hook(block, i))
            self._hooks.extend([h1, h2])

    def _uninstall_guidance(self):
        """Restore original processors and remove hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for i, block in enumerate(self.pipe.transformer.transformer_blocks):
            if i in self.original_processors:
                block.attn.processor = self.original_processors[i]

    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7, attn_k=32, attn_alpha=1.0):
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        # Pre-encode prompts
        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=self.pipe._execution_device,
        )

        # Install guidance hooks
        self._install_guidance(neg_embeds, attn_k, attn_alpha)

        try:
            result = self.pipe(
                prompt_embeds=cond_embeds,
                negative_prompt_embeds=None,
                pooled_prompt_embeds=cond_pooled,
                negative_pooled_prompt_embeds=None,
                image=source_image,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=1.0,
            )
        finally:
            self._uninstall_guidance()

        elapsed = time.time() - t0
        return result.images[0], elapsed
