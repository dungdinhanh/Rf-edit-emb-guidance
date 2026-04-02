"""
SD3 Image Editing with Attention Adaptive Guidance (v3).

Modifies attention WEIGHTS (not K/V) during denoising:
1. Compute attention weights for both positive and negative prompts
2. Score text tokens by how much image tokens attend to them
3. Find top-k overlap between positive and negative important tokens
4. Randomly drop overlapping tokens' attention logits to -inf
5. Re-softmax and apply to original V (V is never modified)
"""

import os
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


class AdaptiveAttnProcessor:
    """Replaces JointAttnProcessor2_0 with attention-weight modification."""

    def __init__(self, top_k=32, drop_rate=0.5):
        self.top_k = top_k
        self.drop_rate = drop_rate
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
        inner_dim = attn.to_q.out_features
        head_dim = inner_dim // attn.heads
        H = attn.heads

        # Image projections
        img_q = attn.to_q(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        img_k = attn.to_k(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        img_v = attn.to_v(hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)

        # Cond text projections
        txt_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        txt_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)
        txt_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, H, head_dim).transpose(1, 2)

        img_len = img_q.shape[2]
        txt_len = txt_k.shape[2]

        # Build full joint Q, K, V (standard path)
        full_q = torch.cat([img_q, txt_q], dim=2)  # [B, H, img+txt, D]
        full_k = torch.cat([img_k, txt_k], dim=2)
        full_v = torch.cat([img_v, txt_v], dim=2)

        if (self.norm_uncond_encoder_hidden_states is not None
                and self.drop_rate > 0 and self.training_or_active):
            # --- Adaptive guidance path ---
            uncond_txt = self.norm_uncond_encoder_hidden_states

            # Uncond text K for scoring
            u_txt_k = attn.add_k_proj(uncond_txt).view(batch_size, -1, H, head_dim).transpose(1, 2)

            scale = head_dim ** -0.5

            # Score cond text tokens: image queries attending to [img_keys, cond_txt_keys]
            cond_logits = img_q @ torch.cat([img_k, txt_k], dim=2).transpose(-2, -1) * scale
            cond_weights = F.softmax(cond_logits, dim=-1)
            # Extract image-to-text portion and score
            cond_img2txt = cond_weights[:, :, :, img_len:]  # [B, H, img_len, txt_len]
            cond_scores = cond_img2txt.sum(dim=2).mean(dim=1)  # [B, txt_len]

            # Score uncond text tokens similarly
            uncond_logits = img_q @ torch.cat([img_k, u_txt_k], dim=2).transpose(-2, -1) * scale
            uncond_weights = F.softmax(uncond_logits, dim=-1)
            uncond_img2txt = uncond_weights[:, :, :, img_len:]
            uncond_scores = uncond_img2txt.sum(dim=2).mean(dim=1)  # [B, txt_len]

            # Find top-k overlap
            k = min(self.top_k, txt_len)
            _, cond_topk = torch.topk(cond_scores, k, dim=-1)
            _, uncond_topk = torch.topk(uncond_scores, k, dim=-1)
            cond_set = torch.zeros(batch_size, txt_len, dtype=torch.bool, device=img_q.device)
            uncond_set = torch.zeros(batch_size, txt_len, dtype=torch.bool, device=img_q.device)
            cond_set.scatter_(1, cond_topk, True)
            uncond_set.scatter_(1, uncond_topk, True)
            overlap = cond_set & uncond_set  # [B, txt_len]

            # Random drop within overlap
            rand_mask = torch.rand(batch_size, txt_len, device=img_q.device) < self.drop_rate
            drop = overlap & rand_mask  # [B, txt_len]

            # Compute full joint attention logits
            logits = full_q @ full_k.transpose(-2, -1) * scale  # [B, H, L, L]

            # Mask dropped text token columns to -inf
            # drop: [B, txt_len] -> [B, 1, 1, txt_len]
            drop_expanded = drop.unsqueeze(1).unsqueeze(2)
            logits[:, :, :, img_len:].masked_fill_(drop_expanded, float('-inf'))

            # Softmax and apply V
            attn_weights = F.softmax(logits, dim=-1)
            attn_weights = attn_weights.nan_to_num(0.0)  # safety for all-inf rows
            out = attn_weights @ full_v
        else:
            # Standard attention (no guidance)
            out = F.scaled_dot_product_attention(full_q, full_k, full_v, dropout_p=0.0, is_causal=False)

        # Reshape and split
        out = out.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(hidden_states.dtype)
        hidden_states_out = out[:, :residual.shape[1]]
        encoder_hidden_states_out = out[:, residual.shape[1]:]

        # Output projections
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        if not attn.context_pre_only:
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

        if input_ndim == 4:
            hidden_states_out = hidden_states_out.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_out = encoder_hidden_states_out.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states_out, encoder_hidden_states_out

    @property
    def training_or_active(self):
        """Always active during inference."""
        return True


class SD3AdaptiveGuidanceEditor:
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

    def _install_guidance(self, uncond_embeds, top_k, drop_rate):
        """Install adaptive processors and hooks."""
        transformer = self.pipe.transformer
        self._hooks = []
        self._guided_processors = {}

        # Project uncond through context_embedder
        self._uncond_projected = transformer.context_embedder(
            uncond_embeds.to(next(transformer.parameters()).device)
        )
        self._uncond_state = [self._uncond_projected.clone()]

        for i, block in enumerate(transformer.transformer_blocks):
            proc = AdaptiveAttnProcessor(top_k=top_k, drop_rate=drop_rate)
            self._guided_processors[i] = proc
            block.attn.processor = proc

            def make_pre_hook(block_ref, proc_ref):
                def pre_hook(module, args, kwargs):
                    temb = kwargs.get('temb', None)
                    uncond = self._uncond_state[0].to(temb.device)
                    if block_ref.context_pre_only:
                        norm_uncond = block_ref.norm1_context(uncond, temb)
                    else:
                        norm_uncond, _, _, _, _ = block_ref.norm1_context(uncond, emb=temb)
                    proc_ref.norm_uncond_encoder_hidden_states = norm_uncond
                    return None
                return pre_hook

            def make_post_hook(block_ref):
                def post_hook(module, args, kwargs, output):
                    if block_ref.context_pre_only:
                        return output
                    temb = kwargs.get('temb', None)
                    uncond = self._uncond_state[0].to(temb.device)
                    _, u_gate_msa, u_shift_mlp, u_scale_mlp, u_gate_mlp = block_ref.norm1_context(
                        uncond, emb=temb
                    )
                    norm_uncond_ff = block_ref.norm2_context(uncond)
                    norm_uncond_ff = norm_uncond_ff * (1 + u_scale_mlp[:, None]) + u_shift_mlp[:, None]
                    uncond_ff = block_ref.ff_context(norm_uncond_ff)
                    self._uncond_state[0] = uncond + u_gate_mlp.unsqueeze(1) * uncond_ff
                    return output
                return post_hook

            h1 = block.register_forward_pre_hook(make_pre_hook(block, proc), with_kwargs=True)
            h2 = block.register_forward_hook(make_post_hook(block), with_kwargs=True)
            self._hooks.extend([h1, h2])

    def _uninstall_guidance(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for i, block in enumerate(self.pipe.transformer.transformer_blocks):
            if i in self.original_processors:
                block.attn.processor = self.original_processors[i]

    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7, top_k=32, drop_rate=0.5):
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

        self._install_guidance(neg_embeds, top_k, drop_rate)

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
