"""
SD3 LayerEmbeddCFG — CFG applied to text QKV (embedding) within each block.

Instead of running attention twice, combines the text K/V with CFG equation
BEFORE attention. Only one attention call, but with guided text K/V.

    txt_k = txt_k_uncond + scale * (txt_k_cond - txt_k_uncond)
    txt_v = txt_v_uncond + scale * (txt_v_cond - txt_v_uncond)
    txt_q = txt_q_cond  (use cond queries)

    Then run single attention with guided K/V.
"""

import os
import time
import math
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline


def make_layer_schedule(schedule, base_scale, num_layers=24):
    if schedule == "uniform":
        return [base_scale] * num_layers
    elif schedule == "linear_up":
        return [base_scale * i / (num_layers - 1) for i in range(num_layers)]
    elif schedule == "peak_mid":
        mid = (num_layers - 1) / 2
        sigma = num_layers / 4
        return [base_scale * math.exp(-0.5 * ((i - mid) / sigma) ** 2) for i in range(num_layers)]
    else:
        return [base_scale] * num_layers


class EmbeddCFGProcessor:
    """Custom attention processor that applies CFG to text K/V before attention."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.uncond_enc = None

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

        # Image QKV (ONCE)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        img_q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        img_k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        img_v = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Cond text QKV
        c_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.uncond_enc is not None and self.scale > 1.0:
            # Uncond text KV (Q not needed — we use cond Q)
            u_k = attn.add_k_proj(self.uncond_enc).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            u_v = attn.add_v_proj(self.uncond_enc).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # CFG on text K and V
            guided_k = u_k + self.scale * (c_k - u_k)
            guided_v = u_v + self.scale * (c_v - u_v)

            # Single attention with guided text K/V
            full_q = torch.cat([img_q, c_q], dim=2)
            full_k = torch.cat([img_k, guided_k], dim=2)
            full_v = torch.cat([img_v, guided_v], dim=2)
        else:
            full_q = torch.cat([img_q, c_q], dim=2)
            full_k = torch.cat([img_k, c_k], dim=2)
            full_v = torch.cat([img_v, c_v], dim=2)

        attn_out = F.scaled_dot_product_attention(full_q, full_k, full_v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, inner_dim).to(hidden_states.dtype)

        hidden_states_out = attn_out[:, :residual.shape[1]]
        encoder_hidden_states_out = attn_out[:, residual.shape[1]:]

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


class SD3LayerEmbeddCFGEditor:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda", task="edit"):
        self.device = torch.device(device)
        self.task = task
        print(f"Loading SD3 pipeline ({task})...")
        if task == "edit":
            self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        else:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.num_layers = len(self.pipe.transformer.transformer_blocks)

        self.original_processors = {}
        for i, block in enumerate(self.pipe.transformer.transformer_blocks):
            self.original_processors[i] = block.attn.processor
        print(f"SD3 loaded. {self.num_layers} layers.")

    def _install(self, neg_embeds, layer_scales):
        transformer = self.pipe.transformer
        enc_uncond = transformer.context_embedder(neg_embeds)
        self._hooks = []
        self._processors = {}

        for i, block in enumerate(transformer.transformer_blocks):
            proc = EmbeddCFGProcessor(scale=layer_scales[i])
            self._processors[i] = proc
            block.attn.processor = proc

            def make_pre_hook(proc_ref, block_ref):
                def pre_hook(module, args, kwargs):
                    temb = kwargs.get('temb', None)
                    if block_ref.context_pre_only:
                        norm_uncond = block_ref.norm1_context(enc_uncond, temb)
                    else:
                        norm_uncond, _, _, _, _ = block_ref.norm1_context(enc_uncond, emb=temb)
                    proc_ref.uncond_enc = norm_uncond
                    return None
                return pre_hook

            h = block.register_forward_pre_hook(make_pre_hook(proc, block), with_kwargs=True)
            self._hooks.append(h)

    def _uninstall(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for i, block in enumerate(self.pipe.transformer.transformer_blocks):
            block.attn.processor = self.original_processors[i]

    @torch.no_grad()
    def run(self, source_image=None, target_prompt="",
            num_steps=25, strength=0.7,
            base_scale=5.0, layer_schedule="peak_mid", seed=42):
        t0 = time.time()
        device = self.pipe._execution_device

        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True, device=device)

        layer_scales = make_layer_schedule(layer_schedule, base_scale, self.num_layers)
        self._install(neg_embeds, layer_scales)

        try:
            if self.task == "edit":
                w, h = source_image.size
                w, h = w - w % 16, h - h % 16
                source_image = source_image.resize((w, h))
                result = self.pipe(
                    prompt_embeds=cond_embeds, negative_prompt_embeds=None,
                    pooled_prompt_embeds=cond_pooled, negative_pooled_prompt_embeds=None,
                    image=source_image, strength=strength,
                    num_inference_steps=num_steps, guidance_scale=1.0)
            else:
                result = self.pipe(
                    prompt_embeds=cond_embeds, negative_prompt_embeds=None,
                    pooled_prompt_embeds=cond_pooled, negative_pooled_prompt_embeds=None,
                    num_inference_steps=num_steps, guidance_scale=1.0,
                    generator=torch.Generator(device=device).manual_seed(seed))
        finally:
            self._uninstall()

        elapsed = time.time() - t0
        return result.images[0], elapsed
