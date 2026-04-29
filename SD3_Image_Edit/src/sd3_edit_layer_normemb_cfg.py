"""
SD3 LayerNormEmbCFG — CFG on normalized text embeddings within each block.

At each block, after AdaLayerNorm normalizes enc_cond and enc_uncond
with the same timestep modulation, combines them with CFG equation:

    norm_txt = norm_txt_uncond + scale * (norm_txt_cond - norm_txt_uncond)

Then runs single QKV projection, single attention, single FF.
The text output from the block uses the cond path for evolution.

This is the earliest injection point — before QKV projection can
absorb the amplification through its learned linear transformation.
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


class NormEmbCFGProcessor:
    """Replaces JointAttnProcessor2_0. Receives pre-guided norm_txt."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        # Standard JointAttnProcessor2_0 — no modification needed here
        # The guidance was already applied to encoder_hidden_states
        # before this processor is called (via the pre-hook on the block)
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

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states


class SD3LayerNormEmbCFGEditor:
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
        # Project uncond through context_embedder (same as cond path)
        self._enc_uncond = transformer.context_embedder(neg_embeds)
        self._hooks = []
        self._layer_scales = layer_scales

        for i, block in enumerate(transformer.transformer_blocks):
            # Install standard processor (no modification inside attention)
            block.attn.processor = NormEmbCFGProcessor()

            # Pre-hook: intercept the block forward to apply CFG on normalized text
            def make_pre_hook(block_ref, block_idx):
                def pre_hook(module, args, kwargs):
                    scale_i = self._layer_scales[block_idx]
                    if scale_i == 1.0:
                        return None  # no guidance

                    temb = kwargs.get('temb', None)
                    enc_cond = kwargs.get('encoder_hidden_states', None)
                    enc_uncond = self._enc_uncond

                    if enc_cond is None or temb is None:
                        return None

                    # Normalize both cond and uncond with same temb
                    if block_ref.context_pre_only:
                        norm_cond = block_ref.norm1_context(enc_cond, temb)
                        norm_uncond = block_ref.norm1_context(enc_uncond, temb)
                    else:
                        norm_cond, c_gate, c_shift, c_scale, c_gate_mlp = block_ref.norm1_context(enc_cond, emb=temb)
                        norm_uncond, _, _, _, _ = block_ref.norm1_context(enc_uncond, emb=temb)

                    # CFG on normalized text
                    guided_norm = norm_uncond + scale_i * (norm_cond - norm_uncond)

                    # Replace encoder_hidden_states with guided version
                    # But we need to bypass the block's own norm1_context
                    # We'll replace enc with a value that after norm1_context gives guided_norm
                    # Trick: set enc such that norm1_context(enc) = guided_norm
                    # Since norm1_context does: LayerNorm(enc) * (1+scale) + shift
                    # We can't easily invert this. Instead, we'll modify the block forward.

                    # Simpler approach: directly modify kwargs to pass guided_norm
                    # and skip the block's norm1_context by modifying the block behavior.
                    # But that requires deeper hooks.

                    # Cleanest approach: store guided_norm and use a post-norm hook
                    # Actually, let's just override encoder_hidden_states with a value
                    # that will produce guided_norm after norm1_context.
                    # Since norm1_context applies: LN(x) * (1+s) + sh
                    # If we want output = guided_norm, we need:
                    # LN(x) = (guided_norm - sh) / (1+s)
                    # x = LN_inverse((guided_norm - sh) / (1+s))
                    # LN inverse is hard. Let's use a different approach.

                    # Better: replace the normalized result directly by patching
                    # We'll store the guided_norm and have the processor use it
                    block_ref._guided_norm_txt = guided_norm
                    block_ref._use_guided = True

                    return None
                return pre_hook

            def make_post_hook(block_ref, block_idx):
                def post_hook(module, args, kwargs, output):
                    if hasattr(block_ref, '_use_guided'):
                        del block_ref._use_guided
                    if hasattr(block_ref, '_guided_norm_txt'):
                        del block_ref._guided_norm_txt

                    # Evolve uncond text through FF (approximate)
                    if not block_ref.context_pre_only:
                        scale_i = self._layer_scales[block_idx]
                        if scale_i != 1.0:
                            temb = kwargs.get('temb', None)
                            enc_uncond = self._enc_uncond
                            _, u_gate, u_shift, u_scale, u_gate_mlp = block_ref.norm1_context(enc_uncond, emb=temb)
                            norm_uf = block_ref.norm2_context(enc_uncond)
                            norm_uf = norm_uf * (1 + u_scale[:, None]) + u_shift[:, None]
                            uf = block_ref.ff_context(norm_uf)
                            self._enc_uncond = enc_uncond + u_gate_mlp.unsqueeze(1) * uf
                    return output
                return post_hook

            h1 = block.register_forward_pre_hook(make_pre_hook(block, i), with_kwargs=True)
            h2 = block.register_forward_hook(make_post_hook(block, i), with_kwargs=True)
            self._hooks.extend([h1, h2])

        # Now we need to patch each block's forward to use guided_norm_txt
        # instead of computing norm1_context internally.
        # We do this by replacing the block's forward method.
        self._orig_forwards = {}
        for i, block in enumerate(transformer.transformer_blocks):
            self._orig_forwards[i] = block.forward
            block_ref = block
            block_idx = i

            def make_patched_forward(orig_forward, blk, idx):
                def patched_forward(hidden_states, encoder_hidden_states, temb):
                    if hasattr(blk, '_use_guided') and blk._use_guided:
                        guided_norm = blk._guided_norm_txt

                        # Image path (unchanged from original)
                        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.norm1(hidden_states, emb=temb)

                        if blk.context_pre_only:
                            # Last block — use guided_norm directly for attention
                            attn_output, _ = blk.attn(
                                hidden_states=norm_hidden,
                                encoder_hidden_states=guided_norm)
                            attn_output = gate_msa.unsqueeze(1) * attn_output
                            hidden_states = hidden_states + attn_output
                            norm_hidden2 = blk.norm2(hidden_states)
                            norm_hidden2 = norm_hidden2 * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                            ff_output = blk.ff(norm_hidden2)
                            ff_output = gate_mlp.unsqueeze(1) * ff_output
                            hidden_states = hidden_states + ff_output
                            return encoder_hidden_states, hidden_states
                        else:
                            # Get cond modulation params (for text residual + FF)
                            _, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = blk.norm1_context(
                                encoder_hidden_states, emb=temb)

                            # Attention with guided_norm for text
                            attn_output, context_attn_output = blk.attn(
                                hidden_states=norm_hidden,
                                encoder_hidden_states=guided_norm)

                            # Image residual
                            attn_output = gate_msa.unsqueeze(1) * attn_output
                            hidden_states = hidden_states + attn_output

                            # Text residual (use cond gates for evolution)
                            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                            encoder_hidden_states = encoder_hidden_states + context_attn_output

                            # Image FF
                            norm_hidden2 = blk.norm2(hidden_states)
                            norm_hidden2 = norm_hidden2 * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                            ff_output = blk.ff(norm_hidden2)
                            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

                            # Text FF
                            norm_enc2 = blk.norm2_context(encoder_hidden_states)
                            norm_enc2 = norm_enc2 * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                            context_ff = blk.ff_context(norm_enc2)
                            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff

                            return encoder_hidden_states, hidden_states
                    else:
                        return orig_forward(hidden_states=hidden_states,
                                          encoder_hidden_states=encoder_hidden_states,
                                          temb=temb)
                return patched_forward

            block.forward = make_patched_forward(self._orig_forwards[i], block, i)

    def _uninstall(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for i, block in enumerate(self.pipe.transformer.transformer_blocks):
            block.attn.processor = self.original_processors[i]
            block.forward = self._orig_forwards[i]

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
