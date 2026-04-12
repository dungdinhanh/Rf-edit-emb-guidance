"""
SD3 Editing with per-layer embedding type selection.

Each layer uses one of three embedding types:
1. "cond" — conditional (target prompt) embedding
2. "uncond" — unconditional (empty prompt) embedding
3. "guided" — normalized embedding guidance: (1+α)*cond - α*uncond, normalized

This allows testing which layers benefit from which type of conditioning.
"""

import os
import time
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def norm_emb_guidance(cond, uncond, alpha):
    cond_f = cond.float()
    uncond_f = uncond.float()
    v = (1.0 + alpha) * cond_f - alpha * uncond_f
    cond_norm = cond_f.norm(dim=-1, keepdim=True)
    v_norm = v.norm(dim=-1, keepdim=True) + 1e-8
    return (v * (cond_norm / v_norm)).to(cond.dtype)


class SD3ThreeEmbEditor:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda"):
        self.device = torch.device(device)
        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.num_layers = len(self.pipe.transformer.transformer_blocks)
        print(f"SD3 loaded. {self.num_layers} layers.")

    @torch.no_grad()
    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7, emb_alpha=0.8,
             layer_modes=None):
        """
        Args:
            layer_modes: list of 24 strings, each "cond", "uncond", or "guided"
                         If None, defaults to all "cond".
        """
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        device = self.pipe._execution_device

        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=device,
        )

        guided_embeds = norm_emb_guidance(cond_embeds, neg_embeds, emb_alpha)
        guided_pooled = norm_emb_guidance(cond_pooled, neg_pooled, emb_alpha)

        if layer_modes is None:
            layer_modes = ["cond"] * self.num_layers

        # Map mode to embeddings
        emb_map = {
            "cond": (cond_embeds, cond_pooled),
            "uncond": (neg_embeds, neg_pooled),
            "guided": (guided_embeds, guided_pooled),
        }

        transformer = self.pipe.transformer
        original_forward = transformer.forward

        def patched_forward(hidden_states, encoder_hidden_states, pooled_projections,
                           timestep, joint_attention_kwargs=None, return_dict=True):
            height, width = hidden_states.shape[-2:]
            hidden_states = transformer.pos_embed(hidden_states)
            temb = transformer.time_text_embed(timestep, pooled_projections)

            # Project all three embedding types
            enc_cond = transformer.context_embedder(cond_embeds)
            enc_uncond = transformer.context_embedder(neg_embeds)
            enc_guided = transformer.context_embedder(guided_embeds)

            enc_map = {
                "cond": enc_cond,
                "uncond": enc_uncond,
                "guided": enc_guided,
            }

            # Start with cond as the initial enc state
            encoder_hidden_states = enc_cond.clone()

            for idx, block in enumerate(transformer.transformer_blocks):
                mode = layer_modes[idx] if idx < len(layer_modes) else "cond"

                # Replace enc with the target mode's projected embedding
                # But preserve the evolution: blend the target into the evolving stream
                target_enc = enc_map[mode]

                # Use target embedding for this layer's attention
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=target_enc,
                    temb=temb,
                )

            hidden_states = transformer.norm_out(hidden_states, temb)
            hidden_states = transformer.proj_out(hidden_states)

            patch_size = transformer.config.patch_size
            h_out = height // patch_size
            w_out = width // patch_size
            hidden_states = hidden_states.reshape(
                hidden_states.shape[0], h_out, w_out,
                patch_size, patch_size, transformer.out_channels,
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                hidden_states.shape[0], transformer.out_channels,
                h_out * patch_size, w_out * patch_size,
            )

            if not return_dict:
                return (output,)
            from diffusers.models.transformers.transformer_sd3 import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=output)

        transformer.forward = patched_forward
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
            transformer.forward = original_forward

        elapsed = time.time() - t0
        return result.images[0], elapsed
