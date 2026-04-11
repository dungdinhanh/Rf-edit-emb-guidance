"""
SD3 Editing with Per-Layer Embedding Guidance (v3).

Each of the 24 transformer blocks gets its own guidance alpha.
A pre-hook on each block blends encoder_hidden_states toward
the guided version: enc = enc + alpha_i * (guided_proj - cond_proj).

Alpha schedules:
- "uniform": same alpha for all layers
- "linear_up": 0 → alpha (stronger guidance in later layers)
- "linear_down": alpha → 0 (stronger in early layers)
- "peak_mid": bell curve peaking at middle layers
- "early_late": strong at edges, weak in middle
- "custom": user-provided list of 24 values
"""

import os
import time
import math
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def norm_emb_guidance(cond, uncond, alpha):
    """L2-normalized embedding guidance."""
    cond_f = cond.float()
    uncond_f = uncond.float()
    v = (1.0 + alpha) * cond_f - alpha * uncond_f
    cond_norm = cond_f.norm(dim=-1, keepdim=True)
    v_norm = v.norm(dim=-1, keepdim=True) + 1e-8
    return (v * (cond_norm / v_norm)).to(cond.dtype)


def make_alpha_schedule(schedule, base_alpha, num_layers=24):
    """Generate per-layer alpha values."""
    if schedule == "uniform":
        return [base_alpha] * num_layers
    elif schedule == "linear_up":
        # 0 → base_alpha
        return [base_alpha * i / (num_layers - 1) for i in range(num_layers)]
    elif schedule == "linear_down":
        # base_alpha → 0
        return [base_alpha * (num_layers - 1 - i) / (num_layers - 1) for i in range(num_layers)]
    elif schedule == "peak_mid":
        # Gaussian peak at middle
        mid = (num_layers - 1) / 2
        sigma = num_layers / 4
        return [base_alpha * math.exp(-0.5 * ((i - mid) / sigma) ** 2) for i in range(num_layers)]
    elif schedule == "early_late":
        # Strong at edges, weak in middle (inverted Gaussian)
        mid = (num_layers - 1) / 2
        sigma = num_layers / 4
        return [base_alpha * (1 - math.exp(-0.5 * ((i - mid) / sigma) ** 2)) for i in range(num_layers)]
    elif schedule == "step_late":
        # 0 for first half, base_alpha for second half
        return [0.0 if i < num_layers // 2 else base_alpha for i in range(num_layers)]
    elif schedule == "step_early":
        # base_alpha for first half, 0 for second half
        return [base_alpha if i < num_layers // 2 else 0.0 for i in range(num_layers)]
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class SD3PerLayerEditor:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda"):
        self.device = torch.device(device)
        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.num_layers = len(self.pipe.transformer.transformer_blocks)
        print(f"SD3 loaded. {self.num_layers} transformer blocks.")

    @torch.no_grad()
    def edit(self, source_image, source_prompt, target_prompt,
             num_steps=25, strength=0.7,
             base_alpha=0.8, schedule="uniform",
             custom_alphas=None):
        t0 = time.time()

        w, h = source_image.size
        w, h = w - w % 16, h - h % 16
        source_image = source_image.resize((w, h))

        device = self.pipe._execution_device

        # Encode prompts
        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=device,
        )

        # Compute guided embeddings (reference at alpha=1 for offset direction)
        guided_embeds = norm_emb_guidance(cond_embeds, neg_embeds, 1.0)
        guided_pooled = norm_emb_guidance(cond_pooled, neg_pooled, 1.0)

        # Get alpha schedule
        if custom_alphas is not None:
            layer_alphas = custom_alphas
        else:
            layer_alphas = make_alpha_schedule(schedule, base_alpha, self.num_layers)

        # Project both through context_embedder to get offset in projected space
        # We'll patch the transformer forward to apply per-layer guidance
        hooks = []
        transformer = self.pipe.transformer

        # We need access to the projected embeddings. Hook into the forward
        # to capture the projected cond and guided embeddings.
        _projected = {}

        original_forward = transformer.forward

        def patched_forward(hidden_states, encoder_hidden_states, pooled_projections,
                           timestep, joint_attention_kwargs=None, return_dict=True):
            height, width = hidden_states.shape[-2:]

            hidden_states = transformer.pos_embed(hidden_states)
            temb = transformer.time_text_embed(timestep, pooled_projections)
            encoder_hidden_states = transformer.context_embedder(encoder_hidden_states)

            # Also project guided embeddings to get offset
            guided_proj = transformer.context_embedder(guided_embeds)
            cond_proj_init = transformer.context_embedder(cond_embeds)
            offset = guided_proj - cond_proj_init

            for idx, block in enumerate(transformer.transformer_blocks):
                alpha_i = layer_alphas[idx]
                if alpha_i > 0:
                    # Blend current enc toward guided direction
                    enc_modified = encoder_hidden_states + alpha_i * offset
                else:
                    enc_modified = encoder_hidden_states

                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=enc_modified,
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

        # Temporarily patch the forward
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
        return result.images[0], elapsed, {"schedule": schedule, "base_alpha": base_alpha,
                                            "layer_alphas": layer_alphas}
