"""
SD3 Per-Layer Embedding Guidance v2 — Parallel Stream Approach.

Maintains two parallel evolving encoder_hidden_states streams:
1. cond stream: evolves through blocks with conditional embedding
2. guided stream: evolves through blocks with guided embedding

At each block, interpolates between the two evolved streams for the
image hidden_states computation.

This is 2x text compute but correctly preserves layer evolution.
"""

import os
import time
import math
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


def make_alpha_schedule(schedule, base_alpha, num_layers=24):
    if schedule == "uniform":
        return [base_alpha] * num_layers
    elif schedule == "linear_up":
        return [base_alpha * i / (num_layers - 1) for i in range(num_layers)]
    elif schedule == "linear_down":
        return [base_alpha * (num_layers - 1 - i) / (num_layers - 1) for i in range(num_layers)]
    elif schedule == "peak_mid":
        mid = (num_layers - 1) / 2
        sigma = num_layers / 4
        return [base_alpha * math.exp(-0.5 * ((i - mid) / sigma) ** 2) for i in range(num_layers)]
    elif schedule == "step_late":
        return [0.0 if i < num_layers // 2 else base_alpha for i in range(num_layers)]
    elif schedule == "step_early":
        return [base_alpha if i < num_layers // 2 else 0.0 for i in range(num_layers)]
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class SD3PerLayerV2Editor:
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
             num_steps=25, strength=0.7,
             base_alpha=0.8, schedule="uniform",
             custom_alphas=None, emb_alpha=1.0):
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

        if custom_alphas is not None:
            layer_alphas = custom_alphas
        else:
            layer_alphas = make_alpha_schedule(schedule, base_alpha, self.num_layers)

        transformer = self.pipe.transformer
        original_forward = transformer.forward

        def patched_forward(hidden_states, encoder_hidden_states, pooled_projections,
                           timestep, joint_attention_kwargs=None, return_dict=True):
            height, width = hidden_states.shape[-2:]

            hidden_states = transformer.pos_embed(hidden_states)
            temb = transformer.time_text_embed(timestep, pooled_projections)

            # Project BOTH streams through context_embedder
            enc_cond = transformer.context_embedder(cond_embeds)
            enc_guided = transformer.context_embedder(guided_embeds)

            for idx, block in enumerate(transformer.transformer_blocks):
                alpha_i = layer_alphas[idx]

                # Run COND stream through this block
                enc_cond_out, hidden_cond = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond,
                    temb=temb,
                )

                if alpha_i > 0 and not block.context_pre_only:
                    # Run GUIDED stream through this block
                    enc_guided_out, hidden_guided = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=enc_guided,
                        temb=temb,
                    )

                    # Interpolate image hidden states
                    hidden_states = (1.0 - alpha_i) * hidden_cond + alpha_i * hidden_guided

                    # Evolve both enc streams independently
                    enc_cond = enc_cond_out
                    enc_guided = enc_guided_out
                else:
                    # No guidance at this layer — use cond only
                    hidden_states = hidden_cond
                    enc_cond = enc_cond_out
                    # Still evolve guided stream through cond path to keep it reasonable
                    enc_guided = enc_cond_out

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
        return result.images[0], elapsed, {"schedule": schedule, "base_alpha": base_alpha,
                                            "layer_alphas": layer_alphas}
