"""
SD3 Editing with Per-Layer CFG in Hidden Space.

Instead of:
- Standard CFG: combine cond/uncond predictions at the END (output space)
- Standard emb guidance: blend embeddings BEFORE the model (input space)

This approach:
- Runs both cond and uncond streams through each block
- Combines image hidden states at EACH LAYER using CFG equation:
    hidden = hidden_uncond + scale * (hidden_cond - hidden_uncond)
- Each text stream (enc_cond, enc_uncond) evolves independently

This applies the CFG principle (amplify cond-uncond difference) inside
the model at every layer, not just at the final output.

The scale can vary per layer (schedule).
"""

import os
import time
import math
import torch
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


def make_scale_schedule(schedule, base_scale, num_layers=24):
    if schedule == "uniform":
        return [base_scale] * num_layers
    elif schedule == "linear_up":
        return [base_scale * i / (num_layers - 1) for i in range(num_layers)]
    elif schedule == "linear_down":
        return [base_scale * (num_layers - 1 - i) / (num_layers - 1) for i in range(num_layers)]
    elif schedule == "peak_mid":
        mid = (num_layers - 1) / 2
        sigma = num_layers / 4
        return [base_scale * math.exp(-0.5 * ((i - mid) / sigma) ** 2) for i in range(num_layers)]
    elif schedule == "step_late":
        return [1.0 if i < num_layers // 2 else base_scale for i in range(num_layers)]
    elif schedule == "step_early":
        return [base_scale if i < num_layers // 2 else 1.0 for i in range(num_layers)]
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class SD3LayerCFGEditor:
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
             base_scale=2.0, schedule="uniform",
             custom_scales=None):
        """
        Args:
            base_scale: CFG-like scale for per-layer guidance.
                        1.0 = no guidance (just cond), >1.0 = amplify cond direction.
            schedule: how scale varies across layers.
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

        if custom_scales is not None:
            layer_scales = custom_scales
        else:
            layer_scales = make_scale_schedule(schedule, base_scale, self.num_layers)

        transformer = self.pipe.transformer
        original_forward = transformer.forward

        def patched_forward(hidden_states, encoder_hidden_states, pooled_projections,
                           timestep, joint_attention_kwargs=None, return_dict=True):
            height, width = hidden_states.shape[-2:]

            hidden_states = transformer.pos_embed(hidden_states)
            temb = transformer.time_text_embed(timestep, pooled_projections)

            # Project both streams
            enc_cond = transformer.context_embedder(cond_embeds)
            enc_uncond = transformer.context_embedder(neg_embeds)

            for idx, block in enumerate(transformer.transformer_blocks):
                scale_i = layer_scales[idx]

                if block.context_pre_only:
                    # Last block — no text output, just run with combined hidden
                    _, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=enc_cond,
                        temb=temb,
                    )
                elif scale_i == 1.0:
                    # No guidance — just run cond
                    enc_cond, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=enc_cond,
                        temb=temb,
                    )
                else:
                    # Run COND stream
                    enc_cond_out, hidden_cond = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=enc_cond,
                        temb=temb,
                    )

                    # Run UNCOND stream (same image input)
                    enc_uncond_out, hidden_uncond = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=enc_uncond,
                        temb=temb,
                    )

                    # Per-layer CFG: hidden = uncond + scale * (cond - uncond)
                    hidden_states = hidden_uncond + scale_i * (hidden_cond - hidden_uncond)

                    # Evolve both text streams independently
                    enc_cond = enc_cond_out
                    enc_uncond = enc_uncond_out

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
        return result.images[0], elapsed, {"schedule": schedule, "base_scale": base_scale,
                                            "layer_scales": layer_scales}
