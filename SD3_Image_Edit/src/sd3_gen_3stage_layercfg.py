"""
SD3 Three-Stage Generation: Standard CFG + Per-Layer CFG.

Same as sd3_edit_3stage_layercfg but for text-to-image generation.
For generation, interval guidance (middle CFG) is optimal, so the
3-stage uses: none/emb (early) + CFG (middle) + layercfg (late).
"""

import os
import time
import math
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline


def make_scale_schedule(schedule, base_scale, num_layers=24):
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


class SD3GenThreeStageLayerCFG:
    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda"):
        self.device = torch.device(device)
        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.num_layers = len(self.pipe.transformer.transformer_blocks)
        print(f"SD3 loaded. {self.num_layers} layers.")

    def _run_layercfg_forward(self, latents, timestep, cond_embeds, neg_embeds,
                               cond_pooled, layer_scales):
        transformer = self.pipe.transformer
        hidden_states = latents
        height, width = hidden_states.shape[-2:]
        hidden_states = transformer.pos_embed(hidden_states)
        temb = transformer.time_text_embed(timestep, cond_pooled)
        enc_cond = transformer.context_embedder(cond_embeds)
        enc_uncond = transformer.context_embedder(neg_embeds)

        for idx, block in enumerate(transformer.transformer_blocks):
            scale_i = layer_scales[idx]
            if block.context_pre_only:
                _, hidden_states = block(hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond, temb=temb)
            elif scale_i == 1.0:
                enc_cond, hidden_states = block(hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond, temb=temb)
            else:
                enc_cond_out, hidden_cond = block(hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond, temb=temb)
                enc_uncond_out, hidden_uncond = block(hidden_states=hidden_states,
                    encoder_hidden_states=enc_uncond, temb=temb)
                hidden_states = hidden_uncond + scale_i * (hidden_cond - hidden_uncond)
                enc_cond = enc_cond_out
                enc_uncond = enc_uncond_out

        hidden_states = transformer.norm_out(hidden_states, temb)
        hidden_states = transformer.proj_out(hidden_states)
        patch_size = transformer.config.patch_size
        h_out = height // patch_size
        w_out = width // patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], h_out, w_out,
            patch_size, patch_size, transformer.out_channels)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            hidden_states.shape[0], transformer.out_channels,
            h_out * patch_size, w_out * patch_size)

    @torch.no_grad()
    def generate(self, prompt, num_steps=25, cfg_scale=7.0,
                 cfg_range=(0.3, 0.8), layercfg_range=(0.8, 1.0),
                 layercfg_scale=6.0, layercfg_schedule="peak_mid",
                 height=512, width=512, seed=42):
        t0 = time.time()
        device = self.pipe._execution_device

        (cond_embeds, neg_embeds,
         cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=prompt, prompt_2=prompt, prompt_3=prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True, device=device)

        generator = torch.Generator(device=device).manual_seed(seed)
        num_channels = self.pipe.transformer.config.in_channels
        latents = torch.randn(1, num_channels, height // self.pipe.vae_scale_factor,
            width // self.pipe.vae_scale_factor, generator=generator, device=device,
            dtype=torch.bfloat16)

        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        total_steps = len(timesteps)

        cfg_start = int(total_steps * cfg_range[0])
        cfg_end = int(total_steps * cfg_range[1])
        lcfg_start = int(total_steps * layercfg_range[0])
        lcfg_end = int(total_steps * layercfg_range[1])

        layer_scales = make_scale_schedule(layercfg_schedule, layercfg_scale, self.num_layers)

        step_modes = []
        for i in range(total_steps):
            if cfg_start <= i < cfg_end:
                step_modes.append("cfg")
            elif lcfg_start <= i < lcfg_end:
                step_modes.append("layercfg")
            else:
                step_modes.append("none")

        for i, t in enumerate(timesteps):
            mode = step_modes[i]
            timestep = t.expand(latents.shape[0])

            if mode == "cfg":
                latent_input = torch.cat([latents] * 2)
                timestep_input = t.expand(latent_input.shape[0])
                prompt_input = torch.cat([neg_embeds, cond_embeds], dim=0)
                pooled_input = torch.cat([neg_pooled, cond_pooled], dim=0)
                noise_pred = self.pipe.transformer(
                    hidden_states=latent_input, timestep=timestep_input,
                    encoder_hidden_states=prompt_input,
                    pooled_projections=pooled_input, return_dict=False)[0]
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            elif mode == "layercfg":
                noise_pred = self._run_layercfg_forward(
                    latents, timestep, cond_embeds, neg_embeds,
                    cond_pooled, layer_scales)
            else:
                noise_pred = self.pipe.transformer(
                    hidden_states=latents, timestep=timestep,
                    encoder_hidden_states=cond_embeds,
                    pooled_projections=cond_pooled, return_dict=False)[0]

            latents_dtype = latents.dtype
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]

        elapsed = time.time() - t0
        return image, elapsed, {"cfg_steps": step_modes.count("cfg"),
                                "layercfg_steps": step_modes.count("layercfg"),
                                "none_steps": step_modes.count("none")}
