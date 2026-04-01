"""
SD3 Image Editing with RF-Solver approach.

Uses Stable Diffusion 3 Medium (no guidance distillation, standard CFG).
Supports:
  - Baseline: standard CFG (2 forward passes per step)
  - Embedding guidance: single forward pass with blended embeddings
"""

import os
import time
import argparse
import json

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler


class SD3Editor:
    """RF-Solver based image editor using SD3."""

    def __init__(self, model_id="stabilityai/stable-diffusion-3-medium-diffusers",
                 device="cuda", dtype=torch.float16, offload=False):
        self.device = torch.device(device)
        self.dtype = dtype

        print("Loading SD3 pipeline...")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=dtype,
        )

        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)

        self.pipe.set_progress_bar_config(disable=True)
        print("SD3 loaded.")

    @torch.no_grad()
    def encode_image(self, image):
        """Encode PIL image to latent space."""
        image = image.convert("RGB")
        w, h = image.size
        # Round to multiple of 16
        w = w - w % 16
        h = h - h % 16
        image = image.resize((w, h))

        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

        latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        return latents, (h, w)

    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to PIL image."""
        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype(np.uint8)
        return Image.fromarray(image[0])

    @torch.no_grad()
    def encode_prompt(self, prompt):
        """Encode text prompt using SD3's 3 text encoders."""
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=self.device,
        )
        return {
            'cond_embeds': prompt_embeds,
            'uncond_embeds': negative_prompt_embeds,
            'cond_pooled': pooled_prompt_embeds,
            'uncond_pooled': negative_pooled_prompt_embeds,
        }

    def get_timesteps(self, num_steps):
        """Get flow matching timesteps."""
        self.pipe.scheduler.set_timesteps(num_steps, device=self.device)
        return self.pipe.scheduler.timesteps

    @torch.no_grad()
    def inversion(self, latents, prompt_embeds, pooled_embeds, num_steps=25):
        """RF-Solver inversion: image -> noise using source prompt."""
        timesteps = self.get_timesteps(num_steps)
        # Reverse for inversion (0 -> 1)
        timesteps_inv = timesteps.flip(0)

        z = latents
        for i in range(len(timesteps_inv) - 1):
            t_curr = timesteps_inv[i]
            t_next = timesteps_inv[i + 1]

            t_input = t_curr.unsqueeze(0).expand(z.shape[0])

            # Single forward pass (no CFG during inversion)
            noise_pred = self.pipe.transformer(
                hidden_states=z,
                timestep=t_input,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                return_dict=False,
            )[0]

            # Euler step (forward: image -> noise)
            dt = (t_next - t_curr)
            z = z + dt * noise_pred

        return z

    @torch.no_grad()
    def denoise_cfg(self, latents, cond_embeds, uncond_embeds,
                    cond_pooled, uncond_pooled,
                    num_steps=25, guidance_scale=7.0, inject_step=4,
                    source_features=None):
        """Standard CFG denoising: 2 forward passes per step."""
        timesteps = self.get_timesteps(num_steps)

        z = latents
        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            t_input = t_curr.unsqueeze(0).expand(z.shape[0])

            # Unconditional forward pass
            noise_uncond = self.pipe.transformer(
                hidden_states=z,
                timestep=t_input,
                encoder_hidden_states=uncond_embeds,
                pooled_projections=uncond_pooled,
                return_dict=False,
            )[0]

            # Conditional forward pass
            noise_cond = self.pipe.transformer(
                hidden_states=z,
                timestep=t_input,
                encoder_hidden_states=cond_embeds,
                pooled_projections=cond_pooled,
                return_dict=False,
            )[0]

            # CFG combination
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Euler step (reverse: noise -> image)
            dt = (t_next - t_curr)
            z = z + dt * noise_pred

        return z

    @torch.no_grad()
    def denoise_emb_guidance(self, latents, cond_embeds, uncond_embeds,
                             cond_pooled, uncond_pooled,
                             num_steps=25, guidance_scale=7.0,
                             emb_alpha=0.5, inject_step=4,
                             source_features=None):
        """Embedding guidance denoising: 1 forward pass per step."""
        timesteps = self.get_timesteps(num_steps)

        # Blend embeddings
        guided_embeds = (1.0 + emb_alpha) * cond_embeds - emb_alpha * uncond_embeds
        guided_pooled = (1.0 + emb_alpha) * cond_pooled - emb_alpha * uncond_pooled

        z = latents
        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            t_input = t_curr.unsqueeze(0).expand(z.shape[0])

            # Single forward pass with guided embeddings
            noise_pred = self.pipe.transformer(
                hidden_states=z,
                timestep=t_input,
                encoder_hidden_states=guided_embeds,
                pooled_projections=guided_pooled,
                return_dict=False,
            )[0]

            # Euler step
            dt = (t_next - t_curr)
            z = z + dt * noise_pred

        return z

    def edit(self, source_image, source_prompt, target_prompt,
             mode="cfg", num_steps=25, guidance_scale=7.0,
             emb_alpha=0.5, inject_step=4):
        """Full editing pipeline: encode -> invert -> denoise -> decode."""
        t0 = time.time()

        # Encode image
        latents, (h, w) = self.encode_image(source_image)

        # Encode prompts
        source_enc = self.encode_prompt(source_prompt)
        target_enc = self.encode_prompt(target_prompt)

        # Inversion with source prompt (no CFG)
        noise = self.inversion(
            latents, source_enc['cond_embeds'], source_enc['cond_pooled'],
            num_steps=num_steps,
        )

        # Denoising with target prompt
        if mode == "cfg":
            edited_latents = self.denoise_cfg(
                noise,
                target_enc['cond_embeds'], target_enc['uncond_embeds'],
                target_enc['cond_pooled'], target_enc['uncond_pooled'],
                num_steps=num_steps, guidance_scale=guidance_scale,
                inject_step=inject_step,
            )
        elif mode == "emb_guidance":
            edited_latents = self.denoise_emb_guidance(
                noise,
                target_enc['cond_embeds'], target_enc['uncond_embeds'],
                target_enc['cond_pooled'], target_enc['uncond_pooled'],
                num_steps=num_steps, guidance_scale=guidance_scale,
                emb_alpha=emb_alpha, inject_step=inject_step,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Decode
        edited_image = self.decode_latents(edited_latents)

        elapsed = time.time() - t0
        return edited_image, elapsed


def main(args):
    editor = SD3Editor(
        model_id=args.model_id,
        device="cuda",
        dtype=torch.float16,
        offload=args.offload,
    )

    source_image = Image.open(args.source_img).convert('RGB')

    print(f"Mode: {args.mode}, Steps: {args.num_steps}, Guidance: {args.guidance_scale}")
    if args.mode == "emb_guidance":
        print(f"Alpha: {args.emb_alpha}")

    edited_image, elapsed = editor.edit(
        source_image,
        args.source_prompt,
        args.target_prompt,
        mode=args.mode,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        emb_alpha=args.emb_alpha,
        inject_step=args.inject_step,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.mode}_g{args.guidance_scale}_a{args.emb_alpha}.jpg")
    edited_image.save(out_path, quality=95)
    print(f"Saved: {out_path} ({elapsed:.1f}s)")

    # Save metadata
    meta = {
        'mode': args.mode,
        'source_prompt': args.source_prompt,
        'target_prompt': args.target_prompt,
        'guidance_scale': args.guidance_scale,
        'emb_alpha': args.emb_alpha if args.mode == 'emb_guidance' else None,
        'num_steps': args.num_steps,
        'inject_step': args.inject_step,
        'elapsed': elapsed,
    }
    with open(out_path.replace('.jpg', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SD3 Image Editing with RF-Solver')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--source_img', required=True)
    parser.add_argument('--source_prompt', default='')
    parser.add_argument('--target_prompt', required=True)
    parser.add_argument('--mode', choices=['cfg', 'emb_guidance'], default='cfg')
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--guidance_scale', type=float, default=7.0)
    parser.add_argument('--emb_alpha', type=float, default=0.5)
    parser.add_argument('--inject_step', type=int, default=4)
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--offload', action='store_true')
    args = parser.parse_args()
    main(args)
