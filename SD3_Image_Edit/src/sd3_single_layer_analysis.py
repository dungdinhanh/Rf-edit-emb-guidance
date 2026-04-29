"""
Single-layer full-block CFG analysis.

For each of the 24 layers, enable full-block CFG on ONLY that one layer
(all others use cond-only). Measures:
1. CLIP score — which layer contributes most to edit quality
2. Output difference vs cond-only — how much each layer changes the output
3. Output difference vs full CFG — which layer moves output toward CFG result

Supports both editing and generation tasks.
"""

import os, sys, io, json, time, math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.metrics import compute_clip_score


class SingleLayerCFGRunner:
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
        print(f"SD3 loaded. {self.num_layers} layers.")

    def _run_with_layer_cfg(self, latents, timestep, cond_embeds, neg_embeds,
                            cond_pooled, neg_pooled, cfg_layer_idx, cfg_scale=7.0):
        """Run transformer with full-block CFG on only one layer."""
        transformer = self.pipe.transformer
        hidden_states = latents
        height, width = hidden_states.shape[-2:]
        hidden_states = transformer.pos_embed(hidden_states)
        temb = transformer.time_text_embed(timestep, cond_pooled)
        enc_cond = transformer.context_embedder(cond_embeds)
        enc_uncond = transformer.context_embedder(neg_embeds)

        for idx, block in enumerate(transformer.transformer_blocks):
            if idx == cfg_layer_idx and not block.context_pre_only:
                # Full-block CFG on this layer
                enc_cond_out, hidden_cond = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond,
                    temb=temb)
                enc_uncond_out, hidden_uncond = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=enc_uncond,
                    temb=temb)
                hidden_states = hidden_uncond + cfg_scale * (hidden_cond - hidden_uncond)
                enc_cond = enc_cond_out
                enc_uncond = enc_uncond_out
            elif block.context_pre_only:
                _, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond,
                    temb=temb)
            else:
                enc_cond, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=enc_cond,
                    temb=temb)

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

    def _run_cond_only(self, latents, timestep, cond_embeds, cond_pooled):
        """Standard cond-only forward (no guidance)."""
        return self.pipe.transformer(
            hidden_states=latents, timestep=timestep,
            encoder_hidden_states=cond_embeds,
            pooled_projections=cond_pooled,
            return_dict=False)[0]

    def _run_standard_cfg(self, latents, timestep, cond_embeds, neg_embeds,
                          cond_pooled, neg_pooled, cfg_scale=7.0):
        """Standard CFG (batch=2)."""
        latent_input = torch.cat([latents] * 2)
        timestep_input = timestep.expand(latent_input.shape[0])
        prompt_input = torch.cat([neg_embeds, cond_embeds], dim=0)
        pooled_input = torch.cat([neg_pooled, cond_pooled], dim=0)
        noise_pred = self.pipe.transformer(
            hidden_states=latent_input, timestep=timestep_input,
            encoder_hidden_states=prompt_input,
            pooled_projections=pooled_input,
            return_dict=False)[0]
        uncond, cond = noise_pred.chunk(2)
        return uncond + cfg_scale * (cond - uncond)

    @torch.no_grad()
    def analyse_sample(self, source_image=None, target_prompt="",
                       num_steps=25, strength=0.7, cfg_scale=7.0, seed=42):
        """Run all 24 single-layer configs + cond-only + standard CFG for one sample."""
        device = self.pipe._execution_device

        (cond_embeds, neg_embeds, cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True, device=device)

        if self.task == "edit":
            w, h = source_image.size
            w, h = w - w % 16, h - h % 16
            source_image = source_image.resize((w, h))
            image_tensor = self.pipe.image_processor.preprocess(source_image)
            image_tensor = image_tensor.to(device=device, dtype=self.pipe.vae.dtype)
            latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
            latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            latents = latents.to(dtype=cond_embeds.dtype)
            self.pipe.scheduler.set_timesteps(num_steps, device=device)
            timesteps, _ = self.pipe.get_timesteps(num_steps, strength, device)
            noise = torch.randn_like(latents)
            latents = self.pipe.scheduler.scale_noise(latents, timesteps[:1], noise)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
            num_channels = self.pipe.transformer.config.in_channels
            latents = torch.randn(1, num_channels, 512 // self.pipe.vae_scale_factor,
                512 // self.pipe.vae_scale_factor, generator=generator, device=device,
                dtype=torch.bfloat16)
            self.pipe.scheduler.set_timesteps(num_steps, device=device)
            timesteps = self.pipe.scheduler.timesteps

        # Pick middle timestep for single-step analysis
        t_idx = len(timesteps) // 2
        t = timesteps[t_idx].unsqueeze(0)

        # Baseline: cond-only prediction
        pred_cond = self._run_cond_only(latents, t, cond_embeds, cond_pooled).float()

        # Standard CFG prediction
        pred_cfg = self._run_standard_cfg(latents, t, cond_embeds, neg_embeds,
                                          cond_pooled, neg_pooled, cfg_scale).float()

        # Per-layer single-layer CFG predictions
        results = []
        for layer_idx in range(self.num_layers):
            pred_layer = self._run_with_layer_cfg(
                latents, t, cond_embeds, neg_embeds,
                cond_pooled, neg_pooled, layer_idx, cfg_scale).float()

            # Metrics
            diff_from_cond = (pred_layer - pred_cond).norm().item()
            diff_from_cfg = (pred_layer - pred_cfg).norm().item()
            cond_norm = pred_cond.norm().item()
            cfg_norm = pred_cfg.norm().item()
            layer_norm = pred_layer.norm().item()
            # Cosine similarity with CFG
            cos_with_cfg = torch.nn.functional.cosine_similarity(
                pred_layer.flatten().unsqueeze(0),
                pred_cfg.flatten().unsqueeze(0)).item()

            results.append({
                'layer': layer_idx,
                'diff_from_cond': diff_from_cond,
                'diff_from_cond_ratio': diff_from_cond / (cond_norm + 1e-8),
                'diff_from_cfg': diff_from_cfg,
                'diff_from_cfg_ratio': diff_from_cfg / (cfg_norm + 1e-8),
                'cos_with_cfg': cos_with_cfg,
                'pred_norm': layer_norm,
            })

        # Also compute full-denoise images for CLIP
        return results, pred_cond, pred_cfg, {
            'cond_norm': cond_norm,
            'cfg_norm': cfg_norm,
            'cfg_cond_diff': (pred_cfg - pred_cond).norm().item(),
        }

    @torch.no_grad()
    def full_denoise_single_layer(self, source_image=None, target_prompt="",
                                   num_steps=25, strength=0.7, cfg_scale=7.0,
                                   cfg_layer_idx=-1, seed=42):
        """Full denoising with single-layer CFG for CLIP measurement."""
        device = self.pipe._execution_device

        (cond_embeds, neg_embeds, cond_pooled, neg_pooled) = self.pipe.encode_prompt(
            prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
            negative_prompt="", negative_prompt_2="", negative_prompt_3="",
            do_classifier_free_guidance=True, device=device)

        if self.task == "edit":
            w, h = source_image.size
            w, h = w - w % 16, h - h % 16
            source_image = source_image.resize((w, h))
            image_tensor = self.pipe.image_processor.preprocess(source_image)
            image_tensor = image_tensor.to(device=device, dtype=self.pipe.vae.dtype)
            latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
            latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            latents = latents.to(dtype=cond_embeds.dtype)
            self.pipe.scheduler.set_timesteps(num_steps, device=device)
            timesteps, _ = self.pipe.get_timesteps(num_steps, strength, device)
            noise = torch.randn_like(latents)
            latents = self.pipe.scheduler.scale_noise(latents, timesteps[:1], noise)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
            num_channels = self.pipe.transformer.config.in_channels
            latents = torch.randn(1, num_channels, 512 // self.pipe.vae_scale_factor,
                512 // self.pipe.vae_scale_factor, generator=generator, device=device,
                dtype=torch.bfloat16)
            self.pipe.scheduler.set_timesteps(num_steps, device=device)
            timesteps = self.pipe.scheduler.timesteps

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])
            if cfg_layer_idx >= 0:
                noise_pred = self._run_with_layer_cfg(
                    latents, timestep, cond_embeds, neg_embeds,
                    cond_pooled, neg_pooled, cfg_layer_idx, cfg_scale)
            elif cfg_layer_idx == -1:
                # Cond only
                noise_pred = self._run_cond_only(latents, timestep, cond_embeds, cond_pooled)
            elif cfg_layer_idx == -2:
                # Standard CFG
                noise_pred = self._run_standard_cfg(latents, timestep, cond_embeds, neg_embeds,
                                                    cond_pooled, neg_pooled, cfg_scale)
            latents_dtype = latents.dtype
            latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        latents = latents / self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor
        image = self.pipe.vae.decode(latents, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--task', choices=['edit', 'gen'], default='edit')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    args = parser.parse_args()

    device = torch.device("cuda")
    runner = SingleLayerCFGRunner(model_id=args.model_id, task=args.task)
    num_layers = runner.num_layers

    categories = sorted([d for d in os.listdir(args.dataset_dir)
                        if os.path.isdir(os.path.join(args.dataset_dir, d)) and not d.startswith('.')])

    # Phase 1: Single-step prediction analysis
    print("\n=== Phase 1: Single-step prediction analysis ===")
    all_results = [[] for _ in range(num_layers)]
    all_baselines = []
    sample_count = 0

    for cat in categories:
        pq_path = os.path.join(args.dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path): continue
        df = pd.read_parquet(pq_path)
        n = min(len(df), args.max_samples)

        for idx in range(n):
            row = df.iloc[idx]
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            img_data = row['image']
            source_img = None
            if args.task == "edit":
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    source_img = Image.open(io.BytesIO(img_data['bytes']))
                else: continue

            results, _, _, baseline = runner.analyse_sample(
                source_image=source_img, target_prompt=target_prompt,
                cfg_scale=args.cfg_scale)

            for r in results:
                all_results[r['layer']].append(r)
            all_baselines.append(baseline)
            sample_count += 1
            print(f"  [{sample_count}] {cat}/{row.get('id', idx)}")

    # Phase 2: Full denoise CLIP for key layers + baselines
    print(f"\n=== Phase 2: CLIP measurement ({sample_count} samples) ===")
    clip_scores = {-2: [], -1: []}  # -2=standard CFG, -1=cond only
    for li in range(num_layers):
        clip_scores[li] = []

    sample_idx = 0
    for cat in categories:
        pq_path = os.path.join(args.dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path): continue
        df = pd.read_parquet(pq_path)
        n = min(len(df), args.max_samples)

        for idx in range(n):
            row = df.iloc[idx]
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            img_data = row['image']
            source_img = None
            if args.task == "edit":
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    source_img = Image.open(io.BytesIO(img_data['bytes']))
                else: continue

            # Standard CFG
            img_cfg = runner.full_denoise_single_layer(
                source_image=source_img, target_prompt=target_prompt,
                cfg_scale=args.cfg_scale, cfg_layer_idx=-2)
            clip_scores[-2].append(compute_clip_score(img_cfg, target_prompt, device=device))

            # Cond only
            img_cond = runner.full_denoise_single_layer(
                source_image=source_img, target_prompt=target_prompt,
                cfg_layer_idx=-1)
            clip_scores[-1].append(compute_clip_score(img_cond, target_prompt, device=device))

            # Each single layer
            for li in range(num_layers):
                img_layer = runner.full_denoise_single_layer(
                    source_image=source_img, target_prompt=target_prompt,
                    cfg_scale=args.cfg_scale, cfg_layer_idx=li)
                clip_scores[li].append(compute_clip_score(img_layer, target_prompt, device=device))

            sample_idx += 1
            print(f"  [{sample_idx}] done")

    # Write report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"single_layer_analysis_{args.task}.md")

    with open(report_path, 'w') as f:
        f.write(f"# Single-Layer Full-Block CFG Analysis ({args.task})\n\n")
        f.write(f"CFG scale={args.cfg_scale}, {sample_count} samples, middle timestep for prediction analysis.\n\n")

        # Phase 1 results
        f.write("## 1. Single-Step Prediction: How Each Layer Moves the Output\n\n")
        f.write("| Layer | Diff from Cond (norm) | Diff from Cond (%) | Diff from CFG (norm) | Diff from CFG (%) | Cos(layer, CFG) |\n")
        f.write("|:-----:|:--------------------:|:-----------------:|:-------------------:|:----------------:|:---------------:|\n")

        for li in range(num_layers):
            entries = all_results[li]
            if not entries: continue
            dc = np.mean([e['diff_from_cond'] for e in entries])
            dcr = np.mean([e['diff_from_cond_ratio'] for e in entries])
            df_val = np.mean([e['diff_from_cfg'] for e in entries])
            dfr = np.mean([e['diff_from_cfg_ratio'] for e in entries])
            cos = np.mean([e['cos_with_cfg'] for e in entries])
            f.write(f"| {li:2d} | {dc:.1f} | {dcr*100:.1f}% | {df_val:.1f} | {dfr*100:.1f}% | {cos:.4f} |\n")

        avg_cfg_diff = np.mean([b['cfg_cond_diff'] for b in all_baselines])
        f.write(f"\nStandard CFG - Cond diff: {avg_cfg_diff:.1f}\n\n")

        # Phase 2 results
        f.write("## 2. Full Denoising CLIP Scores: Contribution of Each Layer\n\n")
        mean_cfg = np.mean(clip_scores[-2])
        mean_cond = np.mean(clip_scores[-1])
        f.write(f"| Config | CLIP | vs Cond-only | vs Standard CFG |\n")
        f.write(f"|:-------|:----:|:------------:|:---------------:|\n")
        f.write(f"| **Standard CFG** | **{mean_cfg:.4f}** | +{mean_cfg-mean_cond:.4f} | — |\n")
        f.write(f"| Cond-only (no guidance) | {mean_cond:.4f} | — | {mean_cond-mean_cfg:.4f} |\n")

        layer_clips = []
        for li in range(num_layers):
            mc = np.mean(clip_scores[li]) if clip_scores[li] else 0
            layer_clips.append(mc)
            delta_cond = mc - mean_cond
            delta_cfg = mc - mean_cfg
            f.write(f"| Layer {li:2d} only | {mc:.4f} | {delta_cond:+.4f} | {delta_cfg:+.4f} |\n")

        # Find best/worst layers
        sorted_layers = sorted(range(num_layers), key=lambda i: layer_clips[i], reverse=True)
        f.write(f"\n### Top 5 most impactful layers:\n")
        for rank, li in enumerate(sorted_layers[:5]):
            f.write(f"  {rank+1}. Layer {li}: CLIP={layer_clips[li]:.4f} (+{layer_clips[li]-mean_cond:.4f} vs cond)\n")

        f.write(f"\n### Bottom 5 (least impactful or harmful):\n")
        for rank, li in enumerate(sorted_layers[-5:]):
            f.write(f"  {num_layers-4+rank}. Layer {li}: CLIP={layer_clips[li]:.4f} ({layer_clips[li]-mean_cond:+.4f} vs cond)\n")

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
