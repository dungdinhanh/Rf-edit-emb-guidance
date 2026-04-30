"""
Simple custom per-layer CFG runner.

Uses pipe.to(device) directly — no CPU offloading.
Tests custom per-layer scales with different global multipliers.
"""
import os, sys, io, json, argparse, time
import torch, numpy as np, pandas as pd
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.metrics import compute_clip_score

# Optimal single-layer scales from the sweep
EDIT_OPTIMAL = [1.05, 1.30, 1.00, 1.05, 1.10, 1.05, 1.05, 1.10, 1.20, 1.00,
                1.05, 1.10, 1.10, 1.05, 1.20, 1.05, 1.00, 1.10, 1.20, 1.50,
                1.00, 1.20, 1.20, 1.20]

GEN_OPTIMAL = [1.00, 1.50, 1.10, 1.00, 1.00, 1.00, 1.00, 1.05, 1.05, 1.00,
               1.00, 1.05, 1.00, 1.20, 1.05, 1.05, 1.05, 1.00, 1.05, 1.20,
               1.10, 1.05, 1.30, 1.00]


def scale_custom(base_scales, multiplier):
    """Apply multiplier to guidance portion: new_s = 1 + multiplier * (s - 1)"""
    return [1.0 + multiplier * (s - 1.0) for s in base_scales]


def run_layercfg_step(transformer, latents, timestep, cond_embeds, neg_embeds,
                      cond_pooled, layer_scales):
    """Single denoising step with per-layer full-block CFG."""
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
def run_single(pipe, task, source_img, target_prompt, layer_scales, device):
    """Full denoise one sample with custom per-layer scales."""
    cond_e, neg_e, cond_p, neg_p = pipe.encode_prompt(
        prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
        negative_prompt="", negative_prompt_2="", negative_prompt_3="",
        do_classifier_free_guidance=True, device=device)

    if task == "edit":
        w, h = source_img.size
        w, h = w - w % 16, h - h % 16
        source_img = source_img.resize((w, h))
        image_tensor = pipe.image_processor.preprocess(source_img)
        image_tensor = image_tensor.to(device=device, dtype=pipe.vae.dtype)
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        latents = latents.to(dtype=cond_e.dtype)
        pipe.scheduler.set_timesteps(25, device=device)
        timesteps, _ = pipe.get_timesteps(25, 0.7, device)
        noise = torch.randn_like(latents)
        latents = pipe.scheduler.scale_noise(latents, timesteps[:1], noise)
    else:
        generator = torch.Generator(device=device).manual_seed(42)
        nc = pipe.transformer.config.in_channels
        latents = torch.randn(1, nc, 512 // pipe.vae_scale_factor,
            512 // pipe.vae_scale_factor, generator=generator,
            device=device, dtype=torch.bfloat16)
        pipe.scheduler.set_timesteps(25, device=device)
        timesteps = pipe.scheduler.timesteps

    for t in timesteps:
        timestep = t.expand(latents.shape[0])
        noise_pred = run_layercfg_step(
            pipe.transformer, latents, timestep, cond_e, neg_e, cond_p, layer_scales)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--task', choices=['edit', 'gen'], required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Loading SD3 pipeline ({args.task})...")
    if args.task == "edit":
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print("SD3 loaded.")

    base_scales = EDIT_OPTIMAL if args.task == "edit" else GEN_OPTIMAL
    multipliers = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    # Collect samples
    categories = sorted([d for d in os.listdir(args.dataset_dir)
                        if os.path.isdir(os.path.join(args.dataset_dir, d)) and not d.startswith('.')])
    samples = []
    for cat in categories:
        pq_path = os.path.join(args.dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path): continue
        df = pd.read_parquet(pq_path)
        n = min(len(df), args.max_samples) if args.max_samples else len(df)
        for idx in range(n):
            row = df.iloc[idx]
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            source_img = None
            if args.task == "edit":
                img_data = row['image']
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    source_img = Image.open(io.BytesIO(img_data['bytes']))
                else: continue
            samples.append((source_img, target_prompt, f"{cat}/{row.get('id', idx)}"))

    print(f"Collected {len(samples)} samples")

    results = []
    for mult in multipliers:
        custom = scale_custom(base_scales, mult)
        print(f"\n=== Multiplier={mult} ===")

        out_base = os.path.join(args.output_dir, f"custom_{args.task}_m{mult}")
        clip_scores = []

        for si, (source_img, target_prompt, sample_id) in enumerate(samples):
            cat = sample_id.split("/")[0]
            sid = sample_id.split("/")[1]
            out_dir = os.path.join(out_base, cat)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{sid}.jpg")

            if os.path.exists(out_path):
                img = Image.open(out_path)
                score = compute_clip_score(img, target_prompt, device=device)
                clip_scores.append(score)
                continue

            try:
                edited_img = run_single(pipe, args.task, source_img, target_prompt, custom, device)
                edited_img.save(out_path, quality=95)
                score = compute_clip_score(edited_img, target_prompt, device=device)
                clip_scores.append(score)
                if (si + 1) % 50 == 0:
                    print(f"  [{si+1}/{len(samples)}] avg CLIP={np.mean(clip_scores):.4f}")
            except Exception as e:
                print(f"  [{si+1}] ERROR: {e}")

        mean_clip = np.mean(clip_scores) if clip_scores else 0
        results.append({'multiplier': mult, 'clip': mean_clip, 'n': len(clip_scores)})
        print(f"  RESULT: mult={mult} → CLIP={mean_clip:.4f} (n={len(clip_scores)})")

    print(f"\n{'='*60}")
    print(f"SUMMARY — Custom LayerCFG ({args.task})")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: -x['clip']):
        print(f"  mult={r['multiplier']:.1f}: CLIP={r['clip']:.4f} (n={r['n']})")


if __name__ == "__main__":
    main()
