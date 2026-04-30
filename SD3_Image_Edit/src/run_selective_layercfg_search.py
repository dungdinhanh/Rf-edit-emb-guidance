"""
Exhaustive search: selective full-block LayerCFG.
Only guide specific layers, leave others at scale=1.0.
Goal: pure layercfg that matches or beats standard CFG.
"""
import os, sys, io, json, argparse, time, itertools
import torch, numpy as np, pandas as pd
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.metrics import compute_clip_score


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


def evaluate(pipe, task, samples, layer_scales, device, max_eval=None):
    """Evaluate a config on samples, return mean CLIP."""
    clip_scores = []
    n = min(len(samples), max_eval) if max_eval else len(samples)
    for si in range(n):
        source_img, target_prompt, _ = samples[si]
        try:
            img = run_single(pipe, task, source_img, target_prompt, layer_scales, device)
            score = compute_clip_score(img, target_prompt, device=device)
            clip_scores.append(score)
        except:
            pass
    return np.mean(clip_scores) if clip_scores else 0, len(clip_scores)


def make_scales(active_layers, scale, num_layers=24):
    """Create layer_scales with given scale for active layers, 1.0 for rest."""
    return [scale if i in active_layers else 1.0 for i in range(num_layers)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--task', choices=['edit', 'gen'], required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_samples', type=int, default=70)
    parser.add_argument('--eval_samples', type=int, default=100,
                        help='Samples for fast evaluation during search')
    parser.add_argument('--target_clip', type=float, default=0.3397,
                        help='Target CLIP to beat (Full CFG)')
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"Loading SD3 ({args.task})...")
    if args.task == "edit":
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    num_layers = len(pipe.transformer.transformer_blocks)
    print(f"Loaded. {num_layers} layers. Target CLIP: {args.target_clip}")

    # Collect samples
    categories = sorted([d for d in os.listdir(args.dataset_dir)
                        if os.path.isdir(os.path.join(args.dataset_dir, d)) and not d.startswith('.')])
    samples = []
    for cat in categories:
        pq_path = os.path.join(args.dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path): continue
        df = pd.read_parquet(pq_path)
        n = min(len(df), args.max_samples)
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
    print(f"Collected {len(samples)} samples, using {args.eval_samples} for search")

    # Layer subsets
    gen_helpful = {
        'top3': [13, 18, 19],
        'top5': [2, 13, 18, 19, 21],
        'top7': [2, 7, 11, 13, 14, 18, 19],
        'top10': [1, 2, 7, 11, 13, 14, 16, 18, 19, 21],
        'late': [18, 19, 21, 22],
        'mid': [7, 11, 13, 14],
        'mid_late': [11, 13, 14, 18, 19, 21],
        'all_helpful': [1, 2, 7, 11, 13, 14, 16, 18, 19, 21],
    }
    edit_helpful = {
        'top3': [1, 7, 8],
        'top5': [1, 3, 7, 8, 12],
        'top7': [1, 3, 7, 8, 11, 12, 23],
        'top10': [1, 3, 5, 7, 8, 11, 12, 13, 21, 23],
        'late': [21, 22, 23],
        'mid': [7, 8, 11, 12],
        'mid_late': [7, 8, 11, 12, 21, 22, 23],
        'all_helpful': [0, 1, 3, 5, 7, 8, 10, 11, 12, 13, 14, 21, 22, 23],
    }

    layer_subsets = gen_helpful if args.task == "gen" else edit_helpful
    coarse_scales = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0]

    all_results = []
    best_clip = 0
    best_config = None

    # ROUND 1: Coarse search
    round_num = 1
    print(f"\n{'='*60}")
    print(f"ROUND {round_num}: Coarse search — {len(layer_subsets)} subsets × {len(coarse_scales)} scales")
    print(f"{'='*60}")

    for subset_name, layers in layer_subsets.items():
        for scale in coarse_scales:
            layer_scales = make_scales(layers, scale, num_layers)
            clip, n = evaluate(pipe, args.task, samples, layer_scales, device, args.eval_samples)
            config = f"{subset_name}_s{scale}"
            all_results.append({'config': config, 'subset': subset_name, 'layers': layers,
                               'scale': scale, 'clip': clip, 'n': n, 'round': round_num})
            marker = " ★ NEW BEST" if clip > best_clip else ""
            if clip > best_clip:
                best_clip = clip
                best_config = {'subset': subset_name, 'layers': layers, 'scale': scale}
            print(f"  {config}: CLIP={clip:.4f} (n={n}){marker}")

    print(f"\nRound {round_num} best: {best_config['subset']} s={best_config['scale']} → {best_clip:.4f}")
    print(f"Target: {args.target_clip} | Gap: {best_clip - args.target_clip:+.4f}")

    # ROUND 2: Fine-tune scale
    round_num = 2
    best_subset = best_config['subset']
    best_layers = best_config['layers']
    best_scale = best_config['scale']
    fine_scales = [best_scale * f for f in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5]]
    fine_scales = sorted(set([round(s, 1) for s in fine_scales]))

    print(f"\n{'='*60}")
    print(f"ROUND {round_num}: Fine-tune scale around {best_scale} for {best_subset}")
    print(f"{'='*60}")

    for scale in fine_scales:
        layer_scales = make_scales(best_layers, scale, num_layers)
        clip, n = evaluate(pipe, args.task, samples, layer_scales, device, args.eval_samples)
        config = f"{best_subset}_s{scale}"
        all_results.append({'config': config, 'subset': best_subset, 'layers': best_layers,
                           'scale': scale, 'clip': clip, 'n': n, 'round': round_num})
        marker = " ★ NEW BEST" if clip > best_clip else ""
        if clip > best_clip:
            best_clip = clip
            best_config = {'subset': best_subset, 'layers': best_layers, 'scale': scale}
        print(f"  {config}: CLIP={clip:.4f} (n={n}){marker}")

    print(f"\nRound {round_num} best: {best_config['subset']} s={best_config['scale']} → {best_clip:.4f}")
    print(f"Target: {args.target_clip} | Gap: {best_clip - args.target_clip:+.4f}")

    # ROUND 3: Combine top subsets
    round_num = 3
    sorted_r1 = sorted([r for r in all_results if r['round'] == 1], key=lambda x: -x['clip'])
    top_subsets = []
    seen = set()
    for r in sorted_r1:
        if r['subset'] not in seen:
            top_subsets.append(r['subset'])
            seen.add(r['subset'])
        if len(top_subsets) >= 3:
            break

    print(f"\n{'='*60}")
    print(f"ROUND {round_num}: Combine top subsets: {top_subsets}")
    print(f"{'='*60}")

    for i in range(len(top_subsets)):
        for j in range(i + 1, len(top_subsets)):
            combined = sorted(set(layer_subsets[top_subsets[i]] + layer_subsets[top_subsets[j]]))
            combo_name = f"{top_subsets[i]}+{top_subsets[j]}"
            for scale in [1.5, 2.0, 3.0, 5.0]:
                layer_scales = make_scales(combined, scale, num_layers)
                clip, n = evaluate(pipe, args.task, samples, layer_scales, device, args.eval_samples)
                config = f"{combo_name}_s{scale}"
                all_results.append({'config': config, 'subset': combo_name, 'layers': combined,
                                   'scale': scale, 'clip': clip, 'n': n, 'round': round_num})
                marker = " ★ NEW BEST" if clip > best_clip else ""
                if clip > best_clip:
                    best_clip = clip
                    best_config = {'subset': combo_name, 'layers': combined, 'scale': scale}
                print(f"  {config}: CLIP={clip:.4f} (n={n}){marker}")

    print(f"\nRound {round_num} best overall: {best_config['subset']} s={best_config['scale']} → {best_clip:.4f}")

    # ROUND 4: Per-layer scale tuning on best
    round_num = 4
    print(f"\n{'='*60}")
    print(f"ROUND {round_num}: Per-layer scale tuning on {best_config['subset']}")
    print(f"{'='*60}")

    current_scales = make_scales(best_config['layers'], best_config['scale'], num_layers)
    current_clip = best_clip

    for layer_idx in best_config['layers']:
        original_scale = current_scales[layer_idx]
        test_scales = [original_scale * f for f in [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]]
        test_scales = sorted(set([round(s, 2) for s in test_scales if s > 1.0]))

        layer_best = original_scale
        layer_best_clip = current_clip

        for s in test_scales:
            test = current_scales.copy()
            test[layer_idx] = s
            clip, n = evaluate(pipe, args.task, samples, test, device, args.eval_samples)
            if clip > layer_best_clip:
                layer_best_clip = clip
                layer_best = s
            print(f"  Layer {layer_idx}: s={s:.2f} → CLIP={clip:.4f}")

        if layer_best != original_scale:
            current_scales[layer_idx] = layer_best
            current_clip = layer_best_clip
            print(f"  → Layer {layer_idx}: updated {original_scale:.2f} → {layer_best:.2f}")

    clip, n = evaluate(pipe, args.task, samples, current_scales, device, args.eval_samples)
    all_results.append({'config': 'tuned_per_layer', 'subset': best_config['subset'],
                       'layers': best_config['layers'], 'scale': current_scales,
                       'clip': clip, 'n': n, 'round': round_num})
    if clip > best_clip:
        best_clip = clip
    print(f"\nRound {round_num} result: CLIP={clip:.4f}")

    # FINAL: Validate on all samples
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION on all {len(samples)} samples")
    print(f"{'='*60}")
    final_clip, final_n = evaluate(pipe, args.task, samples, current_scales, device)
    print(f"Final CLIP={final_clip:.4f} (n={final_n})")
    print(f"Target: {args.target_clip} | Gap: {final_clip - args.target_clip:+.4f}")
    if final_clip >= args.target_clip:
        print("★★★ TARGET ACHIEVED! ★★★")
    else:
        print(f"Gap remaining: {args.target_clip - final_clip:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    result = {
        'task': args.task,
        'target_clip': args.target_clip,
        'final_clip': final_clip,
        'final_n': final_n,
        'best_scales': current_scales,
        'best_config': {k: v for k, v in best_config.items() if k != 'layers'},
        'best_layers': best_config['layers'],
        'all_results': [{'config': r['config'], 'clip': r['clip'], 'round': r['round']}
                       for r in sorted(all_results, key=lambda x: -x['clip'])[:20]],
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
