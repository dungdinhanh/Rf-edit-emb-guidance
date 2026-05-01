"""
Dual-stream CFG with intermediate merge.

Two parallel streams (cond + uncond) run independently.
At merge layer(s), combine with CFG equation.
After last merge, single stream continues.

Phase 1: Single merge point sweep (K=0..23)
Phase 2: Two merge points (non-consecutive)
Phase 3: Three merge points
Phase 4: Scale tuning at best merge points
"""
import os, sys, io, json, argparse, time
import torch, numpy as np, pandas as pd
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.metrics import compute_clip_score


def run_dualstream_merge(transformer, latents, timestep, cond_e, neg_e,
                         cond_p, neg_p, merge_config, cfg_scale=7.0):
    """
    Run dual-stream forward with merge at specified layers.

    merge_config: dict of {layer_idx: scale} — layers where streams merge.
        At merge layers: hidden = uncond + scale*(cond - uncond), then SPLIT back.
        If it's the last merge layer, stream stays single (no split back).

    Between merge layers, both streams run independently (natural evolution).
    """
    height, width = latents.shape[-2:]

    # Initialize both streams from same input
    hidden_cond = transformer.pos_embed(latents)
    hidden_uncond = hidden_cond.clone()

    temb_cond = transformer.time_text_embed(timestep, cond_p)
    temb_uncond = transformer.time_text_embed(timestep, neg_p)

    enc_cond = transformer.context_embedder(cond_e)
    enc_uncond = transformer.context_embedder(neg_e)

    merge_layers = sorted(merge_config.keys()) if isinstance(merge_config, dict) else []
    last_merge = max(merge_layers) if merge_layers else -1
    merged = False  # once merged and not split, stay single

    for idx, block in enumerate(transformer.transformer_blocks):
        if merged:
            # Single stream after final merge
            if block.context_pre_only:
                _, hidden_cond = block(hidden_states=hidden_cond,
                    encoder_hidden_states=enc_cond, temb=temb_cond)
            else:
                enc_cond, hidden_cond = block(hidden_states=hidden_cond,
                    encoder_hidden_states=enc_cond, temb=temb_cond)
        else:
            # Dual stream — run both paths
            if block.context_pre_only:
                _, hidden_cond = block(hidden_states=hidden_cond,
                    encoder_hidden_states=enc_cond, temb=temb_cond)
                _, hidden_uncond = block(hidden_states=hidden_uncond,
                    encoder_hidden_states=enc_uncond, temb=temb_uncond)
            else:
                enc_cond, hidden_cond = block(hidden_states=hidden_cond,
                    encoder_hidden_states=enc_cond, temb=temb_cond)
                enc_uncond, hidden_uncond = block(hidden_states=hidden_uncond,
                    encoder_hidden_states=enc_uncond, temb=temb_uncond)

            # Check if this is a merge point
            if idx in merge_config:
                scale = merge_config[idx]
                hidden_merged = hidden_uncond + scale * (hidden_cond - hidden_uncond)

                if idx == last_merge:
                    # Final merge — stay single stream
                    hidden_cond = hidden_merged
                    merged = True
                else:
                    # Intermediate merge — split back to dual stream
                    hidden_cond = hidden_merged.clone()
                    hidden_uncond = hidden_merged.clone()

    # If never merged (empty config), combine at prediction level like standard CFG
    if not merged and not merge_layers:
        # Standard CFG at prediction level
        h_cond = transformer.norm_out(hidden_cond, temb_cond)
        h_cond = transformer.proj_out(h_cond)
        h_uncond = transformer.norm_out(hidden_uncond, temb_uncond)
        h_uncond = transformer.proj_out(h_uncond)

        patch_size = transformer.config.patch_size
        h_out = height // patch_size
        w_out = width // patch_size

        def reshape(h):
            h = h.reshape(h.shape[0], h_out, w_out, patch_size, patch_size, transformer.out_channels)
            h = torch.einsum("nhwpqc->nchpwq", h)
            return h.reshape(h.shape[0], transformer.out_channels, h_out * patch_size, w_out * patch_size)

        pred_cond = reshape(h_cond)
        pred_uncond = reshape(h_uncond)
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

    # Single stream prediction
    hidden = hidden_cond
    hidden = transformer.norm_out(hidden, temb_cond)
    hidden = transformer.proj_out(hidden)
    patch_size = transformer.config.patch_size
    h_out = height // patch_size
    w_out = width // patch_size
    hidden = hidden.reshape(hidden.shape[0], h_out, w_out, patch_size, patch_size, transformer.out_channels)
    hidden = torch.einsum("nhwpqc->nchpwq", hidden)
    return hidden.reshape(hidden.shape[0], transformer.out_channels, h_out * patch_size, w_out * patch_size)


@torch.no_grad()
def run_single_sample(pipe, task, source_img, target_prompt, merge_config, cfg_scale, device):
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
            512 // pipe.vae_scale_factor, generator=generator, device=device, dtype=torch.bfloat16)
        pipe.scheduler.set_timesteps(25, device=device)
        timesteps = pipe.scheduler.timesteps

    for t in timesteps:
        timestep = t.expand(latents.shape[0])
        noise_pred = run_dualstream_merge(
            pipe.transformer, latents, timestep, cond_e, neg_e, cond_p, neg_p,
            merge_config, cfg_scale)
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def evaluate(pipe, task, samples, merge_config, cfg_scale, device, max_eval=None):
    clip_scores = []
    n = min(len(samples), max_eval) if max_eval else len(samples)
    for si in range(n):
        source_img, target_prompt, _ = samples[si]
        try:
            img = run_single_sample(pipe, task, source_img, target_prompt, merge_config, cfg_scale, device)
            score = compute_clip_score(img, target_prompt, device=device)
            clip_scores.append(score)
        except Exception as e:
            print(f"    ERROR: {e}")
    return np.mean(clip_scores) if clip_scores else 0, len(clip_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--task', choices=['edit', 'gen'], required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_samples', type=int, default=70)
    parser.add_argument('--eval_samples', type=int, default=100)
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    parser.add_argument('--target_clip', type=float, default=0.3397)
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
    print(f"Loaded. {num_layers} layers. Target: {args.target_clip}")

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
    print(f"Collected {len(samples)} samples")

    all_results = []
    best_clip = 0
    best_config = None

    # PHASE 1: Single merge point sweep
    print(f"\n{'='*60}")
    print(f"PHASE 1: Single merge point K=0..{num_layers-1}")
    print(f"{'='*60}")

    # Standard CFG baseline
    clip, n = evaluate(pipe, args.task, samples, {}, args.cfg_scale, device, args.eval_samples)
    print(f"  Standard CFG (merge at pred): CLIP={clip:.4f} (n={n})")
    all_results.append({'config': 'std_cfg', 'merge_layers': [], 'clip': clip})
    best_clip = clip
    best_config = {'merge_layers': [], 'name': 'std_cfg'}

    for K in range(num_layers):
        merge_config = {K: args.cfg_scale}
        clip, n = evaluate(pipe, args.task, samples, merge_config, args.cfg_scale, device, args.eval_samples)
        marker = " ★ NEW BEST" if clip > best_clip else ""
        if clip > best_clip:
            best_clip = clip
            best_config = {'merge_layers': [K], 'name': f'K={K}'}
        print(f"  K={K:2d}: CLIP={clip:.4f} (n={n}){marker}")
        all_results.append({'config': f'K={K}', 'merge_layers': [K], 'clip': clip})

    print(f"\nPhase 1 best: {best_config['name']} → {best_clip:.4f}")
    print(f"Target: {args.target_clip} | Gap: {best_clip - args.target_clip:+.4f}")

    # PHASE 2: Two merge points
    print(f"\n{'='*60}")
    print(f"PHASE 2: Two merge points")
    print(f"{'='*60}")

    best_K = best_config['merge_layers'][0] if best_config['merge_layers'] else num_layers - 1

    test_pairs = []
    for k1 in range(0, num_layers - 1, 3):
        for k2 in range(k1 + 2, num_layers, 3):
            test_pairs.append((k1, k2))
    for delta in [-6, -3, 3, 6]:
        k2 = best_K + delta
        if 0 <= k2 < num_layers and k2 != best_K:
            test_pairs.append((min(best_K, k2), max(best_K, k2)))

    test_pairs = sorted(set(test_pairs))
    print(f"Testing {len(test_pairs)} pairs")

    for k1, k2 in test_pairs:
        merge_config = {k1: args.cfg_scale, k2: args.cfg_scale}
        clip, n = evaluate(pipe, args.task, samples, merge_config, args.cfg_scale, device, args.eval_samples)
        marker = " ★ NEW BEST" if clip > best_clip else ""
        if clip > best_clip:
            best_clip = clip
            best_config = {'merge_layers': [k1, k2], 'name': f'K={k1},{k2}'}
        print(f"  K={k1},{k2}: CLIP={clip:.4f}{marker}")
        all_results.append({'config': f'K={k1},{k2}', 'merge_layers': [k1, k2], 'clip': clip})

    print(f"\nPhase 2 best: {best_config['name']} → {best_clip:.4f}")

    # PHASE 3: Three merge points
    print(f"\n{'='*60}")
    print(f"PHASE 3: Three merge points")
    print(f"{'='*60}")

    if len(best_config['merge_layers']) >= 2:
        bk1, bk2 = best_config['merge_layers'][0], best_config['merge_layers'][-1]
    else:
        bk1, bk2 = max(0, best_K - 6), best_K

    test_triples = []
    for k_mid in range(bk1 + 1, bk2):
        test_triples.append((bk1, k_mid, bk2))
    for k3 in range(bk2 + 2, min(bk2 + 10, num_layers), 2):
        test_triples.append((bk1, bk2, k3))
    for k0 in range(max(0, bk1 - 8), bk1, 2):
        test_triples.append((k0, bk1, bk2))

    test_triples = sorted(set(test_triples))[:30]
    print(f"Testing {len(test_triples)} triples")

    for ks in test_triples:
        merge_config = {k: args.cfg_scale for k in ks}
        clip, n = evaluate(pipe, args.task, samples, merge_config, args.cfg_scale, device, args.eval_samples)
        ks_str = ','.join(str(k) for k in ks)
        marker = " ★ NEW BEST" if clip > best_clip else ""
        if clip > best_clip:
            best_clip = clip
            best_config = {'merge_layers': list(ks), 'name': f'K={ks_str}'}
        print(f"  K={ks_str}: CLIP={clip:.4f}{marker}")
        all_results.append({'config': f'K={ks_str}', 'merge_layers': list(ks), 'clip': clip})

    print(f"\nPhase 3 best: {best_config['name']} → {best_clip:.4f}")

    # PHASE 4: Scale tuning
    print(f"\n{'='*60}")
    print(f"PHASE 4: Scale tuning at {best_config['name']}")
    print(f"{'='*60}")

    best_merge_layers = best_config['merge_layers']
    current_scales = {k: args.cfg_scale for k in best_merge_layers}
    current_clip = best_clip

    for k in best_merge_layers:
        test_scales = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]
        k_best_scale = current_scales[k]
        k_best_clip = current_clip

        for s in test_scales:
            test = current_scales.copy()
            test[k] = s
            clip, n = evaluate(pipe, args.task, samples, test, args.cfg_scale, device, args.eval_samples)
            if clip > k_best_clip:
                k_best_clip = clip
                k_best_scale = s
            print(f"  Layer {k} s={s:.1f}: CLIP={clip:.4f}")

        if k_best_scale != current_scales[k]:
            current_scales[k] = k_best_scale
            current_clip = k_best_clip
            print(f"  → Layer {k}: updated to s={k_best_scale:.1f}")

    # FINAL VALIDATION
    print(f"\n{'='*60}")
    print(f"FINAL VALIDATION on all {len(samples)} samples")
    print(f"{'='*60}")
    print(f"Config: merge at {best_config['merge_layers']} with scales {current_scales}")
    final_clip, final_n = evaluate(pipe, args.task, samples, current_scales, args.cfg_scale, device)
    print(f"Final CLIP={final_clip:.4f} (n={final_n})")
    print(f"Target: {args.target_clip} | Gap: {final_clip - args.target_clip:+.4f}")
    if final_clip >= args.target_clip:
        print("★★★ TARGET ACHIEVED! ★★★")

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    result = {
        'task': args.task, 'target_clip': args.target_clip,
        'final_clip': final_clip, 'final_n': final_n,
        'best_merge_layers': best_config['merge_layers'],
        'best_scales': {str(k): v for k, v in current_scales.items()},
        'all_results': sorted([{'config': r['config'], 'clip': r['clip']}
                               for r in all_results], key=lambda x: -x['clip'])[:30],
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
