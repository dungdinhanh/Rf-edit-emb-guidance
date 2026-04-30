"""
Run pure full-block LayerCFG with custom per-layer scales.
Tests optimal scales from single-layer sweep with various multipliers.
"""
import os, sys, io, json, argparse
import torch, numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sd3_edit_pure_layercfg import SD3PureLayerCFGEditor
from evaluation.metrics import compute_clip_score

# Optimal single-layer scales from the sweep
EDIT_OPTIMAL = [1.05, 1.30, 1.00, 1.05, 1.10, 1.05, 1.05, 1.10, 1.20, 1.00,
                1.05, 1.10, 1.10, 1.05, 1.20, 1.05, 1.00, 1.10, 1.20, 1.50,
                1.00, 1.20, 1.20, 1.20]

GEN_OPTIMAL = [1.00, 1.50, 1.10, 1.00, 1.00, 1.00, 1.00, 1.05, 1.05, 1.00,
               1.00, 1.05, 1.00, 1.20, 1.05, 1.05, 1.05, 1.00, 1.05, 1.20,
               1.10, 1.05, 1.30, 1.00]


def scale_custom(base_scales, multiplier):
    """Apply multiplier to the guidance portion: new_s = 1 + multiplier * (s - 1)"""
    return [1.0 + multiplier * (s - 1.0) for s in base_scales]


def run_eval(editor, dataset_dir, custom_scales, task, max_samples, device):
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    clip_scores = []
    for cat in categories:
        pq_path = os.path.join(dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path): continue
        df = pd.read_parquet(pq_path)
        n = min(len(df), max_samples) if max_samples else len(df)
        for idx in range(n):
            row = df.iloc[idx]
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            source_img = None
            if task == "edit":
                img_data = row['image']
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    source_img = Image.open(io.BytesIO(img_data['bytes']))
                else: continue
            try:
                edited_img, _ = editor.run(
                    source_image=source_img, target_prompt=target_prompt,
                    base_scale=1.0, layer_schedule="uniform",
                    t_schedule="constant")
                # Override: use custom scales directly
                # We need to modify the run method... let's use a workaround
                score = compute_clip_score(edited_img, target_prompt, device=torch.device(device))
                clip_scores.append(score)
            except Exception as e:
                pass
    return np.mean(clip_scores) if clip_scores else 0, len(clip_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--task', choices=['edit', 'gen'], required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    editor = SD3PureLayerCFGEditor(model_id=args.model_id, task=args.task)

    base_scales = EDIT_OPTIMAL if args.task == "edit" else GEN_OPTIMAL

    # Test different multipliers
    multipliers = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    results = []
    for mult in multipliers:
        custom = scale_custom(base_scales, mult)
        # Format for printing
        print(f"\n=== Multiplier={mult} ===")
        print(f"  Scales: [{', '.join(f'{s:.2f}' for s in custom)}]")

        # Run with custom scales via the editor
        # Need to pass custom_scales... let's use the run method directly
        categories = sorted([d for d in os.listdir(args.dataset_dir)
                            if os.path.isdir(os.path.join(args.dataset_dir, d)) and not d.startswith('.')])
        clip_scores = []

        for cat in categories:
            pq_path = os.path.join(args.dataset_dir, cat, "V1-00000-of-00001.parquet")
            if not os.path.exists(pq_path): continue
            df = pd.read_parquet(pq_path)
            n = min(len(df), args.max_samples) if args.max_samples else len(df)

            out_dir = os.path.join(args.output_dir, f"custom_{args.task}_m{mult}", cat)
            os.makedirs(out_dir, exist_ok=True)

            for idx in range(n):
                row = df.iloc[idx]
                target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
                sample_id = str(row.get('id', idx))
                out_path = os.path.join(out_dir, f"{sample_id}.jpg")
                if os.path.exists(out_path):
                    img = Image.open(out_path)
                    score = compute_clip_score(img, target_prompt, device=device)
                    clip_scores.append(score)
                    continue

                source_img = None
                if args.task == "edit":
                    img_data = row['image']
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        source_img = Image.open(io.BytesIO(img_data['bytes']))
                    else: continue

                try:
                    # Use the internal method with custom scales
                    t0 = __import__('time').time()
                    dev = editor.pipe._execution_device

                    (cond_e, neg_e, cond_p, neg_p) = editor.pipe.encode_prompt(
                        prompt=target_prompt, prompt_2=target_prompt, prompt_3=target_prompt,
                        negative_prompt="", negative_prompt_2="", negative_prompt_3="",
                        do_classifier_free_guidance=True, device=dev)

                    if args.task == "edit":
                        w, h = source_img.size
                        w, h = w - w % 16, h - h % 16
                        source_img = source_img.resize((w, h))
                        image_tensor = editor.pipe.image_processor.preprocess(source_img)
                        image_tensor = image_tensor.to(device=dev, dtype=editor.pipe.vae.dtype)
                        latents = editor.pipe.vae.encode(image_tensor).latent_dist.sample()
                        latents = (latents - editor.pipe.vae.config.shift_factor) * editor.pipe.vae.config.scaling_factor
                        latents = latents.to(dtype=cond_e.dtype)
                        editor.pipe.scheduler.set_timesteps(25, device=dev)
                        ts_list, _ = editor.pipe.get_timesteps(25, 0.7, dev)
                        noise = torch.randn_like(latents)
                        latents = editor.pipe.scheduler.scale_noise(latents, ts_list[:1], noise)
                    else:
                        generator = torch.Generator(device=dev).manual_seed(42)
                        nc = editor.pipe.transformer.config.in_channels
                        latents = torch.randn(1, nc, 512 // editor.pipe.vae_scale_factor,
                            512 // editor.pipe.vae_scale_factor, generator=generator,
                            device=dev, dtype=torch.bfloat16)
                        editor.pipe.scheduler.set_timesteps(25, device=dev)
                        ts_list = editor.pipe.scheduler.timesteps

                    for i, t in enumerate(ts_list):
                        timestep = t.expand(latents.shape[0])
                        noise_pred = editor._run_layercfg_step(
                            latents, timestep, cond_e, neg_e, cond_p, custom)
                        latents_dtype = latents.dtype
                        latents = editor.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                        if latents.dtype != latents_dtype:
                            latents = latents.to(latents_dtype)

                    latents = latents / editor.pipe.vae.config.scaling_factor + editor.pipe.vae.config.shift_factor
                    image = editor.pipe.vae.decode(latents, return_dict=False)[0]
                    edited_img = editor.pipe.image_processor.postprocess(image, output_type="pil")[0]

                    edited_img.save(out_path, quality=95)
                    score = compute_clip_score(edited_img, target_prompt, device=device)
                    clip_scores.append(score)
                    print(f"  [{len(clip_scores)}] {sample_id}: CLIP={score:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

        mean_clip = np.mean(clip_scores) if clip_scores else 0
        results.append({'multiplier': mult, 'clip': mean_clip, 'n': len(clip_scores)})
        print(f"  RESULT: mult={mult} → CLIP={mean_clip:.4f} (n={len(clip_scores)})")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — Custom LayerCFG ({args.task})")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: -x['clip']):
        print(f"  mult={r['multiplier']:.1f}: CLIP={r['clip']:.4f} (n={r['n']})")


if __name__ == "__main__":
    main()
