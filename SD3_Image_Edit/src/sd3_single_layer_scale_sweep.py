"""
For each layer, sweep CFG scale to find the optimal single-layer guidance.
Tests scale = 1.1, 1.2, 1.3, 1.5, 2.0, 3.0 per layer.
"""

import os, sys, io, json, time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.metrics import compute_clip_score
from sd3_single_layer_analysis import SingleLayerCFGRunner


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--task', choices=['edit', 'gen'], default='edit')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_samples', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    runner = SingleLayerCFGRunner(model_id=args.model_id, task=args.task)
    num_layers = runner.num_layers

    categories = sorted([d for d in os.listdir(args.dataset_dir)
                        if os.path.isdir(os.path.join(args.dataset_dir, d)) and not d.startswith('.')])

    scales = [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]

    # Collect samples
    samples = []
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
            samples.append((source_img, target_prompt, f"{cat}/{row.get('id', idx)}"))

    print(f"Collected {len(samples)} samples")

    # For each layer × scale, compute CLIP via full denoise
    # results[layer][scale] = [clip_scores]
    results = {li: {s: [] for s in scales} for li in range(num_layers)}

    for si, (source_img, target_prompt, sample_id) in enumerate(samples):
        print(f"\n[{si+1}/{len(samples)}] {sample_id}")
        for li in range(num_layers):
            for s in scales:
                img = runner.full_denoise_single_layer(
                    source_image=source_img, target_prompt=target_prompt,
                    cfg_scale=s, cfg_layer_idx=li)
                clip = compute_clip_score(img, target_prompt, device=device)
                results[li][s].append(clip)
            # Print progress for this layer
            best_s = max(scales, key=lambda s: np.mean(results[li][s]))
            best_clip = np.mean(results[li][best_s])
            cond_clip = np.mean(results[li][1.0])
            print(f"  Layer {li:2d}: best scale={best_s:.2f} CLIP={best_clip:.4f} (cond={cond_clip:.4f}, delta={best_clip-cond_clip:+.4f})")

    # Write report
    with open(args.output, 'w') as f:
        f.write(f"# Single-Layer CFG Scale Sweep ({args.task})\n\n")
        f.write(f"{len(samples)} samples. Scales tested: {scales}\n\n")

        # Table: best scale per layer
        f.write("## 1. Best Scale per Layer\n\n")
        f.write("| Layer | Best Scale | CLIP at Best | CLIP at s=1.0 | Delta | Helpful? |\n")
        f.write("|:-----:|:----------:|:------------:|:-------------:|:-----:|:--------:|\n")

        cond_baseline = np.mean(results[0][1.0])  # s=1.0 at any layer = cond only
        for li in range(num_layers):
            cond_clip = np.mean(results[li][1.0])
            best_s = max(scales, key=lambda s: np.mean(results[li][s]))
            best_clip = np.mean(results[li][best_s])
            delta = best_clip - cond_clip
            helpful = "Yes" if delta > 0.001 else "No" if delta < -0.001 else "Neutral"
            f.write(f"| {li:2d} | {best_s:.2f} | {best_clip:.4f} | {cond_clip:.4f} | {delta:+.4f} | {helpful} |\n")

        # Full table
        f.write("\n## 2. Full CLIP Table (Layer × Scale)\n\n")
        header = "| Layer |" + "|".join([f" s={s} " for s in scales]) + "|\n"
        sep = "|:-----:|" + "|".join([":-----:" for _ in scales]) + "|\n"
        f.write(header)
        f.write(sep)
        for li in range(num_layers):
            row = f"| {li:2d} |"
            clips = [np.mean(results[li][s]) for s in scales]
            best_idx = np.argmax(clips)
            for idx, c in enumerate(clips):
                if idx == best_idx:
                    row += f" **{c:.4f}** |"
                else:
                    row += f" {c:.4f} |"
            f.write(row + "\n")

    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
