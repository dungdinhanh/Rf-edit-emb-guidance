"""PIE-Bench eval for SD3 generation with three-stage normalized embedding guidance."""
import os, json, time, io, argparse
import torch, numpy as np, pandas as pd
from PIL import Image
from sd3_gen_3stage_norm import SD3ThreeStageNormGenerator

def main(args):
    gen = SD3ThreeStageNormGenerator(model_id=args.model_id)

    dataset_dir = args.dataset_dir
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    if args.category:
        categories = [c for c in categories if c == args.category]

    cfg_range = (args.cfg_start, args.cfg_end)
    emb_range = (args.emb_start, args.emb_end)
    emb_early_range = (args.emb_early_start, args.emb_early_end) if args.emb_early_start < args.emb_early_end else None

    method_name = f"gen3sn_cfg{args.cfg_start}-{args.cfg_end}_emb{args.emb_start}-{args.emb_end}_a{args.emb_alpha}"
    if emb_early_range:
        method_name += f"_eemb{args.emb_early_start}-{args.emb_early_end}_ea{args.emb_early_alpha}"
    total_processed = 0

    for cat in categories:
        pq_path = os.path.join(dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path): continue
        df = pd.read_parquet(pq_path)
        n = min(len(df), args.max_samples) if args.max_samples else len(df)
        print(f"\n=== {cat} ({n}/{len(df)}) | {method_name} ===")

        out_dir = os.path.join(args.output_dir, method_name, cat)
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(n):
            row = df.iloc[idx]
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            sample_id = str(row.get('id', idx))
            out_path = os.path.join(out_dir, f"{sample_id}.jpg")
            if os.path.exists(out_path):
                total_processed += 1; continue

            try:
                img, elapsed, info = gen.generate(
                    target_prompt, num_steps=args.num_steps,
                    cfg_scale=args.cfg_scale, cfg_range=cfg_range,
                    emb_range=emb_range, emb_alpha=args.emb_alpha,
                    emb_early_range=emb_early_range,
                    emb_early_alpha=args.emb_early_alpha,
                    seed=42,
                )
                img.save(out_path, quality=95)
                json.dump({'target_prompt': target_prompt, 'category': cat,
                           'sample_id': sample_id, 'mode': 'gen3sn',
                           'cfg_range': list(cfg_range), 'emb_range': list(emb_range),
                           'emb_early_range': list(emb_early_range) if emb_early_range else None,
                           'emb_alpha': args.emb_alpha, 'emb_early_alpha': args.emb_early_alpha,
                           'elapsed': elapsed, **info},
                          open(os.path.join(out_dir, f"{sample_id}_meta.json"), 'w'), indent=2)
                total_processed += 1
                print(f"  [{idx+1}/{n}] {sample_id}: {elapsed:.1f}s")
            except Exception as e:
                print(f"  [{idx+1}/{n}] ERROR: {e}")

    print(f"\n=== Done! Processed: {total_processed} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--emb_alpha', type=float, default=0.5)
    parser.add_argument('--emb_early_alpha', type=float, default=None)
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    parser.add_argument('--cfg_start', type=float, default=0.3)
    parser.add_argument('--cfg_end', type=float, default=0.8)
    parser.add_argument('--emb_start', type=float, default=0.8)
    parser.add_argument('--emb_end', type=float, default=1.0)
    parser.add_argument('--emb_early_start', type=float, default=0.0)
    parser.add_argument('--emb_early_end', type=float, default=0.0)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--category', default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    main(args)
