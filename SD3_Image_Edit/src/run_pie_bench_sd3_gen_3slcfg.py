"""PIE-Bench eval for SD3 generation with 3-stage layercfg."""
import os, json, time, io, argparse
import torch, numpy as np, pandas as pd
from PIL import Image
from sd3_gen_3stage_layercfg import SD3GenThreeStageLayerCFG

def main(args):
    gen = SD3GenThreeStageLayerCFG(model_id=args.model_id)

    dataset_dir = args.dataset_dir
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    if args.category:
        categories = [c for c in categories if c == args.category]

    method_name = f"gen3slcfg_cfg{args.cfg_start}-{args.cfg_end}_lcfg{args.lcfg_start}-{args.lcfg_end}_{args.lcfg_schedule}_s{args.lcfg_scale}"
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
                    cfg_scale=args.cfg_scale, cfg_range=(args.cfg_start, args.cfg_end),
                    layercfg_range=(args.lcfg_start, args.lcfg_end),
                    layercfg_scale=args.lcfg_scale, layercfg_schedule=args.lcfg_schedule,
                    seed=42)
                img.save(out_path, quality=95)
                json.dump({'target_prompt': target_prompt, 'category': cat,
                           'sample_id': sample_id, 'elapsed': elapsed, **info},
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
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    parser.add_argument('--cfg_start', type=float, default=0.3)
    parser.add_argument('--cfg_end', type=float, default=0.8)
    parser.add_argument('--lcfg_scale', type=float, default=6.0)
    parser.add_argument('--lcfg_schedule', default='peak_mid', choices=['uniform', 'linear_up', 'peak_mid'])
    parser.add_argument('--lcfg_start', type=float, default=0.8)
    parser.add_argument('--lcfg_end', type=float, default=1.0)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--category', default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    main(args)
