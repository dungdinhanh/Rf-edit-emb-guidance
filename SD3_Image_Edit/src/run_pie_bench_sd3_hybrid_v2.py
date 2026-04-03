"""PIE-Bench evaluation using SD3 hybrid guidance v2 (single trajectory)."""
import os, sys, json, time, io, argparse
import torch, numpy as np, pandas as pd
from PIL import Image
from sd3_edit_hybrid_v2 import SD3HybridEditorV2

def main(args):
    editor = SD3HybridEditorV2(model_id=args.model_id)

    dataset_dir = args.dataset_dir
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    if args.category:
        categories = [c for c in categories if c == args.category]

    method_name = f"hybridv2_a{args.emb_alpha}_cfg{args.cfg_scale}_k{args.cfg_steps}"
    total_processed, total_errors = 0, 0

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
            source_prompt = row['source_prompt']
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            sample_id = str(row.get('id', idx))
            out_path = os.path.join(out_dir, f"{sample_id}.jpg")
            if os.path.exists(out_path):
                total_processed += 1; continue

            img_data = row['image']
            if isinstance(img_data, dict) and 'bytes' in img_data:
                source_img = Image.open(io.BytesIO(img_data['bytes']))
            else: continue

            try:
                edited_img, elapsed = editor.edit(
                    source_img, source_prompt, target_prompt,
                    num_steps=args.num_steps, strength=args.strength,
                    emb_alpha=args.emb_alpha, cfg_scale=args.cfg_scale,
                    cfg_steps=args.cfg_steps,
                )
                edited_img.save(out_path, quality=95)
                source_img.save(os.path.join(out_dir, f"{sample_id}_source.jpg"), quality=95)
                json.dump({'source_prompt': source_prompt, 'target_prompt': target_prompt,
                           'category': cat, 'sample_id': sample_id,
                           'mode': 'hybrid_v2', 'emb_alpha': args.emb_alpha,
                           'cfg_scale': args.cfg_scale, 'cfg_steps': args.cfg_steps,
                           'strength': args.strength, 'elapsed': elapsed},
                          open(os.path.join(out_dir, f"{sample_id}_meta.json"), 'w'), indent=2)
                total_processed += 1
                print(f"  [{idx+1}/{n}] {sample_id}: {elapsed:.1f}s")
            except Exception as e:
                total_errors += 1
                print(f"  [{idx+1}/{n}] ERROR: {e}")
                import traceback; traceback.print_exc()

    print(f"\n=== Done! Processed: {total_processed}, Errors: {total_errors} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--strength', type=float, default=0.7)
    parser.add_argument('--emb_alpha', type=float, default=0.3)
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    parser.add_argument('--cfg_steps', type=int, default=3)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--category', default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    main(args)
