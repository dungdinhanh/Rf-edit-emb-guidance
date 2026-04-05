"""
Interval guidance experiment: Compare CFG placement for generation vs editing.

Hypothesis: CFG in middle works best for generation, CFG from start works best for editing.
Uses same prompts from PIE-Bench for fair comparison.
"""
import os, json, time, io, argparse
import torch, numpy as np, pandas as pd
from PIL import Image


def run_generation(args):
    """Text-to-image generation with different CFG placements."""
    from sd3_gen_3stage import SD3ThreeStageGenerator
    gen = SD3ThreeStageGenerator(model_id=args.model_id, offload=args.offload)

    dataset_dir = args.dataset_dir
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    if args.category:
        categories = [c for c in categories if c == args.category]

    cfg_range = (args.cfg_start, args.cfg_end)
    emb_range = (args.emb_start, args.emb_end)
    method_name = f"gen_cfg{args.cfg_start}-{args.cfg_end}_emb{args.emb_start}-{args.emb_end}"
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
                    emb_range=emb_range, emb_alpha=args.emb_alpha, seed=42,
                )
                img.save(out_path, quality=95)
                json.dump({'target_prompt': target_prompt, 'category': cat,
                           'sample_id': sample_id, 'mode': 'generation',
                           'cfg_range': list(cfg_range), 'emb_range': list(emb_range),
                           'elapsed': elapsed, **info},
                          open(os.path.join(out_dir, f"{sample_id}_meta.json"), 'w'), indent=2)
                total_processed += 1
                print(f"  [{idx+1}/{n}] {sample_id}: {elapsed:.1f}s")
            except Exception as e:
                print(f"  [{idx+1}/{n}] ERROR: {e}")

    print(f"\n=== Done! Processed: {total_processed} ===")


def run_editing(args):
    """Image editing with different CFG placements."""
    from sd3_edit_3stage import SD3ThreeStageEditor
    editor = SD3ThreeStageEditor(model_id=args.model_id, offload=args.offload)

    dataset_dir = args.dataset_dir
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    if args.category:
        categories = [c for c in categories if c == args.category]

    cfg_range = (args.cfg_start, args.cfg_end)
    emb_range = (args.emb_start, args.emb_end)
    method_name = f"edit_cfg{args.cfg_start}-{args.cfg_end}_emb{args.emb_start}-{args.emb_end}"
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
                edited_img, elapsed, info = editor.edit(
                    source_img, source_prompt, target_prompt,
                    num_steps=args.num_steps, strength=args.strength,
                    cfg_scale=args.cfg_scale, cfg_range=cfg_range,
                    emb_range=emb_range, emb_alpha=args.emb_alpha,
                )
                edited_img.save(out_path, quality=95)
                source_img.save(os.path.join(out_dir, f"{sample_id}_source.jpg"), quality=95)
                json.dump({'source_prompt': source_prompt, 'target_prompt': target_prompt,
                           'category': cat, 'sample_id': sample_id, 'mode': 'editing',
                           'cfg_range': list(cfg_range), 'emb_range': list(emb_range),
                           'strength': args.strength, 'elapsed': elapsed, **info},
                          open(os.path.join(out_dir, f"{sample_id}_meta.json"), 'w'), indent=2)
                total_processed += 1
                print(f"  [{idx+1}/{n}] {sample_id}: {elapsed:.1f}s")
            except Exception as e:
                print(f"  [{idx+1}/{n}] ERROR: {e}")

    print(f"\n=== Done! Processed: {total_processed} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['gen', 'edit'], required=True)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--strength', type=float, default=0.7)
    parser.add_argument('--emb_alpha', type=float, default=0.3)
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    parser.add_argument('--cfg_start', type=float, default=0.0)
    parser.add_argument('--cfg_end', type=float, default=1.0)
    parser.add_argument('--emb_start', type=float, default=1.0)
    parser.add_argument('--emb_end', type=float, default=1.0)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--category', default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--offload', action='store_true')
    args = parser.parse_args()

    if args.task == 'gen':
        run_generation(args)
    else:
        run_editing(args)
