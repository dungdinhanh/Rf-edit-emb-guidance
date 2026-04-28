"""Run LayerAttenCFG or LayerEmbeddCFG on PIE-Bench."""
import os, sys, json, time, io, argparse
import torch, numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.metrics import compute_clip_score


def main(args):
    if args.method == "atten":
        from sd3_edit_layer_atten_cfg import SD3LayerAttenCFGEditor
        editor = SD3LayerAttenCFGEditor(model_id=args.model_id, task=args.task)
    else:
        from sd3_edit_layer_embedd_cfg import SD3LayerEmbeddCFGEditor
        editor = SD3LayerEmbeddCFGEditor(model_id=args.model_id, task=args.task)

    device = torch.device("cuda")
    dataset_dir = args.dataset_dir
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    if args.category:
        categories = [c for c in categories if c == args.category]

    method_name = f"layer{args.method}cfg_{args.task}_{args.layer_schedule}_s{args.base_scale}"
    total_processed, total_errors = 0, 0
    clip_scores = []

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

            source_img = None
            if args.task == "edit":
                img_data = row['image']
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    source_img = Image.open(io.BytesIO(img_data['bytes']))
                else: continue

            try:
                edited_img, elapsed = editor.run(
                    source_image=source_img, target_prompt=target_prompt,
                    base_scale=args.base_scale, layer_schedule=args.layer_schedule)
                edited_img.save(out_path, quality=95)
                if source_img:
                    source_img.save(os.path.join(out_dir, f"{sample_id}_source.jpg"), quality=95)
                score = compute_clip_score(edited_img, target_prompt, device=device)
                clip_scores.append(score)
                json.dump({'target_prompt': target_prompt, 'sample_id': sample_id,
                           'method': f"layer{args.method}cfg", 'task': args.task,
                           'base_scale': args.base_scale, 'layer_schedule': args.layer_schedule,
                           'elapsed': elapsed, 'clip': score},
                          open(os.path.join(out_dir, f"{sample_id}_meta.json"), 'w'), indent=2)
                total_processed += 1
                print(f"  [{idx+1}/{n}] {sample_id}: CLIP={score:.4f} ({elapsed:.1f}s)")
            except Exception as e:
                total_errors += 1
                print(f"  [{idx+1}/{n}] ERROR: {e}")
                import traceback; traceback.print_exc()

    mean_clip = np.mean(clip_scores) if clip_scores else 0
    print(f"\n=== {method_name}: CLIP={mean_clip:.4f} (n={len(clip_scores)}) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--method', choices=['atten', 'embedd'], required=True)
    parser.add_argument('--task', choices=['edit', 'gen'], required=True)
    parser.add_argument('--base_scale', type=float, default=5.0)
    parser.add_argument('--layer_schedule', default='peak_mid',
                        choices=['uniform', 'linear_up', 'peak_mid'])
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--category', default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    main(args)
