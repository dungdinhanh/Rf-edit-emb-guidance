"""
Search for best per-layer embedding type combination.

Each group of layers can use: "cond", "uncond", or "guided".
Exhaustive search over all 3^num_groups combinations.
"""
import os, sys, json, time, io, argparse, itertools
import torch, numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sd3_edit_3emb import SD3ThreeEmbEditor
from evaluation.metrics import compute_clip_score


def evaluate_config(editor, dataset_dir, layer_modes, max_samples, device="cuda"):
    categories = sorted([d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')])
    clip_scores = []
    for cat in categories:
        pq_path = os.path.join(dataset_dir, cat, "V1-00000-of-00001.parquet")
        if not os.path.exists(pq_path):
            continue
        df = pd.read_parquet(pq_path)
        n = min(len(df), max_samples) if max_samples else len(df)
        for idx in range(n):
            row = df.iloc[idx]
            target_prompt = row['target_prompt'].replace('[', '').replace(']', '')
            img_data = row['image']
            if isinstance(img_data, dict) and 'bytes' in img_data:
                source_img = Image.open(io.BytesIO(img_data['bytes']))
            else:
                continue
            try:
                edited_img, _, = editor.edit(
                    source_img, "", target_prompt,
                    layer_modes=layer_modes,
                )
                score = compute_clip_score(edited_img, target_prompt, device=torch.device(device))
                clip_scores.append(score)
            except Exception as e:
                pass
    return np.mean(clip_scores) if clip_scores else 0.0, len(clip_scores)


def main(args):
    editor = SD3ThreeEmbEditor(model_id=args.model_id)
    num_layers = editor.num_layers

    # Define groups
    num_groups = args.num_groups
    group_size = num_layers // num_groups
    groups = []
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size if g < num_groups - 1 else num_layers
        groups.append(list(range(start, end)))

    modes = ["cond", "uncond", "guided"]
    total_combos = len(modes) ** num_groups

    print(f"Groups ({num_groups}):")
    for i, g in enumerate(groups):
        print(f"  Group {i}: layers {g[0]}-{g[-1]}")
    print(f"Total combinations: {total_combos}")
    print(f"Samples per eval: {args.max_samples} per category")

    results = []
    best_clip = -1
    best_combo = None

    for combo_idx, combo in enumerate(itertools.product(modes, repeat=num_groups)):
        # Build layer_modes
        layer_modes = []
        for g_idx, mode in enumerate(combo):
            for _ in groups[g_idx]:
                layer_modes.append(mode)
        # Pad if needed
        while len(layer_modes) < num_layers:
            layer_modes.append("cond")

        combo_str = "-".join(combo)
        print(f"\n[{combo_idx+1}/{total_combos}] {combo_str}")

        clip, n = evaluate_config(editor, args.dataset_dir, layer_modes,
                                  args.max_samples)
        print(f"  → CLIP={clip:.4f} (n={n})")

        results.append({
            'combo': list(combo),
            'combo_str': combo_str,
            'clip': clip,
            'n': n,
            'layer_modes': layer_modes,
        })

        if clip > best_clip:
            best_clip = clip
            best_combo = combo
            print(f"  ★ New best!")

    # Sort by CLIP
    results.sort(key=lambda x: -x['clip'])

    print(f"\n{'='*60}")
    print(f"TOP 10 COMBINATIONS")
    print(f"{'='*60}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. {r['combo_str']}: CLIP={r['clip']:.4f}")

    print(f"\n{'='*60}")
    print(f"BEST: {'-'.join(best_combo)} → CLIP={best_clip:.4f}")
    print(f"{'='*60}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"3emb_search_{num_groups}groups.json")
    with open(out_path, 'w') as f:
        json.dump({
            'best_combo': list(best_combo),
            'best_clip': best_clip,
            'num_groups': num_groups,
            'groups': [f"{g[0]}-{g[-1]}" for g in groups],
            'all_results': results,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', default='3emb_search_results')
    parser.add_argument('--num_groups', type=int, default=4,
                        help='Number of layer groups. 4 groups = 81 combos, 3 = 27')
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--emb_alpha', type=float, default=0.8)
    args = parser.parse_args()
    main(args)
