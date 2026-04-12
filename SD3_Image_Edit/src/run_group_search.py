"""
Grouped per-layer alpha search for SD3 embedding guidance.

Splits 24 layers into groups and greedily searches each group's alpha.
Strategy: fix all groups, sweep one at a time, pick best, move to next.
"""
import os, sys, json, time, io, argparse
import torch, numpy as np, pandas as pd
from PIL import Image

# Add RF-Solver-Edit root to path for evaluation module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sd3_edit_perlayer import SD3PerLayerEditor
from evaluation.metrics import compute_clip_score


def evaluate_config(editor, dataset_dir, custom_alphas, max_samples=None, device="cuda"):
    """Run editing with custom alphas and return mean CLIP."""
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
                edited_img, _, _ = editor.edit(
                    source_img, "", target_prompt,
                    custom_alphas=custom_alphas,
                )
                score = compute_clip_score(edited_img, target_prompt, device=torch.device(device))
                clip_scores.append(score)
            except Exception as e:
                print(f"  ERROR: {e}")

    return np.mean(clip_scores) if clip_scores else 0.0, len(clip_scores)


def main(args):
    editor = SD3PerLayerEditor(model_id=args.model_id)
    num_layers = editor.num_layers  # 24

    # Define groups
    num_groups = args.num_groups
    group_size = num_layers // num_groups
    groups = []
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size if g < num_groups - 1 else num_layers
        groups.append(list(range(start, end)))

    print(f"Groups ({num_groups}):")
    for i, g in enumerate(groups):
        print(f"  Group {i}: layers {g[0]}-{g[-1]}")

    # Alpha candidates to search
    alpha_candidates = [float(x) for x in args.alpha_candidates.split(',')]
    print(f"Alpha candidates: {alpha_candidates}")

    # Greedy sequential search
    best_alphas = [0.0] * num_layers  # start with no guidance
    results_log = []

    for g_idx, group_layers in enumerate(groups):
        print(f"\n{'='*60}")
        print(f"Searching Group {g_idx}: layers {group_layers[0]}-{group_layers[-1]}")
        print(f"Current best alphas: {[f'{a:.2f}' for a in best_alphas]}")
        print(f"{'='*60}")

        best_clip = -1
        best_alpha = 0.0

        for alpha in alpha_candidates:
            # Set this group's alpha
            test_alphas = best_alphas.copy()
            for l in group_layers:
                test_alphas[l] = alpha

            print(f"\n  Testing α={alpha} for layers {group_layers[0]}-{group_layers[-1]}...")
            clip, n = evaluate_config(editor, args.dataset_dir, test_alphas,
                                      max_samples=args.max_samples)
            print(f"  → CLIP={clip:.4f} (n={n})")

            results_log.append({
                'group': g_idx,
                'layers': f"{group_layers[0]}-{group_layers[-1]}",
                'alpha': alpha,
                'clip': clip,
                'n': n,
                'full_alphas': test_alphas.copy(),
            })

            if clip > best_clip:
                best_clip = clip
                best_alpha = alpha

        # Lock this group's best alpha
        for l in group_layers:
            best_alphas[l] = best_alpha
        print(f"\n  ★ Best for Group {g_idx}: α={best_alpha} → CLIP={best_clip:.4f}")

    print(f"\n{'='*60}")
    print(f"FINAL RESULT")
    print(f"{'='*60}")
    print(f"Best per-layer alphas: {[f'{a:.2f}' for a in best_alphas]}")

    # Final evaluation on full dataset
    print(f"\nFinal evaluation...")
    final_clip, final_n = evaluate_config(editor, args.dataset_dir, best_alphas,
                                           max_samples=None)
    print(f"Final CLIP={final_clip:.4f} (n={final_n})")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'best_alphas': best_alphas,
        'final_clip': final_clip,
        'final_n': final_n,
        'num_groups': num_groups,
        'groups': [{'layers': f"{g[0]}-{g[-1]}", 'alpha': best_alphas[g[0]]} for g in groups],
        'search_log': results_log,
    }
    out_path = os.path.join(args.output_dir, f"group_search_{num_groups}groups.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--output_dir', default='group_search_results')
    parser.add_argument('--num_groups', type=int, default=4)
    parser.add_argument('--alpha_candidates', type=str, default='0.0,0.3,0.5,0.7,1.0,1.5')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='Samples per category for search (use small for speed)')
    args = parser.parse_args()
    main(args)
