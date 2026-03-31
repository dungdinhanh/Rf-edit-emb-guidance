#!/usr/bin/env python3
"""
Evaluation script for RF-Solver-Edit experiments.

Usage examples:

  # Image editing: compare source vs edited, with CLIP score for target prompt
  python -m evaluation.evaluate image \
      --source path/to/source.png \
      --edited path/to/edited.png \
      --target-prompt "a cat wearing sunglasses"

  # Batch image evaluation: directories of source/edited pairs (matched by filename)
  python -m evaluation.evaluate image \
      --source path/to/source_dir/ \
      --edited path/to/edited_dir/ \
      --target-prompt "a cat wearing sunglasses"

  # FID between two image directories
  python -m evaluation.evaluate fid \
      --real path/to/real_images/ \
      --generated path/to/generated_images/

  # Video editing: frame-level metrics between source and edited video
  python -m evaluation.evaluate video \
      --source path/to/source.mp4 \
      --edited path/to/edited.mp4

  # VBench metrics on edited videos
  python -m evaluation.evaluate vbench \
      --videos path/to/edited_videos/

Run from the RF-Solver-Edit directory:
  cd RF-Solver-Edit && python -m evaluation.evaluate <subcommand> [args]
"""

import argparse
import json
import sys
from pathlib import Path

import torch


def image_eval(args):
    from .metrics import compute_all_image_metrics, compute_clip_score

    device = torch.device(args.device)
    source_path = Path(args.source)
    edited_path = Path(args.edited)

    # Batch mode: directories
    if source_path.is_dir() and edited_path.is_dir():
        src_files = sorted(source_path.glob("*.png")) + sorted(source_path.glob("*.jpg"))
        results_list = []
        for sf in src_files:
            ef = edited_path / sf.name
            if not ef.exists():
                print(f"  skip {sf.name} (no match in edited dir)")
                continue
            r = compute_all_image_metrics(str(sf), str(ef), args.target_prompt, device)
            r["file"] = sf.name
            results_list.append(r)
            print(f"  {sf.name}: PSNR={r['psnr']:.2f} SSIM={r['ssim']:.4f} "
                  f"LPIPS={r['lpips']:.4f} MSE={r['mse']:.6f}", end="")
            if "clip_score" in r:
                print(f" CLIP={r['clip_score']:.4f}", end="")
            print()

        # Averages
        if results_list:
            avg = {}
            keys = [k for k in results_list[0] if k != "file"]
            for k in keys:
                vals = [r[k] for r in results_list if k in r]
                avg[k] = sum(vals) / len(vals)
            print(f"\n  Average ({len(results_list)} pairs):")
            for k, v in avg.items():
                print(f"    {k}: {v:.4f}")
            if args.output:
                _save_json({"per_image": results_list, "average": avg}, args.output)
    else:
        # Single pair
        results = compute_all_image_metrics(
            str(source_path), str(edited_path), args.target_prompt, device
        )
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        if args.output:
            _save_json(results, args.output)


def fid_eval(args):
    from .metrics import compute_fid

    device = torch.device(args.device)
    real_dir = Path(args.real)
    gen_dir = Path(args.generated)

    real_imgs = sorted(real_dir.glob("*.png")) + sorted(real_dir.glob("*.jpg"))
    gen_imgs = sorted(gen_dir.glob("*.png")) + sorted(gen_dir.glob("*.jpg"))

    print(f"  Real images: {len(real_imgs)}")
    print(f"  Generated images: {len(gen_imgs)}")

    fid = compute_fid(
        [str(p) for p in real_imgs],
        [str(p) for p in gen_imgs],
        device=device,
        batch_size=args.batch_size,
    )
    print(f"  FID: {fid:.4f}")
    if args.output:
        _save_json({"fid": fid, "n_real": len(real_imgs), "n_gen": len(gen_imgs)}, args.output)


def video_eval(args):
    from .video_metrics import compute_video_psnr, compute_video_ssim, compute_video_lpips, compute_video_clip_score

    device = torch.device(args.device)
    results = {}

    print("  Computing PSNR...")
    results.update(compute_video_psnr(args.source, args.edited))
    print(f"    PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f}")

    print("  Computing SSIM...")
    results.update(compute_video_ssim(args.source, args.edited))
    print(f"    SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")

    print("  Computing LPIPS...")
    results.update(compute_video_lpips(args.source, args.edited, device=device))
    print(f"    LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")

    if hasattr(args, 'target_prompt') and args.target_prompt:
        print("  Computing CLIP Score...")
        results.update(compute_video_clip_score(args.edited, args.target_prompt, device=device))
        print(f"    CLIP: {results['clip_score_mean']:.4f} ± {results['clip_score_std']:.4f}")

    if args.output:
        _save_json(results, args.output)


def vbench_eval(args):
    from .video_metrics import compute_vbench_metrics

    device = torch.device(args.device)
    dims = args.dimensions.split(",") if args.dimensions else None

    print(f"  Evaluating VBench dimensions: {dims or 'default (SC, MS, AQ, IQ)'}")
    results = compute_vbench_metrics(args.videos, dimensions=dims, device=device)
    for k, v in results.items():
        print(f"    {k}: {v:.4f}")
    if args.output:
        _save_json(results, args.output)


def _save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation metrics for RF-Solver-Edit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", "-o", help="Save results to JSON file")

    sub = parser.add_subparsers(dest="command", required=True)

    # image subcommand
    p_img = sub.add_parser("image", help="Per-image metrics (MSE, PSNR, SSIM, LPIPS, CLIP)")
    p_img.add_argument("--source", required=True, help="Source image or directory")
    p_img.add_argument("--edited", required=True, help="Edited image or directory")
    p_img.add_argument("--target-prompt", default=None, help="Target prompt for CLIP score")

    # fid subcommand
    p_fid = sub.add_parser("fid", help="FID between two image sets")
    p_fid.add_argument("--real", required=True, help="Directory of real images")
    p_fid.add_argument("--generated", required=True, help="Directory of generated images")
    p_fid.add_argument("--batch-size", type=int, default=32)

    # video subcommand
    p_vid = sub.add_parser("video", help="Frame-level video metrics (PSNR, SSIM, LPIPS)")
    p_vid.add_argument("--source", required=True, help="Source video")
    p_vid.add_argument("--edited", required=True, help="Edited video")
    p_vid.add_argument("--target-prompt", default=None, help="Target prompt for CLIP score")

    # vbench subcommand
    p_vb = sub.add_parser("vbench", help="VBench video quality metrics")
    p_vb.add_argument("--videos", required=True, help="Video path, list, or directory")
    p_vb.add_argument("--dimensions", default=None,
                       help="Comma-separated VBench dimensions (default: scene,motion_smoothness,aesthetic_quality,imaging_quality)")

    args = parser.parse_args()
    {"image": image_eval, "fid": fid_eval, "video": video_eval, "vbench": vbench_eval}[args.command](args)


if __name__ == "__main__":
    main()
