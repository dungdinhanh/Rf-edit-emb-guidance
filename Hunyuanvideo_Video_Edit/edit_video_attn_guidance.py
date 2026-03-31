"""
Experiment: Attention-level Guidance for Video Editing (v2)

Instead of blending embeddings before the transformer (v1),
this applies guidance inside each attention layer by:
- Scoring text tokens for both cond and uncond
- Identifying top-k overlapping tokens
- Zeroing overlap to amplify editing signal
"""

import os
import sys
import time
import argparse
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference_attn_guidance import HunyuanVideoEditAttnGuidance


def main():
    # Extract custom args before parse_args
    attn_guidance_alpha = 1.0
    attn_guidance_k = 32
    overlap_target = "uncond"
    filtered_argv = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--attn-guidance-alpha":
            attn_guidance_alpha = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--attn-guidance-k":
            attn_guidance_k = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--overlap-target":
            overlap_target = sys.argv[i + 1]
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1
    sys.argv = [sys.argv[0]] + filtered_argv

    args = parse_args()

    print(f"=== Attention-level Guidance Experiment ===")
    print(f"Alpha: {attn_guidance_alpha}, K: {attn_guidance_k}, Overlap target: {overlap_target}")
    print(f"Source prompt: {args.source_prompt}")
    print(f"Target prompt: {args.target_prompt}")
    print(args)

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_edit = HunyuanVideoEditAttnGuidance.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_edit.args

    # Start sampling
    outputs = hunyuan_video_edit.predict(
        source_video_path=args.source_path,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        inject_step=args.inject_step,
        attn_guidance_alpha=attn_guidance_alpha,
        attn_guidance_k=attn_guidance_k,
        overlap_target=overlap_target,
    )
    samples = outputs['samples']

    for i, sample in enumerate(samples):
        sample = samples[i].unsqueeze(0)
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        save_file = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_k{attn_guidance_k}_alpha{attn_guidance_alpha}_{outputs['prompts'][i][:100].replace('/', '')}.mp4"
        save_videos_grid(sample, save_file, fps=24)
        logger.info(f'Sample save to: {save_file}')


if __name__ == "__main__":
    main()
