"""
Experiment: Embedding-level Classifier-Free Guidance for Video Editing

Instead of standard CFG on noise predictions:
    noise = uncond + scale * (cond - uncond)

We apply CFG directly on the prompt embeddings:
    guided_emb = (1 + alpha) * f(prompt) - alpha * f(empty)

Then use a single forward pass with the guided embedding.

Benefits:
- Only 1 forward pass per step instead of 2 (2x speedup in denoising)
- Guidance applied in the embedding space, which may produce different characteristics
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
from hyvideo.inference_emb_guidance import HunyuanVideoEditEmbGuidance


def main():
    # Extract --emb-guidance-alpha before parse_args sees it (since it's not in the original config)
    emb_guidance_alpha = 1.0
    filtered_argv = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--emb-guidance-alpha":
            emb_guidance_alpha = float(sys.argv[i + 1])
            i += 2
        else:
            filtered_argv.append(sys.argv[i])
            i += 1
    sys.argv = [sys.argv[0]] + filtered_argv

    args = parse_args()

    print(f"=== Embedding-level CFG Experiment ===")
    print(f"Alpha: {emb_guidance_alpha}")
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
    hunyuan_video_edit = HunyuanVideoEditEmbGuidance.from_pretrained(models_root_path, args=args)
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
        emb_guidance_alpha=emb_guidance_alpha,
    )
    samples = outputs['samples']

    for i, sample in enumerate(samples):
        sample = samples[i].unsqueeze(0)
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        save_file = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_alpha{emb_guidance_alpha}_{outputs['prompts'][i][:100].replace('/', '')}.mp4"
        save_videos_grid(sample, save_file, fps=24)
        logger.info(f'Sample save to: {save_file}')


if __name__ == "__main__":
    main()
