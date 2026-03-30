#!/bin/bash
cd "$(dirname "$0")/.."

python3 edit_video.py   \
        --target_prompt 'A cat is eating a watermelon'     \
        --infer-steps 25    \
        --source_prompt ""     \
        --flow-reverse   \
        --multi-gpu --gpu-ids 0 1    \
        --save-path ./rabbit \
        --source_path './scripts/rabbit_watermelon.mp4' \
        --inject_step 1 \
        --embedded-cfg-scale 1
