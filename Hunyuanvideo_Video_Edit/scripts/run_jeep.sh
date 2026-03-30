#!/bin/bash
cd "$(dirname "$0")/.."

python3 edit_video.py  \
    --target_prompt 'A pink porsche'    \
    --infer-steps 25    \
    --source_prompt ''  \
    --flow-reverse   \
    --multi-gpu --gpu-ids 0 1   \
    --save-path ./jeep  \
    --source_path './scripts/jeep.mp4' \
    --inject_step 2 \
    --embedded-cfg-scale 6
