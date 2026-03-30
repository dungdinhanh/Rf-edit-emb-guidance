#!/bin/bash
cd "$(dirname "$0")/.."

python3 edit_video.py   \
        --target_prompt 'A panda wearing a Crown walking in the snow'  \
        --infer-steps 25   \
        --source_prompt 'A panda walking in the snow'  \
        --flow-reverse  \
        --multi-gpu --gpu-ids 0 1   \
        --save-path ./panda \
        --source_path './scripts/panda.mp4' \
        --inject_step 3 \
        --embedded-cfg-scale 7
