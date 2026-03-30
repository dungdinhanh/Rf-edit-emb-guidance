#!/bin/bash
# Experiment: Embedding-level CFG for video editing
# Run with different alpha values to compare

cd "$(dirname "$0")/.."

ALPHA=${1:-1.0}

echo "=== Running Embedding-level CFG with alpha=${ALPHA} ==="

python3 edit_video_emb_guidance.py   \
        --target_prompt 'A panda wearing a Crown walking in the snow'  \
        --infer-steps 25   \
        --source_prompt 'A panda walking in the snow'  \
        --flow-reverse  \
        --multi-gpu --gpu-ids 0 1   \
        --save-path ./results_emb_guidance/panda_alpha${ALPHA} \
        --source_path './scripts/panda.mp4' \
        --inject_step 3 \
        --embedded-cfg-scale 7 \
        --emb-guidance-alpha ${ALPHA}
