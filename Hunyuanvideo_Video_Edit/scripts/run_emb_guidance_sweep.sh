#!/bin/bash
# Sweep embedded-cfg-scale values for embedding guidance experiment
# Runs inside Docker container with rfsolver venv
# Usage: bash scripts/run_emb_guidance_sweep.sh [gpu1] [gpu2] [output_dir]

GPU1=${1:-0}
GPU2=${2:-1}
OUTPUT_BASE=${3:-/workspace/RF-Solver-Edit/Hunyuanvideo_Video_Edit/sampled_videos}

cd "$(dirname "$0")/.."

SCALES=(3 5 7 9)
ALPHA=1.0

for SCALE in "${SCALES[@]}"; do
    echo "=== Running embedded-cfg-scale=${SCALE}, alpha=${ALPHA} ==="

    CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python3 edit_video_emb_guidance.py \
        --target_prompt 'A panda wearing a Crown walking in the snow' \
        --infer-steps 25 \
        --source_prompt 'A panda walking in the snow' \
        --flow-reverse \
        --multi-gpu --gpu-ids 0 1 \
        --save-path "${OUTPUT_BASE}/emb_guidance_scale${SCALE}_alpha${ALPHA}" \
        --source_path './scripts/panda.mp4' \
        --inject_step 3 \
        --embedded-cfg-scale ${SCALE} \
        --emb-guidance-alpha ${ALPHA}

    echo "=== Finished scale=${SCALE} ==="
    echo ""
done

echo "All runs complete. Results in ${OUTPUT_BASE}/"
