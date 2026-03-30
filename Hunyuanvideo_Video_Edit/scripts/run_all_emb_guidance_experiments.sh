#!/bin/bash
# Master experiment runner for Embedding-level CFG
# Distributes different alpha values across labs
# Usage: bash scripts/run_all_emb_guidance_experiments.sh <lab> <gpu1> <gpu2> <alpha_values...>
# Example: bash scripts/run_all_emb_guidance_experiments.sh lab1 0 1 0.5 1.0 2.0
# gpu1: transformer GPU, gpu2: VAE+text encoder GPU

LAB=$1
GPU1=$2
GPU2=$3
shift 3
ALPHAS=("$@")

PROJ_DIR="~/sddev/video-editing-research/RF-Solver-Edit/Hunyuanvideo_Video_Edit"
ACTIVATE="source /opt/venvs/rfsolver/bin/activate"

for ALPHA in "${ALPHAS[@]}"; do
    echo "=== Running alpha=${ALPHA} on ${LAB} GPUs ${GPU1},${GPU2} ==="

    ssh ${LAB} "${ACTIVATE} && \
        cd ${PROJ_DIR} && \
        CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python3 edit_video_emb_guidance.py \
            --target_prompt 'A panda wearing a Crown walking in the snow' \
            --infer-steps 25 \
            --source_prompt 'A panda walking in the snow' \
            --flow-reverse \
            --multi-gpu --gpu-ids 0 1 \
            --save-path ./results_emb_guidance/panda_alpha${ALPHA} \
            --source_path './scripts/panda.mp4' \
            --inject_step 3 \
            --embedded-cfg-scale 7 \
            --emb-guidance-alpha ${ALPHA} \
            2>&1 | tee /tmp/rfedit_alpha${ALPHA}.log"

    echo "=== Finished alpha=${ALPHA} on ${LAB} ==="
done
