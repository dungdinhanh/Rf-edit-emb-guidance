#!/bin/bash
# Distributed experiment launcher for Embedding-level CFG
# Runs across lab1, lab2, lab3 with different alpha values
# Also runs baseline (original CFG) for comparison

set -e

PROJ="sddev/video-editing-research/RF-Solver-Edit/Hunyuanvideo_Video_Edit"
ACTIVATE="source /opt/venvs/rfsolver/bin/activate"

# Common args
COMMON_ARGS="--infer-steps 25 --flow-reverse --multi-gpu --gpu-ids 0 1 --inject_step 3 --embedded-cfg-scale 7"
SOURCE_PROMPT="A panda walking in the snow"
TARGET_PROMPT="A panda wearing a Crown walking in the snow"
SOURCE_PATH="./scripts/panda.mp4"

run_emb_guidance() {
    local LAB=$1
    local GPU1=$2
    local GPU2=$3
    local ALPHA=$4
    local SAVE="./results_emb_guidance/panda_alpha${ALPHA}"

    echo "[$(date)] Starting emb_guidance alpha=${ALPHA} on ${LAB} GPUs ${GPU1},${GPU2}"
    ssh ${LAB} "nohup bash -c '${ACTIVATE} && cd ~/${PROJ} && \
        CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python3 edit_video_emb_guidance.py \
            --target_prompt \"${TARGET_PROMPT}\" \
            --source_prompt \"${SOURCE_PROMPT}\" \
            --source_path \"${SOURCE_PATH}\" \
            --save-path \"${SAVE}\" \
            --emb-guidance-alpha ${ALPHA} \
            ${COMMON_ARGS}' > /tmp/rfedit_emb_alpha${ALPHA}.log 2>&1 &
    echo \$!"
}

run_baseline() {
    local LAB=$1
    local GPU1=$2
    local GPU2=$3
    local CFG=$4
    local SAVE="./results_baseline/panda_cfg${CFG}"

    echo "[$(date)] Starting baseline cfg=${CFG} on ${LAB} GPUs ${GPU1},${GPU2}"
    ssh ${LAB} "nohup bash -c '${ACTIVATE} && cd ~/${PROJ} && \
        CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python3 edit_video.py \
            --target_prompt \"${TARGET_PROMPT}\" \
            --source_prompt \"${SOURCE_PROMPT}\" \
            --source_path \"${SOURCE_PATH}\" \
            --save-path \"${SAVE}\" \
            ${COMMON_ARGS}' > /tmp/rfedit_baseline_cfg${CFG}.log 2>&1 &
    echo \$!"
}

echo "========================================"
echo "Launching distributed experiments"
echo "========================================"

# Lab1 (4 free GPUs): baseline + alpha=0.5 (2 GPUs each)
run_baseline     lab1 0 1 7
run_emb_guidance lab1 2 3 0.5

# Lab2 (GPU 2,3 free): alpha=1.0
run_emb_guidance lab2 2 3 1.0

# Lab3 (GPU 0,1 free + GPU 2,3 free): alpha=2.0, 3.0
run_emb_guidance lab3 0 1 2.0
run_emb_guidance lab3 2 3 3.0

echo ""
echo "========================================"
echo "All experiments launched!"
echo "Monitor with:"
echo "  ssh labX 'tail -f /tmp/rfedit_*.log'"
echo "========================================"
