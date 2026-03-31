#!/bin/bash
# Full evaluation: baseline + embedding guidance on all 3 paper examples
# Usage: bash scripts/run_full_eval.sh <gpu1> <gpu2> <mode> [output_base]
#   mode: "baseline" or "emb_guidance"
#   output_base: default /hdd/dungda/video_editting/eval_31march
#
# Examples:
#   bash scripts/run_full_eval.sh 2 3 baseline
#   bash scripts/run_full_eval.sh 2 3 emb_guidance

GPU1=${1:-0}
GPU2=${2:-1}
MODE=${3:-baseline}
OUTPUT_BASE=${4:-/hdd/dungda/video_editting/eval_31march}

cd "$(dirname "$0")/.."

SEED=42

# Reduce CUDA memory fragmentation across sequential runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paper test cases: video, source_prompt, target_prompt, inject_step, embedded_cfg_scale
declare -a VIDEOS=("panda" "rabbit" "jeep")
declare -a SOURCES=("./scripts/panda.mp4" "./scripts/rabbit_watermelon.mp4" "./scripts/jeep.mp4")
declare -a SRC_PROMPTS=("A panda walking in the snow" "" "")
declare -a TGT_PROMPTS=("A panda wearing a Crown walking in the snow" "A cat is eating a watermelon" "A pink porsche")
declare -a INJECT_STEPS=(3 1 2)
declare -a CFG_SCALES=(7 1 6)

if [ "$MODE" == "baseline" ]; then
    echo "=== Running BASELINE (standard CFG) ==="
    for i in "${!VIDEOS[@]}"; do
        VID=${VIDEOS[$i]}
        OUTDIR="${OUTPUT_BASE}/baseline/${VID}"
        mkdir -p "$OUTDIR"
        echo "=== Baseline: ${VID} (inject=${INJECT_STEPS[$i]}, cfg=${CFG_SCALES[$i]}) ==="
        CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python3 edit_video.py \
            --target_prompt "${TGT_PROMPTS[$i]}" \
            --infer-steps 25 \
            --source_prompt "${SRC_PROMPTS[$i]}" \
            --flow-reverse \
            --multi-gpu --gpu-ids 0 1 \
            --save-path "$OUTDIR" \
            --source_path "${SOURCES[$i]}" \
            --inject_step ${INJECT_STEPS[$i]} \
            --embedded-cfg-scale ${CFG_SCALES[$i]} \
            --seed $SEED
        echo "=== Done: ${VID} ==="
    done

elif [ "$MODE" == "emb_guidance" ]; then
    # Sweep alpha values with embedded-cfg-scale matching paper defaults
    ALPHAS=(0.1 0.3 0.5 0.7 1.0)

    echo "=== Running EMB GUIDANCE ==="
    for i in "${!VIDEOS[@]}"; do
        VID=${VIDEOS[$i]}
        for ALPHA in "${ALPHAS[@]}"; do
            OUTDIR="${OUTPUT_BASE}/emb_guidance/${VID}/alpha${ALPHA}_cfg${CFG_SCALES[$i]}"
            mkdir -p "$OUTDIR"
            echo "=== EmbGuidance: ${VID} alpha=${ALPHA} cfg=${CFG_SCALES[$i]} ==="
            CUDA_VISIBLE_DEVICES=${GPU1},${GPU2} python3 edit_video_emb_guidance.py \
                --target_prompt "${TGT_PROMPTS[$i]}" \
                --infer-steps 25 \
                --source_prompt "${SRC_PROMPTS[$i]}" \
                --flow-reverse \
                --multi-gpu --gpu-ids 0 1 \
                --save-path "$OUTDIR" \
                --source_path "${SOURCES[$i]}" \
                --inject_step ${INJECT_STEPS[$i]} \
                --embedded-cfg-scale ${CFG_SCALES[$i]} \
                --emb-guidance-alpha ${ALPHA} \
                --seed $SEED
            echo "=== Done: ${VID} alpha=${ALPHA} ==="
        done
    done
fi

echo "=== ALL COMPLETE ==="
