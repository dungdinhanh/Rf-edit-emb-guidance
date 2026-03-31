#!/bin/bash
# Full FLUX evaluation: baseline + embedding guidance on 3 test images
# Usage: bash run_flux_eval.sh <mode> [output_base]
#   mode: "baseline" or "emb_guidance"

MODE=${1:-baseline}
OUTPUT_BASE=${2:-/hdd/dungda/video_editting/flux_eval_31march}

cd "$(dirname "$0")"

# Test cases: source_img, source_prompt, target_prompt, inject, guidance
declare -a NAMES=("horse" "hiking" "boy")
declare -a SOURCES=("examples/source/horse.jpg" "examples/source/hiking.jpg" "examples/source/boy.jpg")
declare -a SRC_PROMPTS=(
    "A young boy is riding a brown horse in a countryside field, with a large tree in the background."
    "A woman is standing in a field with a backpack on her back."
    "A little boy is sitting in a field."
)
declare -a TGT_PROMPTS=(
    "A young boy is riding a camel in a countryside field, with a large tree in the background."
    "A woman is standing in a field with a hiking stick."
    "A little boy is sitting in a field with a cute dog."
)
declare -a INJECTS=(3 4 4)
declare -a GUIDANCES=(2 2 2)

if [ "$MODE" == "baseline" ]; then
    echo "=== Running FLUX BASELINE ==="
    for i in "${!NAMES[@]}"; do
        NAME=${NAMES[$i]}
        OUTDIR="${OUTPUT_BASE}/baseline/${NAME}"
        mkdir -p "$OUTDIR"
        echo "=== Baseline: ${NAME} (inject=${INJECTS[$i]}, guidance=${GUIDANCES[$i]}) ==="
        python edit.py \
            --source_img_dir "${SOURCES[$i]}" \
            --source_prompt "${SRC_PROMPTS[$i]}" \
            --target_prompt "${TGT_PROMPTS[$i]}" \
            --inject ${INJECTS[$i]} \
            --guidance ${GUIDANCES[$i]} \
            --num_steps 25 \
            --output_dir "$OUTDIR" \
            --offload \
            --name flux-dev
        echo "=== Done: ${NAME} ==="
    done

elif [ "$MODE" == "emb_guidance" ]; then
    ALPHAS=(0.1 0.3 0.5 0.7 1.0)
    echo "=== Running FLUX EMB GUIDANCE ==="
    for i in "${!NAMES[@]}"; do
        NAME=${NAMES[$i]}
        for ALPHA in "${ALPHAS[@]}"; do
            OUTDIR="${OUTPUT_BASE}/emb_guidance/${NAME}/alpha${ALPHA}"
            mkdir -p "$OUTDIR"
            echo "=== EmbGuidance: ${NAME} alpha=${ALPHA} ==="
            python edit_emb_guidance.py \
                --source_img_dir "${SOURCES[$i]}" \
                --source_prompt "${SRC_PROMPTS[$i]}" \
                --target_prompt "${TGT_PROMPTS[$i]}" \
                --inject ${INJECTS[$i]} \
                --guidance ${GUIDANCES[$i]} \
                --num_steps 25 \
                --output_dir "$OUTDIR" \
                --offload \
                --name flux-dev \
                --emb-guidance-alpha ${ALPHA}
            echo "=== Done: ${NAME} alpha=${ALPHA} ==="
        done
    done
fi

echo "=== ALL COMPLETE ==="
