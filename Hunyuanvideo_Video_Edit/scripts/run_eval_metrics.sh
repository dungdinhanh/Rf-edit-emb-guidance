#!/bin/bash
# Evaluate all generated videos using video metrics (PSNR, SSIM, LPIPS)
# Usage: bash scripts/run_eval_metrics.sh [output_base]

OUTPUT_BASE=${1:-/hdd/dungda/video_editting/eval_31march}
EVAL_JSON="${OUTPUT_BASE}/eval_results.json"

cd "$(dirname "$0")/../.."

echo "=== Evaluating all results in ${OUTPUT_BASE} ==="
echo "{" > "$EVAL_JSON"
FIRST=true

# Source videos
declare -A SOURCE_VIDEOS
SOURCE_VIDEOS[panda]="Hunyuanvideo_Video_Edit/scripts/panda.mp4"
SOURCE_VIDEOS[rabbit]="Hunyuanvideo_Video_Edit/scripts/rabbit_watermelon.mp4"
SOURCE_VIDEOS[jeep]="Hunyuanvideo_Video_Edit/scripts/jeep.mp4"

declare -A TARGET_PROMPTS
TARGET_PROMPTS[panda]="A panda wearing a Crown walking in the snow"
TARGET_PROMPTS[rabbit]="A cat is eating a watermelon"
TARGET_PROMPTS[jeep]="A pink porsche"

# Find all mp4 results
find "$OUTPUT_BASE" -name "*.mp4" | sort | while read EDITED; do
    # Determine which video this is
    for VID in panda rabbit jeep; do
        if echo "$EDITED" | grep -q "/$VID/"; then
            SOURCE="${SOURCE_VIDEOS[$VID]}"
            PROMPT="${TARGET_PROMPTS[$VID]}"
            RELPATH=$(echo "$EDITED" | sed "s|${OUTPUT_BASE}/||")
            echo ""
            echo "=== Evaluating: ${RELPATH} ==="
            python -m evaluation.evaluate video \
                --source "$SOURCE" \
                --edited "$EDITED" \
                -o "${EDITED%.mp4}_metrics.json"
            break
        fi
    done
done

echo "=== Evaluation complete ==="
echo "Per-video metrics saved as *_metrics.json alongside each video"
