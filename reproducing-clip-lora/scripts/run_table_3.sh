#!/bin/bash
# =============================================================================
# Phase 1.A — Table 3: ViT-B/16 across all available datasets
# Runs: 10 datasets × 5 shots × 3 seeds = 150 runs
#
# Usage: bash run_table_3.sh [DATA_DIR] [LOG_DIR]
#   DATA_DIR defaults to ./datasets
#   LOG_DIR  defaults to ./results/table3
#
# Each run logs to a separate file. A CSV summary is appended after each run.
# To resume after a crash, the script skips runs whose log files already exist.
# =============================================================================

set -u

# Move to repo root so relative paths work
cd "$(dirname "$0")/.." || exit 1

DATA_DIR="${1:-./datasets}"
LOG_DIR="${2:-./results/table3}"
CSV_FILE="./results/clip_lora_results.csv"
BACKBONE="ViT-B/16"

mkdir -p "$LOG_DIR"
mkdir -p "./results"

# Write CSV header if file doesn't exist
if [ ! -f "$CSV_FILE" ]; then
    echo "table,dataset,backbone,shots,seed,rank,params,encoder,position,dropout,accuracy,status,start_time,seconds" > "$CSV_FILE"
fi

# All available datasets (excluding SUN397 due to dead URLs)
DATASETS=(eurosat caltech101 oxford_pets oxford_flowers food101 fgvc dtd ucf101 stanford_cars imagenet)

SHOTS=(1 2 4 8 16)
SEEDS=(1 2 3)

TOTAL=$(( ${#DATASETS[@]} * ${#SHOTS[@]} * ${#SEEDS[@]} ))
COUNT=0
FAILED=0

echo "=============================================="
echo " Phase 1.A — Table 3 (ViT-B/16)"
echo " Datasets: ${#DATASETS[@]}"
echo " Shots:    ${SHOTS[*]}"
echo " Seeds:    ${SEEDS[*]}"
echo " Total:    $TOTAL runs"
echo "=============================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    for shots in "${SHOTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))
            RUN_NAME="${dataset}_s${shots}_seed${seed}"
            LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

            # Skip if already completed successfully
            if [ -f "$LOG_FILE" ] && grep -q "Final test accuracy" "$LOG_FILE" 2>/dev/null; then
                echo "[$COUNT/$TOTAL] SKIP $RUN_NAME (already done)"
                continue
            fi

            echo "[$COUNT/$TOTAL] RUN  $RUN_NAME ..."
            RUN_TS=$(date +%Y-%m-%dT%H:%M:%S)
            RUN_START=$(date +%s)

            python main.py \
                --root_path "$DATA_DIR" \
                --dataset "$dataset" \
                --backbone "$BACKBONE" \
                --shots "$shots" \
                --seed "$seed" \
                2>&1 | tee "$LOG_FILE"

            RUN_SECS=$(( $(date +%s) - RUN_START ))

            # Extract accuracy from log and append to CSV
            ACC=$(grep -oP 'Final test accuracy: \K[0-9.]+' "$LOG_FILE" 2>/dev/null || echo "FAILED")
            if [ "$ACC" = "FAILED" ]; then
                FAILED=$((FAILED + 1))
                STATUS="FAILED"
            else
                STATUS="OK"
            fi

            echo "table3,$dataset,$BACKBONE,$shots,$seed,2,q_k_v,both,all,0.25,$ACC,$STATUS,$RUN_TS,$RUN_SECS" >> "$CSV_FILE"
            echo "  -> Accuracy: $ACC ($STATUS) in ${RUN_SECS}s"
            echo ""
        done
    done
done

echo "=============================================="
echo " Table 3 Complete"
echo " Passed: $((COUNT - FAILED))/$COUNT"
echo " Failed: $FAILED/$COUNT"
echo " Results: $CSV_FILE"
echo "=============================================="
