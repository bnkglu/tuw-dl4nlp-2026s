#!/bin/bash
# =============================================================================
# Extension (broad) — Table 3 grid WITH KL distillation
#
# Same grid as run_table_3.sh (ViT-B/16, 10 datasets x shots {1,2,4,8,16} x
# seeds {1,2,3} = 150 runs) but with the knowledge-preserving KL term enabled,
# to see how KL affects EVERY dataset (not just the Food101/OxfordPets failures).
# The kl_weight=0 baseline is the existing Table 3 in results/clip_lora_results.csv.
#
# Usage: bash run_kl_table3.sh [DATA_DIR] [KL_WEIGHT] [KL_TEMP]
#   DATA_DIR   defaults to ./datasets
#   KL_WEIGHT  defaults to 0.1   (best universal setting from run_kl_ablation.sh:
#                                 helps every dataset, never hurts the EuroSAT control)
#   KL_TEMP    defaults to 8     (T=8 was >= T=4 across the ablation)
#
# Resume-safe; writes to results/clip_lora_kl.csv with the `kl_table3` tag.
# =============================================================================

set -u

cd "$(dirname "$0")/.." || exit 1

DATA_DIR="${1:-./datasets}"
KL_WEIGHT="${2:-0.1}"
KL_TEMP="${3:-8}"
CSV_FILE="./results/clip_lora_kl.csv"
BACKBONE="ViT-B/16"
LOG_DIR="./results/kl_table3/w${KL_WEIGHT}_t${KL_TEMP}"

mkdir -p "$LOG_DIR"
mkdir -p "./results"

if [ ! -f "$CSV_FILE" ]; then
    echo "table,dataset,backbone,shots,seed,rank,params,encoder,position,dropout,kl_weight,kl_temp,accuracy,status,start_time,seconds" > "$CSV_FILE"
fi

DATASETS=(eurosat caltech101 oxford_pets oxford_flowers food101 fgvc dtd ucf101 stanford_cars imagenet)
SHOTS=(1 2 4 8 16)
SEEDS=(1 2 3)

TOTAL=$(( ${#DATASETS[@]} * ${#SHOTS[@]} * ${#SEEDS[@]} ))
COUNT=0
FAILED=0

echo "=============================================="
echo " Extension — Table 3 grid + KL (ViT-B/16)"
echo " kl_weight: $KL_WEIGHT   kl_temp: $KL_TEMP"
echo " Datasets:  ${#DATASETS[@]}"
echo " Shots:     ${SHOTS[*]}"
echo " Seeds:     ${SEEDS[*]}"
echo " Total:     $TOTAL runs"
echo "=============================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    for shots in "${SHOTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            COUNT=$((COUNT + 1))
            RUN_NAME="${dataset}_s${shots}_seed${seed}"
            RUN_LOG_DIR="$LOG_DIR/${dataset}"
            mkdir -p "$RUN_LOG_DIR"
            LOG_FILE="$RUN_LOG_DIR/s${shots}_seed${seed}.log"

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
                --kl_weight "$KL_WEIGHT" \
                --kl_temp "$KL_TEMP" \
                2>&1 | tee "$LOG_FILE"

            RUN_SECS=$(( $(date +%s) - RUN_START ))

            ACC=$(grep -oP 'Final test accuracy: \K[0-9]+\.[0-9]+' "$LOG_FILE" 2>/dev/null || echo "FAILED")
            if [ "$ACC" = "FAILED" ]; then
                FAILED=$((FAILED + 1))
                STATUS="FAILED"
            else
                STATUS="OK"
            fi

            echo "kl_table3,$dataset,$BACKBONE,$shots,$seed,2,q_k_v,both,all,0.25,$KL_WEIGHT,$KL_TEMP,$ACC,$STATUS,$RUN_TS,$RUN_SECS" >> "$CSV_FILE"
            echo "  -> Accuracy: $ACC ($STATUS) in ${RUN_SECS}s"
            echo ""
        done
    done
done

echo "=============================================="
echo " Table 3 + KL complete (kl_weight=$KL_WEIGHT, kl_temp=$KL_TEMP)"
echo " Passed: $((COUNT - FAILED))/$COUNT"
echo " Failed: $FAILED/$COUNT"
echo " Results: $CSV_FILE"
echo "=============================================="
