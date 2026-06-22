#!/bin/bash
# =============================================================================
# Extension — Knowledge-Preserving CLIP-LoRA via KL distillation
#
# Tests whether a KL term toward frozen zero-shot CLIP mitigates the Food101 /
# OxfordPets degradation the paper attributes to CE-only (unregularized) LoRA.
# Loss: CE + kl_weight * T^2 * KL(teacher || student).
#
# Grid: 3 datasets x shots {1,4,16} x kl_weight {0.1,0.3,1.0} x kl_temp {4,8}, seed 1.
# The kl_weight=0 baseline is NOT re-run here — reuse it from Table 3
# (same backbone / config) in results/clip_lora_results.csv.
#
# Usage: bash run_kl_ablation.sh [DATA_DIR] [LOG_DIR]
#   DATA_DIR defaults to ./datasets
#   LOG_DIR  defaults to ./results/kl
#
# Resume-safe: a run whose log already contains "Final test accuracy" is skipped.
# Writes to its OWN CSV (results/clip_lora_kl.csv) so it never clashes with the
# reproduction grids.
# =============================================================================

set -u

# Move to repo root so relative paths work
cd "$(dirname "$0")/.." || exit 1

DATA_DIR="${1:-./datasets}"
LOG_DIR="${2:-./results/kl}"
CSV_FILE="./results/clip_lora_kl.csv"
BACKBONE="ViT-B/16"

mkdir -p "$LOG_DIR"
mkdir -p "./results"

# Write CSV header if file doesn't exist (note the extra kl_weight / kl_temp columns)
if [ ! -f "$CSV_FILE" ]; then
    echo "table,dataset,backbone,shots,seed,rank,params,encoder,position,dropout,kl_weight,kl_temp,accuracy,status,start_time,seconds" > "$CSV_FILE"
fi

# Failure cases from the paper + one control dataset where LoRA already works well.
DATASETS=(food101 oxford_pets eurosat)

SHOTS=(1 4 16)
KL_WEIGHTS=(0.1 0.3 1.0)
KL_TEMPS=(4 8)
SEEDS=(1)   # seed 1 first; add 2 3 for the best settings only if results look interesting

TOTAL=$(( ${#DATASETS[@]} * ${#SHOTS[@]} * ${#KL_WEIGHTS[@]} * ${#KL_TEMPS[@]} * ${#SEEDS[@]} ))
COUNT=0
FAILED=0

echo "=============================================="
echo " Extension — KL distillation ablation (ViT-B/16)"
echo " Datasets:   ${DATASETS[*]}"
echo " Shots:      ${SHOTS[*]}"
echo " kl_weight:  ${KL_WEIGHTS[*]}"
echo " kl_temp:    ${KL_TEMPS[*]}"
echo " Seeds:      ${SEEDS[*]}"
echo " Total:      $TOTAL runs"
echo "=============================================="
echo ""

for dataset in "${DATASETS[@]}"; do
    for shots in "${SHOTS[@]}"; do
        for klw in "${KL_WEIGHTS[@]}"; do
            for klt in "${KL_TEMPS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    COUNT=$((COUNT + 1))
                    RUN_NAME="${dataset}_s${shots}_w${klw}_t${klt}_seed${seed}"
                    RUN_LOG_DIR="$LOG_DIR/${dataset}"
                    mkdir -p "$RUN_LOG_DIR"
                    LOG_FILE="$RUN_LOG_DIR/s${shots}_w${klw}_t${klt}_seed${seed}.log"

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
                        --kl_weight "$klw" \
                        --kl_temp "$klt" \
                        2>&1 | tee "$LOG_FILE"

                    RUN_SECS=$(( $(date +%s) - RUN_START ))

                    ACC=$(grep -oP 'Final test accuracy: \K[0-9.]+' "$LOG_FILE" 2>/dev/null || echo "FAILED")
                    if [ "$ACC" = "FAILED" ]; then
                        FAILED=$((FAILED + 1))
                        STATUS="FAILED"
                    else
                        STATUS="OK"
                    fi

                    echo "kl,$dataset,$BACKBONE,$shots,$seed,2,q_k_v,both,all,0.25,$klw,$klt,$ACC,$STATUS,$RUN_TS,$RUN_SECS" >> "$CSV_FILE"
                    echo "  -> Accuracy: $ACC ($STATUS) in ${RUN_SECS}s"
                    echo ""
                done
            done
        done
    done
done

echo "=============================================="
echo " KL ablation complete"
echo " Passed: $((COUNT - FAILED))/$COUNT"
echo " Failed: $FAILED/$COUNT"
echo " Results: $CSV_FILE"
echo "=============================================="
