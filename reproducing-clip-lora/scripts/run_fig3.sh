#!/bin/bash
# =============================================================================
# Phase 2 — Figure 3: Ablation grid for a given dataset
#
# Runs the full ablation grid (rank × matrix set × encoder + placement set)
# for one dataset at a time. Each teammate runs their assigned datasets.
#
# Usage:
#   bash run_fig3.sh imagenet ./datasets
#   bash run_fig3.sh eurosat  ./datasets
#   bash run_fig3.sh stanford_cars ./datasets
#
# With nohup (master log under logs/, which is git-ignored):
#   mkdir -p logs
#   nohup bash run_fig3.sh eurosat ./datasets > logs/fig3_eurosat.log 2>&1 &
# =============================================================================

set -u

# Move to repo root so relative paths work
cd "$(dirname "$0")/.." || exit 1

DATASET="${1:?Usage: bash run_fig3.sh <dataset> [DATA_DIR]}"
DATA_DIR="${2:-./datasets}"
LOG_DIR="./results/fig3"
CSV_FILE="./results/clip_lora_results.csv"
BACKBONE="ViT-B/16"
SHOTS=4
SEEDS=(1)  # Start with seed 1; add 2 and 3 if time permits

mkdir -p "$LOG_DIR"
mkdir -p "./results"

if [ ! -f "$CSV_FILE" ]; then
    echo "table,dataset,backbone,shots,seed,rank,params,encoder,position,dropout,accuracy,status,start_time,seconds" > "$CSV_FILE"
fi

# ---- Main grid: rank × matrix set × encoder (placement = all) ----
RANKS=(1 2 4 8 16 32)
ENCODERS=(vision text both)

# Matrix sets: space-separated params for --params
MATRIX_SETS=(
    "k"
    "q"
    "v"
    "o"
    "q v"
    "q v k"
    "q v k o"
)
MATRIX_LABELS=(
    "k"
    "q"
    "v"
    "o"
    "q_v"
    "q_v_k"
    "q_v_k_o"
)

# ---- Placement set: rank=2, params=q k v, encoder=both ----
PLACEMENTS=(bottom up all)

# Count total runs
MAIN_GRID=$(( ${#RANKS[@]} * ${#MATRIX_SETS[@]} * ${#ENCODERS[@]} * ${#SEEDS[@]} ))
PLACEMENT_RUNS=$(( ${#PLACEMENTS[@]} * ${#SEEDS[@]} ))
TOTAL=$((MAIN_GRID + PLACEMENT_RUNS))
COUNT=0
FAILED=0

echo "=============================================="
echo " Phase 2 — Figure 3 Ablations"
echo " Dataset:  $DATASET"
echo " Backbone: $BACKBONE"
echo " Shots:    $SHOTS"
echo " Seeds:    ${SEEDS[*]}"
echo " Main grid:     $MAIN_GRID runs"
echo " Placement set: $PLACEMENT_RUNS runs"
echo " Total:         $TOTAL runs"
echo "=============================================="
echo ""

run_one() {
    local rank="$1"
    local params_label="$2"
    local encoder="$3"
    local position="$4"
    local seed="$5"
    shift 5
    local params_args=("$@")

    COUNT=$((COUNT + 1))
    local RUN_NAME="${DATASET}_r${rank}_${params_label}_${encoder}_${position}_seed${seed}"
    local RUN_LOG_DIR="$LOG_DIR/${DATASET}"
    mkdir -p "$RUN_LOG_DIR"
    local LOG_FILE="$RUN_LOG_DIR/r${rank}_${params_label}_${encoder}_${position}_seed${seed}.log"

    if [ -f "$LOG_FILE" ] && grep -q "Final test accuracy" "$LOG_FILE" 2>/dev/null; then
        echo "[$COUNT/$TOTAL] SKIP $RUN_NAME (already done)"
        return
    fi

    echo "[$COUNT/$TOTAL] RUN  $RUN_NAME ..."
    local RUN_TS RUN_START RUN_SECS
    RUN_TS=$(date +%Y-%m-%dT%H:%M:%S)
    RUN_START=$(date +%s)

    python main.py \
        --root_path "$DATA_DIR" \
        --dataset "$DATASET" \
        --backbone "$BACKBONE" \
        --shots "$SHOTS" \
        --seed "$seed" \
        --r "$rank" \
        --params "${params_args[@]}" \
        --encoder "$encoder" \
        --position "$position" \
        2>&1 | tee "$LOG_FILE"

    RUN_SECS=$(( $(date +%s) - RUN_START ))

    local ACC
    ACC=$(grep -oP 'Final test accuracy: \K[0-9.]+' "$LOG_FILE" 2>/dev/null || echo "FAILED")
    local STATUS="OK"
    if [ "$ACC" = "FAILED" ]; then
        FAILED=$((FAILED + 1))
        STATUS="FAILED"
    fi

    echo "fig3,$DATASET,$BACKBONE,$SHOTS,$seed,$rank,$params_label,$encoder,$position,0.25,$ACC,$STATUS,$RUN_TS,$RUN_SECS" >> "$CSV_FILE"
    echo "  -> Accuracy: $ACC ($STATUS) in ${RUN_SECS}s"
    echo ""
}

# ---- Run main grid ----
echo "=== Main Grid (rank × matrix × encoder, placement=all) ==="
echo ""
for seed in "${SEEDS[@]}"; do
    for rank in "${RANKS[@]}"; do
        for i in "${!MATRIX_SETS[@]}"; do
            IFS=' ' read -ra params_arr <<< "${MATRIX_SETS[$i]}"
            label="${MATRIX_LABELS[$i]}"
            for encoder in "${ENCODERS[@]}"; do
                run_one "$rank" "$label" "$encoder" "all" "$seed" "${params_arr[@]}"
            done
        done
    done
done

# ---- Run placement set ----
echo "=== Placement Set (rank=2, params=q k v, encoder=both) ==="
echo ""
for seed in "${SEEDS[@]}"; do
    for position in "${PLACEMENTS[@]}"; do
        run_one "2" "q_k_v" "both" "$position" "$seed" q k v
    done
done

echo "=============================================="
echo " Figure 3 Complete for $DATASET"
echo " Passed: $((COUNT - FAILED))/$COUNT"
echo " Failed: $FAILED/$COUNT"
echo " Results: $CSV_FILE"
echo "=============================================="
