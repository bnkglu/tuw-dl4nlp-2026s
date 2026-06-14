#!/bin/bash
# =============================================================================
# Phase 0: Sanity Check
# Quick verification that environment, dataset paths, and logging work.
# Each teammate runs one backbone on EuroSAT 4-shot seed=1.
#
# Usage:
#   Teammate 1: bash run_sanity_check.sh ViT-B/16 ./datasets
#   Teammate 2: bash run_sanity_check.sh ViT-B/32 ./datasets
# =============================================================================

# Move to repo root so relative paths work
cd "$(dirname "$0")/.." || exit 1

BACKBONE="${1:-ViT-B/16}"
DATA_DIR="${2:-./datasets}"

echo "=============================================="
echo " Phase 0: Sanity Check"
echo " Backbone: $BACKBONE"
echo " Dataset:  EuroSAT, 4-shot, seed=1"
echo "=============================================="

python main.py \
    --root_path "$DATA_DIR" \
    --dataset eurosat \
    --backbone "$BACKBONE" \
    --shots 4 \
    --seed 1

echo ""
echo "Sanity check complete."
