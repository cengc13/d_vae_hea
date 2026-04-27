#!/usr/bin/env bash
# Example inference pipeline using the pre-trained SSVAE checkpoint.
# Run from the project root:  bash run_inference.sh
set -euo pipefail

MODEL="models/ssvae.model"
DATA="data"
FIGURES="figures"

echo "=== 1. Accuracy + ROC curves + latent space plots ==="
python3 evaluate.py \
    --model-path "$MODEL" \
    --data-dir   "$DATA" \
    --figures-dir "$FIGURES"

echo ""
echo "=== 2. Alloy reconstruction error analysis ==="
python3 reconstruct.py \
    --model-path  "$MODEL" \
    --data-dir    "$DATA" \
    --figures-dir "$FIGURES" \
    --output-csv  "$DATA/test_data_reconstruction_analysis.csv"

echo ""
echo "=== 3. Latent space scan (refractory single-phase region) ==="
python3 interpolate.py \
    --model-path  "$MODEL" \
    --z1-range -0.1 0.1 \
    --z2-range -0.5 -0.3 \
    --n-points 5 \
    --phase-label 1

echo ""
echo "=== 4. Iterative inverse design: push a multi-phase alloy toward single-phase ==="
python3 interpolate.py \
    --model-path  "$MODEL" \
    --alloy       "Al1.4Co0.9Cr1.4Cu0.5Fe0.9Ni1" \
    --target-prob 0.6 \
    --max-iter    10

echo ""
echo "=== 5. SHAP feature importance (slow — uses KernelExplainer on 138 test samples) ==="
python3 shap_analysis.py \
    --model-path  "$MODEL" \
    --data-dir    "$DATA" \
    --figures-dir "$FIGURES"

echo ""
echo "Done. Figures written to $FIGURES/"
