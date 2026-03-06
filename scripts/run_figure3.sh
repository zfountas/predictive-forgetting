#!/bin/bash
# Reproduce Figure 3: The bidirectional consolidation mechanism.
#
# Trains LPC models across datasets, then generates dream visualisations
# comparing wake inputs vs sleep replays.
#
# Estimated GPU time: ~1-2 hours per dataset on a single GPU.

set -e

DATASETS="mnist fashion cifar10 svhn emnist"

# Phase A: Train LPC models
for dataset in $DATASETS; do
    out_dir="runs/lpc_${dataset}_s124"
    echo "=== Training LPC: dataset=$dataset ==="
    python lpc_experiment.py \
        --out "$out_dir" \
        --seed 124 \
        --dataset "$dataset" \
        --p2_head_hidden 512 \
        --p3_head_hidden 512 \
        --p3_noise_wake 0.05 \
        --no_wandb
done

# Phase B: Generate dream visualisations (Figure 3c)
for dataset in $DATASETS; do
    out_dir="runs/lpc_${dataset}_s124"
    echo "=== Generating dreams: $dataset ==="
    python lpc_plot_dream.py "$out_dir"
done

echo "=== Done! Dream figures saved in each run directory ==="
