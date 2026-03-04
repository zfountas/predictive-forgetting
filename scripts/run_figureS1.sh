#!/bin/bash
# Reproduce Extended Data Figure S1: The fidelity-generalisation frontier.
#
# Runs the AE experiment on MNIST with ponder mode, then generates
# the Pareto frontier comparison against VIB and dropout baselines.
#
# Estimated GPU time: ~10-20 hours total (20 seeds).

set -e

SEEDS=$(seq 121 140)

# Phase A: Train AE models (same hyperparameters as Figure 2 but MNIST only)
for seed in $SEEDS; do
    out_dir="runs/pareto_mnist_ponder_s${seed}"
    echo "=== Training: seed=$seed ==="
    python ae_experiment.py \
        --out "$out_dir" \
        --seed "$seed" \
        --dataset mnist \
        --latent 64 \
        --mode ponder \
        --T 4 \
        --ponder_lambda 0.4 \
        --ponder_beta 0.02 \
        --p1_epochs 30 \
        --p3_ref_hidden 128 \
        --p3_wd 0.001 \
        --p3_znoise 0.05 \
        --p3_contract 0.001 \
        --p3_epochs 50 \
        --p3_crossfit_ratio 0.5 \
        --no_wandb
done

# Phase B: Summarise results
echo "=== Summarising ==="
python ae_summarise_all.py --runs_dir runs --recursive --datasets mnist

echo "=== Done! ==="
