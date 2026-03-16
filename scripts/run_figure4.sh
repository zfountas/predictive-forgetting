#!/bin/bash
# Reproduce Figure 4: Consolidation resolves the capacity-generalisation trade-off.
#
# Sweeps latent dimension d in {32, 64, 128, 256, 512, 1024} on CIFAR-10,
# comparing Online vs Replay agents.
#
# Estimated GPU time: ~6-12 hours total (larger latent dims take longer).

set -e

LATENT_DIMS="32 64 128 256 512 1024"

# Phase A: Train LPC models at each capacity
for lat in $LATENT_DIMS; do
    out_dir="runs/capacity_cifar10_lat${lat}_s124"
    echo "=== Training LPC: latent=$lat ==="
    python src/lpc_experiment.py \
        --out "$out_dir" \
        --seed 124 \
        --dataset cifar10 \
        --latent "$lat" \
        --p2_head_hidden 512 \
        --p3_head_hidden 512 \
        --p3_noise_wake 0.05 \
        --no_wandb
done

# Phase B: Generate latent representations for plotting
for lat in $LATENT_DIMS; do
    out_dir="runs/capacity_cifar10_lat${lat}_s124"
    echo "=== Generating latents: latent=$lat ==="
    python src/lpc_generate_latents_for_plotting.py "$out_dir"
done

# Phase C: Plot Figure 4 for the high-capacity case (d=512)
echo "=== Plotting Figure 4 ==="
python src/lpc_plot_fig4.py runs/capacity_cifar10_lat512_s124

echo "=== Done! Figure saved in runs/capacity_cifar10_lat512_s124/ ==="
