#!/bin/bash
# Reproduce Figure 2: Iterative refinement tightens the generalisation bound.
#
# Trains AE with PonderNet refinement across 5 datasets, then computes
# mutual information proxies. Runs 20 seeds per dataset for error bars.
#
# Estimated GPU time: ~2-4 hours per dataset per seed on a single GPU.

set -e

SEEDS=$(seq 121 140)
DATASETS="mnist fashion cifar10 svhn emnist"

# Phase A: Train models
for dataset in $DATASETS; do
    for seed in $SEEDS; do
        out_dir="runs/${dataset}_ponder_s${seed}"
        echo "=== Training: dataset=$dataset seed=$seed ==="
        python ae_experiment.py \
            --out "$out_dir" \
            --seed "$seed" \
            --dataset "$dataset" \
            --latent 128 \
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
done

# Phase B: Export latent representations
echo "=== Exporting representations ==="
python ae_export_reprs.py runs --replace-existing

# Phase C: Compute mutual information
echo "=== Computing MI ==="
python ae_compute_mi.py --runs-dir runs --outdir mi_results

# Phase D: Plot MI figures
echo "=== Plotting MI figures ==="
python ae_plot_mi_from_csv.py --results-dir mi_results --outdir mi_figs

# Phase E: Summarise all runs
echo "=== Summarising ==="
python ae_summarise_all.py --runs_dir runs --recursive

echo "=== Done! Figures saved in mi_figs/ ==="
