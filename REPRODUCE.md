# Reproducing All Figures

This document provides detailed instructions for reproducing every figure in the paper. For a quick overview, see the main [README](README.md).

## Table of Contents

- [Figure 2: Iterative refinement tightens the generalisation bound](#figure-2-iterative-refinement-tightens-the-generalisation-bound)
- [Figure 3: The bidirectional consolidation mechanism](#figure-3-the-bidirectional-consolidation-mechanism)
- [Figure 4: Consolidation resolves the capacity–generalisation trade-off](#figure-4-consolidation-resolves-the-capacitygeneralisation-trade-off)
- [Figure 5: Hierarchical cache refinement in LLMs](#figure-5-hierarchical-cache-refinement-in-llms)
- [Extended Data Figure 1: The fidelity–generalisation frontier](#extended-data-figure-1-the-fidelitygeneralisation-frontier)
- [Key hyperparameters](#key-hyperparameters)
- [Output structure](#output-structure)

---

## Figure 2: Iterative refinement tightens the generalisation bound

**One-line:** `bash scripts/run_figure2.sh`

This trains the autoencoder with PonderNet-style adaptive refinement across 5 datasets (MNIST, Fashion-MNIST, EMNIST, CIFAR-10, SVHN), 20 seeds each, then computes mutual information proxies.

**Single-run example:**

```bash
python src/ae_experiment.py \
    --dataset mnist \
    --latent 128 \
    --mode ponder \
    --T 4 \
    --ponder_lambda 0.4 \
    --ponder_beta 0.02 \
    --p1_epochs 30 \
    --p3_epochs 50 \
    --p3_crossfit_ratio 0.5 \
    --seed 121 \
    --out runs/mnist_ponder_s121 \
    --no_wandb
```

Repeat for datasets: `mnist`, `fashion`, `cifar10`, `svhn`, `emnist` and seeds 121–140.

**Mutual information computation and plotting:**

```bash
python src/ae_export_reprs.py runs --replace-existing
python src/ae_compute_mi.py --runs-dir runs --outdir mi_results
python src/ae_plot_mi_from_csv.py --results-dir mi_results --outdir mi_figs
```

**Estimated GPU time:** ~2–4 hours per dataset per seed on a single GPU.

---

## Figure 3: The bidirectional consolidation mechanism

**One-line:** `bash scripts/run_figure3.sh`

Trains LPC models across datasets, then generates dream visualisations comparing wake inputs vs sleep replays.

```bash
python src/lpc_experiment.py \
    --dataset fashion \
    --seed 124 \
    --p2_head_hidden 512 \
    --p3_head_hidden 512 \
    --p3_noise_wake 0.05 \
    --out runs/lpc_fashion_s124 \
    --no_wandb

python src/lpc_plot_dream.py runs/lpc_fashion_s124
```

**Estimated GPU time:** ~1–2 hours per dataset.

---

## Figure 4: Consolidation resolves the capacity–generalisation trade-off

**One-line:** `bash scripts/run_figure4.sh`

Sweeps latent dimension d ∈ {32, 64, 128, 256, 512, 1024} on CIFAR-10, comparing Online vs Replay agents.

```bash
for lat in 32 64 128 256 512 1024; do
    python src/lpc_experiment.py \
        --dataset cifar10 \
        --latent $lat \
        --seed 124 \
        --p2_head_hidden 512 \
        --p3_head_hidden 512 \
        --p3_noise_wake 0.05 \
        --out runs/capacity_cifar10_lat${lat}_s124 \
        --no_wandb
    python src/lpc_generate_latents_for_plotting.py runs/capacity_cifar10_lat${lat}_s124
done

python src/lpc_plot_fig4.py runs/capacity_cifar10_lat512_s124
```

**Estimated GPU time:** ~6–12 hours total (larger latent dims take longer).

---

## Figure 5: Hierarchical cache refinement in LLMs

**One-line:** `bash scripts/run_figure5.sh`

> **Note:** The LLM experiments require Llama-3-8B-Instruct and the [Bottlenecked Transformer](https://openreview.net/forum?id=...) infrastructure. This script runs the **analysis and plotting** on pre-computed KV cache data.

**Data:** Download the pre-computed KV cache logs from [Figshare](https://doi.org/10.6084/m9.figshare.31534807) and place them in `sample_data/`.

```bash
# KV cache motion for a single example (panels c, d)
python src/kv_motion_figure.py --log_path sample_data/log_17 --layer 0 --step -1

# Grand-average hierarchical refinement (panel e)
python src/plot_grand_average.py --stats_dir stats_cache --out fig_grand_average.svg
```

**Run time:** Minutes on CPU.

---

## Extended Data Figure 1: The fidelity–generalisation frontier

**One-line:** `bash scripts/run_figureS1.sh`

Runs the AE experiment on MNIST with PonderNet mode (20 seeds, latent=64) for comparison against VIB and dropout baselines.

**Estimated GPU time:** ~10–20 hours total.

---

## Key Hyperparameters

| Component | Parameter | Default |
|-----------|-----------|---------|
| Data | `batch_size` | 128 |
| Encoder/Decoder (Phase 1) | `epochs` | 30 |
| | `learning_rate` | 1e-3 |
| | `weight_decay` | 1e-4 |
| Readout (Phase 2) | `epochs` | 15 |
| | `learning_rate` | 5e-4 |
| | `hidden_dim` | 128 |
| Refiner (Phase 3) | `T` (steps) | 4 |
| | `ref_hidden` | 128 |
| | `epochs` | 50 |
| | `label_smoothing` | 0.05 |
| | `z_noise` | 0.05 |
| PonderNet | `lambda` | 0.4 |
| | `beta_ponder` | 0.02 |
| Cross-fit | `ratio` | 0.5 |
| MI proxy | `sigma` | 0.1 |

See Supplementary Table 3 in the paper for full details.

---

## Output Structure

Each experiment run produces:

```
runs/<experiment_name>/
├── args.json              # Saved hyperparameters
├── phase1_best.pt         # Encoder/decoder checkpoint
├── phase2_head_best.pt    # Classifier head checkpoint
├── phase3_refiner_best.pt # Refiner checkpoint (AE only)
├── phase2_summary.json    # Phase 2 accuracy metrics
├── phase3_summary.json    # Phase 3 accuracy metrics
├── phase3_history.json    # Per-epoch training curves
├── phase3_Tsweep.csv      # Accuracy vs refinement steps
└── repr/                  # Exported latent representations
    └── <split>/Teval_<k>/zT.npy
```

---

## Lightweight Validation

Before launching expensive GPU runs, you can run a quick repository sanity check:

```bash
bash scripts/validate_repo.sh
```

This validates shell script syntax, Python syntax, and key README/script consistency.
