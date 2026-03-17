# Predictive Forgetting: A Computational Theory of Memory Consolidation

Code for reproducing the experiments in:

> **Why the Brain Consolidates: Predictive Forgetting for Optimal Generalisation**
> Zafeirios Fountas, Adnan Oomerjee, Haitham Bou-Ammar, Jun Wang, Neil Burgess
>
> Paper: [arXiv:2603.04688](https://arxiv.org/abs/2603.04688)

This repository contains the full pipeline for training, evaluating, and visualising
the autoencoder (AE) and Langevin Predictive Coding (LPC) experiments, as well as
analysis scripts for the LLM key-value cache consolidation experiments.

## Repository Structure

```
predictive-forgetting/
├── src/ae_experiment.py              # AE refinement pipeline (Figures 2, S1)
├── src/lpc_experiment.py             # LPC wake/sleep pipeline (Figures 3, 4)
├── src/run_multiseed.py              # Multi-seed experiment runner
├── src/ae_export_reprs.py            # Export latent representations for MI
├── src/ae_compute_mi.py              # Compute mutual information proxies
├── src/ae_summarise_all.py           # Aggregate results across seeds
├── src/api_adapter.py                # Model loading bridge
├── src/ae_plots.py                   # Single-run result plotting
├── src/ae_plot_mi_from_csv.py        # MI curve plotting
├── src/lpc_plots.py                  # LPC result plotting
├── src/lpc_plot_dream.py             # Dream visualisation (Figure 3c)
├── src/lpc_plot_fig4.py              # Capacity sweep figure (Figure 4)
├── src/lpc_plot_for_paper_figs.py    # t-SNE and alignment figures
├── src/lpc_generate_latents_for_plotting.py  # Generate latents for Figure 4
├── src/kv_motion_figure.py           # KV cache motion analysis (Figure 5)
├── src/plot_grand_average.py         # Grand-average cache refinement (Figure 5e)
├── sample_data/                  # Pre-computed KV cache logs for Figure 5
├── scripts/                      # Shell scripts to reproduce each figure
│   ├── run_figure2.sh
│   ├── run_figure3.sh
│   ├── run_figure4.sh
│   ├── run_figure5.sh
│   └── run_figureS1.sh
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/zfountas/predictive-forgetting.git
cd predictive-forgetting
pip install -r requirements.txt
```

Requires Python 3.8+ and PyTorch 1.10+. A CUDA-capable GPU is recommended.

## Quick Start

To reproduce all figures:

```bash
# Figure 2: AE refinement across 5 datasets
bash scripts/run_figure2.sh

# Figure 3: Wake/sleep dream visualisation
bash scripts/run_figure3.sh

# Figure 4: Capacity-generalisation sweep
bash scripts/run_figure4.sh

# Figure 5: LLM KV cache analysis (requires pre-computed data)
bash scripts/run_figure5.sh

# Extended Data Figure S1: Pareto frontier
bash scripts/run_figureS1.sh
```

**Compute requirements:** Figures 2 and S1 together involve training 120 models
(Figure 2: 5 datasets x 20 seeds = 100 models; Figure S1: 1 dataset x 20 seeds = 20 models),
and are the most compute-intensive (~2-4 GPU-hours per model on a single GPU).
Figures 3 and 4 require ~1-2 GPU-hours each. Figure 5 analysis runs in minutes on CPU.

## Reproducing Individual Figures

### Figure 2: Iterative refinement tightens the generalisation bound

Train the autoencoder with PonderNet-style adaptive refinement:

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

Repeat for datasets: `mnist`, `fashion`, `cifar10`, `svhn`, `emnist` and seeds 121-140.

Then compute and plot mutual information:

```bash
python src/ae_export_reprs.py runs --replace-existing
python src/ae_compute_mi.py --runs-dir runs --outdir mi_results
python src/ae_plot_mi_from_csv.py --results-dir mi_results --outdir mi_figs
```

### Figure 3: Bidirectional consolidation mechanism

Train LPC models and generate dream visualisations:

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

### Figure 4: Capacity-generalisation trade-off

Sweep latent dimension on CIFAR-10:

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

### Figure 5: LLM KV cache refinement

The LLM experiments require Llama-3 and the bottlenecked transformer
infrastructure. The analysis scripts operate on pre-computed KV cache data.

**Data:** A curated set of 100 pre-computed KV cache logs is publicly available on Figshare:

> Fountas, Z. & Oomerjee, A. (2026). *KV Cache Consolidation Logs — Predictive
> Forgetting for Optimal Generalisation (Figure 5)*. Figshare.
> https://doi.org/10.6084/m9.figshare.31534807

Download these files and place them in `sample_data/` before running the scripts below.
The full dataset (~60 GB, N=1,318 logs) is available upon reasonable request to the
corresponding author (zafeirios.fountas@huawei.com).
```bash
# Plot KV cache motion for a single example
# (replace with any existing sample_data/log_* file)
python src/kv_motion_figure.py --log_path sample_data/log_17 --layer 0 --step -1

# Plot grand-average hierarchical refinement
python src/plot_grand_average.py --stats_dir stats_cache --out fig_grand_average.svg
```

## Key Hyperparameters

| Component | Parameter | Default |
|-----------|-----------|---------|
| Data | batch_size | 128 |
| Encoder/Decoder (Phase 1) | epochs | 30 |
| | learning_rate | 1e-3 |
| | weight_decay | 1e-4 |
| Readout (Phase 2) | epochs | 15 |
| | learning_rate | 5e-4 |
| | hidden_dim | 128 |
| Refiner (Phase 3) | T (steps) | 4 |
| | ref_hidden | 128 |
| | epochs | 50 |
| | label_smoothing | 0.05 |
| | z_noise | 0.05 |
| PonderNet | lambda | 0.4 |
| | beta_ponder | 0.02 |
| Cross-fit | ratio | 0.5 |
| MI proxy | sigma | 0.1 |

See Table 2 in the paper for full details.

## Datasets

All datasets are downloaded automatically via torchvision:
- **MNIST** (28x28, grayscale, 10 classes)
- **Fashion-MNIST** (28x28, grayscale, 10 classes)
- **EMNIST** (28x28, grayscale, 47 classes, balanced split)
- **CIFAR-10** (32x32, RGB, 10 classes)
- **SVHN** (32x32, RGB, 10 classes)

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

## Citation

```bibtex
@misc{fountas2026predictiveforgetting,
  title         = {Why the Brain Consolidates: Predictive Forgetting
                   for Optimal Generalisation},
  author        = {Zafeirios Fountas and Adnan Oomerjee and
                   Haitham Bou-Ammar and Jun Wang and Neil Burgess},
  year          = {2026},
  eprint        = {2603.04688},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2603.04688}
}
```



## License

MIT License. See [LICENSE](LICENSE) for details.

## Lightweight Validation

Before launching expensive GPU runs, you can run a quick repository sanity check:

```bash
bash scripts/validate_repo.sh
```

This validates shell script syntax, Python syntax, and key README/script consistency.
