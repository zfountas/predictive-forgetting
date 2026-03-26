# Predictive Forgetting

Code for reproducing the experiments in:

> **Why the Brain Consolidates: Predictive Forgetting for Optimal Generalisation**
> Zafeirios Fountas, Adnan Oomerjee, Haitham Bou-Ammar, Jun Wang, Neil Burgess
>
> Paper: [arXiv:2603.04688](https://arxiv.org/abs/2603.04688)

This repository contains the full pipeline for the autoencoder (AE) refinement experiments, the Langevin Predictive Coding (LPC) wake/sleep experiments, and the analysis scripts for the LLM key-value cache consolidation experiments reported in the paper.

## System Requirements

### Software dependencies

- Python 3.8+ (tested with Python 3.10)
- PyTorch 1.10+ (tested with PyTorch 2.1, CUDA 12.1)
- NumPy ≥ 1.20, SciPy ≥ 1.7, pandas ≥ 1.3, matplotlib ≥ 3.4, seaborn ≥ 0.11
- scikit-learn ≥ 1.0 (optional, for t-SNE visualisations)
- Full dependency list: [`requirements.txt`](requirements.txt)

### Operating systems

Tested on Ubuntu Linux (Google Colab). Compatible with any OS supporting Python and PyTorch (Linux, macOS, Windows).

### Hardware

- **Figures 2–4 and Extended Data Fig. 1** (AE and LPC experiments): A single NVIDIA GPU (e.g., A100, V100, or RTX 3090) is recommended. CPU execution is supported but significantly slower.
- **Figure 5** (LLM KV cache analysis): Runs on CPU in minutes using pre-computed data. Full LLM pipeline requires ≥ 40 GB GPU VRAM (e.g., NVIDIA A100) for Llama-3 inference.
- **Minimum RAM**: 8 GB (16 GB recommended for LLM analysis).

## Installation

```bash
git clone https://github.com/zfountas/predictive-forgetting.git
cd predictive-forgetting
pip install -r requirements.txt
```

Typical install time: **~5 minutes** on a standard desktop computer (excluding PyTorch/CUDA setup, which varies by system).

## Demo

A quick demo reproduces the core result (Figure 2) on MNIST with a single seed:

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
    --out runs/demo \
    --no_wandb
```

**Expected output:** The run directory `runs/demo/` will contain:
- `phase2_summary.json` — baseline classifier accuracy (expected test accuracy ~0.975 on MNIST)
- `phase3_summary.json` — refined accuracy and generalisation gap after consolidation
- `phase3_Tsweep.csv` — accuracy and generalisation gap at each refinement step (T=0 to T=4), showing monotonic gap reduction consistent with Figure 2c

**Expected run time:** ~10–15 minutes on a single GPU; ~1–2 hours on CPU.

## Reproducing All Figures

Shell scripts are provided to reproduce each figure in the paper. Detailed instructions, including hyperparameter tables and per-figure commands, are in [`REPRODUCE.md`](REPRODUCE.md).

| Figure | Script | Description | Estimated GPU time |
|--------|--------|-------------|--------------------|
| Figure 2 | `bash scripts/run_figure2.sh` | AE refinement across 5 datasets (20 seeds each) | ~2–4 hours per dataset × seed |
| Figure 3 | `bash scripts/run_figure3.sh` | Wake/sleep dream visualisation | ~1–2 hours per dataset |
| Figure 4 | `bash scripts/run_figure4.sh` | Capacity–generalisation sweep on CIFAR-10 | ~6–12 hours total |
| Figure 5 | `bash scripts/run_figure5.sh` | LLM KV cache analysis (pre-computed data) | Minutes (CPU) |
| Ext. Data Fig. 1 | `bash scripts/run_figureS1.sh` | Fidelity–generalisation frontier | ~10–20 hours total |

## Data

All image classification datasets (MNIST, Fashion-MNIST, EMNIST, CIFAR-10, SVHN) are downloaded automatically via `torchvision` on first run.

Pre-computed KV cache data for Figure 5 (100 example logs) is available on Figshare:

> Fountas, Z. & Oomerjee, A. (2026). *KV Cache Consolidation Logs — Predictive Forgetting for Optimal Generalisation (Figure 5)*. Figshare.  
> [https://doi.org/10.6084/m9.figshare.31534807](https://doi.org/10.6084/m9.figshare.31534807)

Download and place the log files in `sample_data/` before running `scripts/run_figure5.sh`. The full dataset (~60 GB, N=1,318 logs) is available upon reasonable request to the corresponding author.

## Repository Structure

```
predictive-forgetting/
├── src/
│   ├── ae_experiment.py          # AE refinement pipeline (Figures 2, Ext. Data Fig. 1)
│   ├── lpc_experiment.py         # LPC wake/sleep pipeline (Figures 3, 4)
│   ├── ae_compute_mi.py          # Mutual information proxy computation
│   ├── ae_export_reprs.py        # Export latent representations for MI analysis
│   ├── kv_motion_figure.py       # KV cache motion analysis (Figure 5)
│   ├── plot_grand_average.py     # Grand-average hierarchical refinement (Figure 5e)
│   └── ...                       # Additional plotting and utility scripts
├── scripts/                      # Shell scripts to reproduce each figure
├── sample_data/                  # Pre-computed KV cache logs for Figure 5
├── requirements.txt
├── REPRODUCE.md                  # Detailed reproduction instructions
└── LICENSE                       # MIT License
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

[MIT License](LICENSE)
