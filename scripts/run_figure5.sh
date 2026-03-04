#!/bin/bash
# Reproduce Figure 5: Hierarchical cache refinement in large language models.
#
# NOTE: The LLM experiments require a server with Llama-3-8B-Instruct and
# the bottlenecked transformer infrastructure. This script only runs the
# *analysis and plotting* on pre-computed KV cache data.
#
# The KV cache data must be placed in sample_data/ as PyTorch tensor files.
#
# To reproduce the full LLM pipeline, see the paper's Methods section.

set -e

# Plot KV cache motion for a single example (panels a-c)
if ls sample_data/log_* 1> /dev/null 2>&1; then
    echo "=== Plotting KV motion figure ==="
    python kv_motion_figure.py \
        --log_path sample_data/log_17 \
        --layer 0 \
        --step -1 \
        --max_arrows 200 \
        --out kv_motion_layer0
else
    echo "[SKIP] No KV cache data found in sample_data/"
    echo "Place log files (e.g., log_17) in sample_data/ to generate KV motion figures."
fi

# Plot grand average across layers (panel e)
if [ -d "stats_cache" ]; then
    echo "=== Plotting grand average ==="
    python plot_grand_average.py --stats_dir stats_cache --out fig_grand_average.svg
else
    echo "[SKIP] No stats_cache/ directory found."
    echo "Place bootstrap CSV files in stats_cache/ to generate the grand average figure."
fi

echo "=== Done! ==="
