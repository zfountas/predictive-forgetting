#!/usr/bin/env python3
"""
Plot grand-average hierarchical cache refinement across layers (Figure 5e).

Aggregates bootstrap statistics from multiple GSM8K logs to show how the
global update fraction varies across transformer layers, revealing the
transition from global renormalization to selective editing.

Usage:
  python plot_grand_average.py --stats_dir stats_cache --out fig_grand_average.svg
"""
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_dir", default="stats_cache")
    parser.add_argument("--out", default="fig_grand_average.svg")
    args = parser.parse_args()

    # 1. Load all available CSVs
    files = glob.glob(f"{args.stats_dir}/*.csv")
    if not files:
        print("No CSVs found yet!")
        return
    
    print(f"Aggregating {len(files)} logs...")
    df = pd.concat([pd.read_csv(f) for f in files])

    # 2. Compute Statistics per Layer
    # Mean of the ratios (The Main Curve)
    # Std of the ratios (Population Variance - Outer Bar)
    # Mean of CI widths (Measurement Uncertainty - Inner Bar)
    stats = df.groupby("layer").agg(
        mean_ratio=("ratio", "mean"),
        std_ratio=("ratio", "std"),
        mean_ci_width=("ci_width", "mean")
    ).reset_index()

    # 3. Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    layers = stats["layer"]
    mean = stats["mean_ratio"]
    std_pop = stats["std_ratio"]      # Variation across examples
    avg_ci = stats["mean_ci_width"]   # Bootstrap uncertainty

    # A. The Population Variance (Shaded Band)
    # This shows "How much does this vary between different math problems?"
    ax.fill_between(layers, mean - std_pop, mean + std_pop, 
                    color="blue", alpha=0.15, label="Population Variability (±1 SD)")

    # B. The Bootstrap Uncertainty (Error Bars)
    # This shows "How sure are we about the measurement within a log?"
    # We center these on the mean.
    ax.errorbar(layers, mean, yerr=avg_ci/2, fmt='o', color='blue', 
                ecolor='black', elinewidth=1.5, capsize=4, 
                label="Avg Bootstrap Uncertainty (95% CI)")

    # C. The Main Trend Line
    ax.plot(layers, mean, color="blue", linewidth=2)

    # D. Random Baseline
    # We estimate N roughly from the first log found
    N_approx = 65 
    ax.axhline(1/N_approx, color="gray", linestyle="--", label="Random Baseline")

    # Styling
    ax.set_xlabel("Transformer Layer", fontsize=12)
    ax.set_ylabel("Global Update Fraction", fontsize=12)
    ax.set_title(f"Hierarchical Cache Refinement (N={len(files)} logs)", fontsize=14)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Add regime annotations
    ax.text(2, 0.1, "Global Renormalization", fontsize=12, color='blue', alpha=0.5, ha='center')
    ax.text(14, 0.1, "Selective Editing", fontsize=12, color='orange', alpha=0.8, ha='center')

    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()