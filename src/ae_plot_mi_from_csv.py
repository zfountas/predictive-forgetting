"""
Plot mutual information figures from CSV outputs of ae_compute_mi.py.

Generates: I(Y;Z_T) curves, I(X;Z_T) curves, and I(X;Z) vs I(Y;Z) scatter.

Usage:
  python src/ae_plot_mi_from_csv.py --results-dir mi_results --outdir mi_figs
  python src/ae_plot_mi_from_csv.py --results-dir mi_results --dataset mnist --split test
"""

import argparse, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="mi_results",
                    help="Directory with mi_points_*.csv and mi_stats_*.csv")
    ap.add_argument("--outdir", type=str, default="mi_figs",
                    help="Where to save figures.")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Optional dataset filter (e.g. mnist).")
    ap.add_argument("--family", type=str, default=None,
                    help="Optional family filter (exact match on 'family' column).")
    ap.add_argument("--split", type=str, default="test",
                    help="Split for scatter / default plots (train/val/test).")
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    res_dir = os.path.normpath(args.results_dir)

    # Load CSVs
    path_points_iyz = os.path.join(res_dir, "mi_points_Iyz.csv")
    path_points_ixz = os.path.join(res_dir, "mi_points_Ixz.csv")
    path_stats_iyz = os.path.join(res_dir, "mi_stats_Iyz.csv")
    path_stats_ixz = os.path.join(res_dir, "mi_stats_Ixz.csv")

    if not os.path.isfile(path_points_iyz) or not os.path.isfile(path_stats_iyz):
        print("[ERROR] I(Y;Z) CSVs not found in", res_dir)
        return

    df_points_iyz = pd.read_csv(path_points_iyz)
    df_stats_iyz = pd.read_csv(path_stats_iyz)

    df_points_ixz = pd.read_csv(path_points_ixz) if os.path.isfile(path_points_ixz) else pd.DataFrame()
    df_stats_ixz = pd.read_csv(path_stats_ixz) if os.path.isfile(path_stats_ixz) else pd.DataFrame()

    # Apply dataset filter
    if args.dataset is not None:
        df_points_iyz = df_points_iyz[df_points_iyz["dataset"] == args.dataset]
        df_stats_iyz = df_stats_iyz[df_stats_iyz["dataset"] == args.dataset]
        if not df_points_ixz.empty:
            df_points_ixz = df_points_ixz[df_points_ixz["dataset"] == args.dataset]
        if not df_stats_ixz.empty:
            df_stats_ixz = df_stats_ixz[df_stats_ixz["dataset"] == args.dataset]

    # If dataset not specified, just take the first available (for title convenience)
    dataset_used = None
    if not df_points_iyz.empty and "dataset" in df_points_iyz.columns:
        vals = df_points_iyz["dataset"].unique()
        dataset_used = vals[0] if len(vals) > 0 else None

    # Apply family filter
    if args.family is not None:
        df_points_iyz = df_points_iyz[df_points_iyz["family"] == args.family]
        df_stats_iyz = df_stats_iyz[df_stats_iyz["family"] == args.family]
        if not df_points_ixz.empty:
            df_points_ixz = df_points_ixz[df_points_ixz["family"] == args.family]
        if not df_stats_ixz.empty:
            df_stats_ixz = df_stats_ixz[df_stats_ixz["family"] == args.family]

    # ---- Plot I(Y;Z_T) vs T_eval ----
    if not df_stats_iyz.empty:
        plt.figure(figsize=(7,5))
        grouped = df_stats_iyz.sort_values("T_eval").groupby(["family", "split"])
        for (family, split), df in grouped:
            label = f"{family} · {split} (n={int(df['n'].max())})"
            plt.plot(df["T_eval"], df["mean"], label=label)
            plt.fill_between(df["T_eval"], df["mean"]-df["sem"], df["mean"]+df["sem"], alpha=0.2)
        plt.xlabel("T_eval (refinement steps)")
        plt.ylabel("I(Y;Z_T) lower bound (nats, Fano)")
        ttl = "I(Y;Z_T) via Fano"
        if dataset_used is not None:
            ttl = f"{dataset_used} · " + ttl
        plt.title(ttl)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_png = os.path.join(args.outdir, "Iyz_fano_curves.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[INFO] Saved {out_png}")
    else:
        print("[INFO] No I(Y;Z_T) stats to plot.")

    # ---- Plot I(X;Z_T) vs T_eval ----
    if not df_stats_ixz.empty:
        plt.figure(figsize=(7,5))
        grouped = df_stats_ixz.sort_values("T_eval").groupby(["family", "split"])
        for (family, split), df in grouped:
            label = f"{family} · {split} (n={int(df['n'].max())})"
            plt.plot(df["T_eval"], df["mean"], label=label)
            plt.fill_between(df["T_eval"], df["mean"]-df["sem"], df["mean"]+df["sem"], alpha=0.2)
        plt.xlabel("T_eval (refinement steps)")
        plt.ylabel("Ĩσ(X;Z_T) Gaussian proxy (nats)")
        ttl = "I(X;Z_T) proxy"
        if dataset_used is not None:
            ttl = f"{dataset_used} · " + ttl
        plt.title(ttl)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_png = os.path.join(args.outdir, "Ixz_proxy_curves.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[INFO] Saved {out_png}")
    else:
        print("[INFO] No I(X;Z_T) stats to plot.")

    # ---- Scatter: I(X;Z_t) vs I(Y;Z_t) on a given split (default test) ----
    if df_points_ixz.empty:
        print("[INFO] No I(X;Z) points; skipping scatter.")
        print("\n=== DONE plot_mi ===")
        return

    split = args.split
    df_iyz_s = df_points_iyz[df_points_iyz["split"] == split]
    df_ixz_s = df_points_ixz[df_points_ixz["split"] == split]

    if df_iyz_s.empty or df_ixz_s.empty:
        print(f"[INFO] No data for split '{split}' to make scatter plot.")
        print("\n=== DONE plot_mi ===")
        return

    # Merge on run_name + T_eval + family + dataset for robustness
    merge_keys = ["run_name", "family", "dataset", "split", "T_eval"]
    merged = pd.merge(
        df_ixz_s,
        df_iyz_s[merge_keys + ["Iyz_fano"]],
        on=merge_keys,
        how="inner"
    )

    if merged.empty:
        print("[INFO] No overlapping I(X;Z) / I(Y;Z) points after merge; skipping scatter.")
        print("\n=== DONE plot_mi ===")
        return

    plt.figure(figsize=(7,6))
    sc = plt.scatter(
        merged["Ixz_proxy"],
        merged["Iyz_fano"],
        c=merged["T_eval"],
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
    )
    cbar = plt.colorbar(sc, label="T_eval")

    plt.xlabel("I(X;Z_t) [nats] (Gaussian proxy)")
    plt.ylabel("I(Y;Z_t) [nats] (Fano bound)")
    ttl = f"I(X;Z_t) vs I(Y;Z_t) · split={split}"
    if dataset_used is not None:
        ttl = f"{dataset_used} · " + ttl
    plt.title(ttl)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(args.outdir, f"scatter_Ixz_Iyz_{split}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[INFO] Saved {out_png}")

    print("\n=== DONE plot_mi ===")
    print("Figures in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
