"""
Compute mutual information proxies from trained AE runs (Figure 2d).

Scans a runs directory and computes:
  - I(Y;Z_T) lower bound via Fano's inequality (from phase3_Tsweep.csv)
  - I(X;Z_T) Gaussian diagonal-channel proxy (from exported repr/*.npy)

Alternative estimators (PCA-whitened fixed-basis proxy, joint Gaussian MI)
were also explored; the diagonal proxy is used for the reported figures.

Outputs:
  - mi_points_Iyz.csv, mi_points_Ixz.csv (per-run, per-T_eval)
  - mi_stats_Iyz.csv, mi_stats_Ixz.csv (aggregated across seeds)

Usage:
  python src/ae_compute_mi.py --runs-dir runs --outdir mi_results

After that, use ae_plot_mi_from_csv.py to draw figures.
"""

import argparse, os, json, glob, math, warnings
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------- Helpers ----------------

def is_run_dir(d):
    """A 'run' has args.json and phase3_Tsweep.csv."""
    return (
        os.path.isdir(d)
        and os.path.isfile(os.path.join(d, "args.json"))
        and os.path.isfile(os.path.join(d, "phase3_Tsweep.csv"))
    )

def find_run_dirs(runs_root):
    """Return all immediate subdirectories of runs_root that look like runs."""
    runs_root = os.path.normpath(runs_root)
    if not os.path.isdir(runs_root):
        raise ValueError(f"runs_root does not exist or is not a directory: {runs_root}")

    run_dirs = []
    for name in os.listdir(runs_root):
        d = os.path.join(runs_root, name)
        if is_run_dir(d):
            run_dirs.append(d)
    run_dirs.sort()
    return run_dirs

def parse_family_and_seed(run_dir):
    """
    Derive family + seed from folder name.
    Expects something like ".../cifar10_..._cf50_s120"
    -> family="cifar10_..._cf50", seed="120"
    If no "_s" is present, treat as single-run family.
    """
    base = os.path.basename(os.path.normpath(run_dir))
    if "_s" in base:
        prefix, seed_str = base.rsplit("_s", 1)
        family = prefix
        seed = seed_str
    else:
        family = base
        seed = None
    return family, seed, base

def load_args(run_dir):
    with open(os.path.join(run_dir, "args.json"), "r") as f:
        return json.load(f)

def infer_dataset_and_classes(args_json):
    # default to 10 classes for MNIST/FashionMNIST/CIFAR-10 unless specified
    dataset = args_json.get("dataset", "unknown")
    dataset_l = dataset.lower()
    if dataset_l in ["mnist", "fashion", "fashionmnist", "fashion_mnist", "cifar10", "cifar-10", "svhn", "emnist"]:
        k = 10 if dataset_l != "emnist" else 47
    else:
        k = int(args_json.get("num_classes", 10))
    return dataset, k

def fano_lower_bound_Iyz(err, K):
    """
    Fano's inequality (nats):
      H(Y|Z) <= h(err) + err*ln(K-1)
      I(Y;Z) >= H(Y) - H(Y|Z)
    We take H(Y) ~ ln K (balanced).
    """
    err = np.clip(err, 1e-9, 1-1e-9)
    Hy = math.log(K)
    h_e = -(err*np.log(err) + (1-err)*np.log(1-err))
    return Hy - (h_e + err*math.log(max(K-1,1)))

def load_tsweep_rows(run_dir, splits, K, meta):
    """
    Read phase3_Tsweep.csv and return a list of dicts:
    {run_name, family, seed, dataset, split, T_eval, err, Iyz_fano, ...}
    """
    ts_path = os.path.join(run_dir, "phase3_Tsweep.csv")
    df = pd.read_csv(ts_path)
    rows = []
    for sp in splits:
        sub = df[df["split"] == sp]
        if sub.empty:
            continue
        if "err" not in sub.columns or "T_eval" not in sub.columns:
            continue
        for _, r in sub.iterrows():
            te = int(r["T_eval"])
            err = float(r["err"])
            rows.append({
                "run_dir": meta["run_dir"],
                "run_name": meta["run_name"],
                "family": meta["family"],
                "seed": meta["seed"],
                "dataset": meta["dataset"],
                "num_classes": K,
                "split": sp,
                "T_eval": te,
                "err": err,
                "Iyz_fano": fano_lower_bound_Iyz(err, K),
            })
    return rows

def load_repr_blocks(run_dir, split):
    """
    Look for representation blocks under:
      run_dir/repr/<split>/Teval_*/*  (expects zT.npy)
    Returns dict: {Teval: np.ndarray [N, D]}.
    """
    base = os.path.join(run_dir, "repr", split)
    if not os.path.isdir(base):
        return {}
    out = {}
    te_dirs = sorted(glob.glob(os.path.join(base, "Teval_*")))
    for td in te_dirs:
        name = os.path.basename(td)
        if not name.startswith("Teval_"):
            continue
        try:
            te = int(name.split("_")[1])
        except Exception:
            continue
        zt_path = os.path.join(td, "zT.npy")
        if os.path.isfile(zt_path):
            try:
                zt = np.load(zt_path)
                if zt.ndim == 2:
                    out[te] = zt
            except Exception as e:
                warnings.warn(f"[{run_dir}] Failed to load {zt_path}: {e}")
    return out

def gaussian_proxy_Ixz(Z, sigma=0.1, eps=1e-6):
    """
    Gaussian diagonal-channel proxy for I(X;Z) with additive noise sigma:
      I ≈ 0.5 * sum_i log(1 + Var(Z_i) / sigma^2)
    where Var(Z_i) is the empirical variance across samples of dimension i.

    Z: [N, D]
    Returns a scalar (nats).
    """
    Zc  = Z - Z.mean(axis=0, keepdims=True)
    var = Zc.var(axis=0, ddof=1) + eps   # [D]
    return 0.5 * float(np.sum(np.log(1.0 + var / (sigma**2))))

def load_ixz_rows(run_dir, splits, sigma, meta):
    """
    From repr/<split>/Teval_k/zT.npy, compute I(X;Z_T) per run / split / T_eval.
    Returns a list of dicts with metadata + Ixz_proxy.

    IMPORTANT CHANGE: we no longer whiten per T_eval.
    We directly use the empirical variance of Z_t under a fixed sigma,
    so differences in compression across refinement steps are preserved.
    """
    rows = []
    for sp in splits:
        blocks = load_repr_blocks(run_dir, sp)
        if not blocks:
            continue
        for te, Z in blocks.items():
            ixz = gaussian_proxy_Ixz(Z, sigma=sigma)
            rows.append({
                "run_dir": meta["run_dir"],
                "run_name": meta["run_name"],
                "family": meta["family"],
                "seed": meta["seed"],
                "dataset": meta["dataset"],
                "split": sp,
                "T_eval": te,
                "Ixz_proxy": ixz,
                "sigma": sigma,
                "N_samples": int(Z.shape[0]),
                "D_latent": int(Z.shape[1]),
            })
    return rows

def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs",
                    help="Root directory containing run subfolders.")
    ap.add_argument("--splits", type=str, default="train,val,test",
                    help="Comma list among train,val,test")
    ap.add_argument("--sigma", type=float, default=0.1,
                    help="Eval noise std used for I(X;Z) proxy.")
    ap.add_argument("--outdir", type=str, default="mi_results",
                    help="Where to save CSVs.")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    safe_mkdir(args.outdir)

    run_dirs = find_run_dirs(args.runs_dir)
    if not run_dirs:
        print(f"[WARN] No run dirs found under: {args.runs_dir}")
        return

    print("[INFO] Found runs:")
    for r in run_dirs:
        print("  -", r)

    # Collect raw per-run datapoints
    iyz_rows = []
    ixz_rows = []

    for run_dir in run_dirs:
        family, seed, run_name = parse_family_and_seed(run_dir)
        args_json = load_args(run_dir)
        dataset, K = infer_dataset_and_classes(args_json)

        meta = {
            "run_dir": run_dir,
            "run_name": run_name,
            "family": family,
            "seed": seed,
            "dataset": dataset,
        }

        # I(Y;Z_T)
        iyz_rows.extend(load_tsweep_rows(run_dir, splits, K, meta))

        # I(X;Z_T)
        ixz_rows.extend(load_ixz_rows(run_dir, splits, args.sigma, meta))

    # Convert to DataFrames
    df_iyz = pd.DataFrame(iyz_rows) if iyz_rows else pd.DataFrame()
    df_ixz = pd.DataFrame(ixz_rows) if ixz_rows else pd.DataFrame()

    # Save raw datapoints
    if not df_iyz.empty:
        out_iyz = os.path.join(args.outdir, "mi_points_Iyz.csv")
        df_iyz.to_csv(out_iyz, index=False)
        print(f"[INFO] Saved raw I(Y;Z_T) points to: {out_iyz}")
    else:
        print("[INFO] No I(Y;Z_T) datapoints found (missing Tsweep?).")

    if not df_ixz.empty:
        out_ixz = os.path.join(args.outdir, "mi_points_Ixz.csv")
        df_ixz.to_csv(out_ixz, index=False)
        print(f"[INFO] Saved raw I(X;Z_T) points to: {out_ixz}")
    else:
        print("[INFO] No I(X;Z_T) datapoints found (no repr/ folders?).")

    # Aggregated stats across runs in the same family
    if not df_iyz.empty:
        gcols = ["dataset", "family", "split", "T_eval"]
        g = df_iyz.groupby(gcols)["Iyz_fano"]
        stats_iyz = g.agg(mean="mean", std="std", n="count").reset_index()
        stats_iyz["sem"] = stats_iyz["std"] / np.sqrt(stats_iyz["n"].clip(lower=1))
        out_stats_iyz = os.path.join(args.outdir, "mi_stats_Iyz.csv")
        stats_iyz.to_csv(out_stats_iyz, index=False)
        print(f"[INFO] Saved I(Y;Z_T) stats to: {out_stats_iyz}")

    if not df_ixz.empty:
        gcols = ["dataset", "family", "split", "T_eval"]
        g = df_ixz.groupby(gcols)["Ixz_proxy"]
        stats_ixz = g.agg(mean="mean", std="std", n="count").reset_index()
        stats_ixz["sem"] = stats_ixz["std"] / np.sqrt(stats_ixz["n"].clip(lower=1))
        out_stats_ixz = os.path.join(args.outdir, "mi_stats_Ixz.csv")
        stats_ixz.to_csv(out_stats_ixz, index=False)
        print(f"[INFO] Saved I(X;Z_T) stats to: {out_stats_ixz}")

    print("\n=== DONE compute_mi ===")
    print("Results in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
