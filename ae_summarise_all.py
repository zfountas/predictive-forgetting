#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Summarize consolidation runs into:
  • Per-dataset T-sweep grid: rows = {acc, gap} on test split, cols = groups (cf0, cf25, cf50, ponder...)
  • Per-dataset learning-curve grid: rows = {val, train}, cols = groups; ref = solid, t0 = dashed
  • Text + CSV summary with the most important stats (mean±sd across seeds)

Usage examples:
  python summarize_all.py --runs_dir runs --recursive
  python summarize_all.py --runs_dir runs --recursive --datasets mnist fashion
"""

import argparse, json, re
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers for run discovery ----------

SEED_RE = re.compile(r"(?:^|[\\/])(?:seed[_-]?(\d+)|[^\\/]*?_s(\d+))(?:$|[\\/])", re.IGNORECASE)

def infer_seed(p: Path):
    m = SEED_RE.search(str(p))
    if not m: return None
    for g in (1,2):
        if m.group(g):
            try: return int(m.group(g))
            except: pass
    return None

def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def get_run_dirs(runs_root: Path, recursive: bool):
    it = runs_root.rglob("*") if recursive else runs_root.iterdir()
    return [p for p in it if p.is_dir() and (p / "args.json").exists()]

def get_dataset_for_run(run_dir: Path):
    args = load_json(run_dir / "args.json")
    if args and "dataset" in args: 
        return str(args["dataset"]).lower()
    # fallback heuristic
    m = re.search(r"(mnist|fashion)", str(run_dir).lower())
    return m.group(1) if m else "unknown"

def group_label(run_dir: Path):
    """A compact condition label (e.g., mnist_fixed_cf25, mnist_ponder_L0p4)."""
    name = run_dir.name
    # strip seed suffix
    name = re.sub(r"_s\d+$", "", name, flags=re.IGNORECASE)
    return name

def canonical_group_family(label: str):
    """Remove dataset and seed to get a comparable 'family' across seeds."""
    # drop seed endings and keep everything except trailing _s123
    return re.sub(r"_s\d+$", "", label, flags=re.IGNORECASE)

# ---------- aggregation utilities ----------

def agg_mean_sd(values):
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr)), int(np.sum(~np.isnan(arr)))

def align_epochs(list_of_series):
    """
    Align by epoch index: intersect all epoch sets to common epochs.
    list_of_series: list of DataFrames (each must have an 'epoch' column).
    Returns:
      (common_epochs_numpy, [aligned_df_or_None, ...]) or None if nothing aligns.
    """
    # keep only non-None with 'epoch' column
    series = [s for s in list_of_series if s is not None and "epoch" in s.columns]
    if not series:
        return None

    common = None
    for s in series:
        ep = np.asarray(s["epoch"].values)
        common = ep if common is None else np.intersect1d(common, ep)

    if common is None or len(common) == 0:
        return None

    aligned = []
    for s in series:
        sub = s[s["epoch"].isin(common)].sort_values("epoch")
        # if this seed is missing some epochs, mark as None
        if len(sub) != len(common):
            aligned.append(None)
        else:
            aligned.append(sub)

    # if every aligned entry became None, bail
    if not any(df is not None for df in aligned):
        return None

    return common, aligned

def stack_metric(aligned_list, col):
    mats = []
    for df in aligned_list:
        if df is None or col not in df.columns: 
            mats.append(None); continue
        mats.append(df[col].to_numpy(dtype=float))
    mats = [m for m in mats if m is not None]
    if not mats: 
        return None
    return np.stack(mats, axis=0)  # [n_seeds, n_steps]

# ---------- load all runs and structure them ----------

def collect_runs(runs_root: Path, recursive: bool):
    info = []
    for rd in get_run_dirs(runs_root, recursive):
        seed = infer_seed(rd) or -1
        dataset = get_dataset_for_run(rd)
        label = group_label(rd)
        info.append({"run_dir": rd, "dataset": dataset, "group_label": label, "seed": seed})
    return pd.DataFrame(info) if info else pd.DataFrame(columns=["run_dir","dataset","group_label","seed"])

# ---------- T-sweep aggregation ----------

def load_tsweep(run_dir: Path, split="test"):
    csvp = run_dir / "phase3_Tsweep.csv"
    if not csvp.exists(): return None
    df = load_csv(csvp)
    if df is None: return None
    # allow both with/without split column
    if "split" in df.columns:
        df = df[df["split"].str.lower() == split.lower()]
    if "T_eval" not in df.columns: return None
    keep = [c for c in ("T_eval","acc","gap","err") if c in df.columns]
    df = df[keep].sort_values("T_eval").reset_index(drop=True)
    return df

def aggregate_tsweep_for_group(df_runs, split="test"):
    # df_runs: rows for one group with different seeds
    seeds_to_df = {}
    for _, r in df_runs.iterrows():
        seeds_to_df[r["seed"]] = load_tsweep(r["run_dir"], split=split)
    # find common T grid
    common_T = None
    for s, d in seeds_to_df.items():
        if d is None: continue
        T = d["T_eval"].values
        common_T = T if common_T is None else np.intersect1d(common_T, T)
    if common_T is None or len(common_T)==0:
        return None

    out = {}
    for metric in ("acc","gap"):
        rows = []
        for s, d in seeds_to_df.items():
            if d is None or metric not in d.columns: continue
            sub = d[d["T_eval"].isin(common_T)].sort_values("T_eval")
            if len(sub)!=len(common_T): continue
            rows.append(sub[metric].to_numpy(dtype=float))
        if not rows: 
            out[metric] = None; continue
        stack = np.stack(rows, axis=0)  # [n_seeds, n_T]
        mu = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0)
        out[metric] = {"T": common_T, "mean": mu, "sd": sd, "n": stack.shape[0]}
    return out

# ---------- learning curves aggregation ----------

CURVE_COLS = dict(
    val_ref="val_acc_ref", val_t0="val_acc_t0",
    train_ref="train_acc_ref", train_t0="train_acc_t0",
    train_loss="train_loss", val_loss="val_loss"
)

def load_phase3_history(run_dir: Path):
    p = run_dir / "phase3_history.json"
    js = load_json(p)
    if not js: return None
    # normalize into DataFrame with expected cols
    df = pd.DataFrame(js)
    # ensure required columns exist (skip if missing)
    needed = {"epoch"} | set(CURVE_COLS.values())
    missing_all = needed - set(df.columns)
    if len(missing_all)==len(needed)-1:  # allow no losses sometimes
        pass
    return df

def aggregate_curves_for_group(df_runs):
    """
    Aggregate phase3_history across seeds for one group.
    Returns dict with keys: 'epoch', 'val_ref', 'val_t0', 'train_ref', 'train_t0', 'train_loss', 'val_loss'
    Each of those (except 'epoch') is a dict {mean, sd, n} or None if missing.
    """
    per_seed = []
    for _, r in df_runs.iterrows():
        per_seed.append(load_phase3_history(r["run_dir"]))

    # Only proceed if at least one valid DataFrame with 'epoch'
    if not any((df is not None and "epoch" in df.columns) for df in per_seed):
        return None

    # Align by common epochs across valid seeds
    valid = [df for df in per_seed if df is not None and "epoch" in df.columns]
    aligned_out = align_epochs(valid)
    if aligned_out is None:
        return None
    epochs, aligned = aligned_out

    out = {"epoch": epochs}
    for name, col in CURVE_COLS.items():
        stack = stack_metric(aligned, col)
        if stack is None:
            out[name] = None
        else:
            out[name] = dict(
                mean=np.nanmean(stack, axis=0),
                sd=np.nanstd(stack, axis=0),
                n=stack.shape[0],
            )
    return out

# ---------- phase summaries for key stats ----------

def load_phase2_summary(run_dir: Path):
    js = load_json(run_dir / "phase2_summary.json")
    if not js: return None
    keep = {k: js.get(k, np.nan) for k in ["train_acc","test_acc","gen_gap","train_err","test_err"]}
    return keep

def load_phase3_summary(run_dir: Path):
    js = load_json(run_dir / "phase3_summary.json")
    if not js: return None
    keep = {k: js.get(k, np.nan) for k in [
        "train_acc_ref","train_acc_t0","test_acc_ref","test_acc_t0",
        "gen_gap_ref","gen_gap_t0","train_err_ref","train_err_t0","test_err_ref","test_err_t0"
    ]}
    return keep

def summarize_key_stats(df_runs):
    rows2 = []
    rows3 = []
    for _, r in df_runs.iterrows():
        p2 = load_phase2_summary(r["run_dir"])
        if p2: rows2.append(p2 | {"seed": r["seed"]})
        p3 = load_phase3_summary(r["run_dir"])
        if p3: rows3.append(p3 | {"seed": r["seed"]})

    out = {}
    if rows2:
        d2 = pd.DataFrame(rows2)
        out["phase2"] = d2.agg(["mean","std"]).to_dict()
    if rows3:
        d3 = pd.DataFrame(rows3)
        # add deltas (ref - t0) if available
        if "test_acc_ref" in d3 and "test_acc_t0" in d3:
            d3["delta_test_acc"] = d3["test_acc_ref"] - d3["test_acc_t0"]
        if "gen_gap_ref" in d3 and "gen_gap_t0" in d3:
            d3["delta_gen_gap"] = d3["gen_gap_ref"] - d3["gen_gap_t0"]
        out["phase3"] = d3.agg(["mean","std"]).to_dict()
    return out

# ---------- plotting ----------

def ensure_cols(ncols):
    return max(1, ncols)

def pretty_group_name(lbl,number_of_seeds=None):
    # mnist_fixed_cf25 -> cf25; mnist_ponder_L0p4 -> ponder L=0.4
    s = lbl
    s = re.sub(r"^mnist[_-]?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^fashion[_-]?", "", s, flags=re.IGNORECASE)
    s = s.replace("_", " ")
    s = s.replace("cf", "cf=")
    s = s.replace("L0p", "L=0.")
    if number_of_seeds is not None:
        s += f" (n={number_of_seeds})"
    return s

def plot_tsweep_grid(dataset, groups_ordered, tsweep_by_group, figs_root: Path, split="test"):
    cols = len(groups_ordered)
    if cols == 0: return None
    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(4.0*cols, 6.0), squeeze=False)
    for ci, g in enumerate(groups_ordered):
        ts = tsweep_by_group.get(g)
        title = pretty_group_name(g, number_of_seeds=len(ts.get("acc", {}).get("T", [])) if ts and ts.get("acc") is not None else None)
        # Row 0: acc
        ax = axes[0, ci]
        if ts and ts.get("acc") is not None:
            T = ts["acc"]["T"]; mu = ts["acc"]["mean"]; sd = ts["acc"]["sd"]
            ax.plot(T, mu, linewidth=2.0, label="mean acc")
            ax.fill_between(T, mu - sd, mu + sd, alpha=0.2, label="±1 sd")
        ax.set_title(title)
        ax.set_xlabel("T_eval"); ax.set_ylabel(f"acc ({split})")
        ax.grid(True, alpha=0.25)

        # Row 1: gap
        ax = axes[1, ci]
        if ts and ts.get("gap") is not None:
            T = ts["gap"]["T"]; mu = ts["gap"]["mean"]; sd = ts["gap"]["sd"]
            ax.plot(T, mu, linewidth=2.0, label="mean gap")
            ax.fill_between(T, mu - sd, mu + sd, alpha=0.2, label="±1 sd")
        ax.set_xlabel("T_eval"); ax.set_ylabel(f"gap ({split})")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{dataset.upper()} — Temperature sweep ({split})", y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])
    out = figs_root / f"{dataset}_tsweep_{split}_grid.png"
    fig.savefig(out, dpi=200)
    out = figs_root / f"{dataset}_tsweep_{split}_grid.svg"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out

def plot_curves_grid(dataset, groups_ordered, curves_by_group, figs_root: Path):
    cols = len(groups_ordered)
    if cols == 0: return None
    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(4.0*cols, 6.0), squeeze=False)
    for ci, g in enumerate(groups_ordered):
        cur = curves_by_group.get(g)
        title = pretty_group_name(g, number_of_seeds=cur["val_ref"]["n"] if cur and cur.get("val_ref") is not None else None)

        # Row 0: validation acc (ref vs t0)
        ax = axes[0, ci]
        if cur and cur.get("val_ref") is not None:
            ep = cur["epoch"]; m = cur["val_ref"]["mean"]; s = cur["val_ref"]["sd"]
            ax.plot(ep, m, linewidth=2.0, label="val ref (solid)")
            ax.fill_between(ep, m - s, m + s, alpha=0.15)
        if cur and cur.get("val_t0") is not None:
            ep = cur["epoch"]; m = cur["val_t0"]["mean"]; s = cur["val_t0"]["sd"]
            ax.plot(ep, m, linestyle="--", linewidth=2.0, label="val t0 (dashed)")
        # also overlay train lightly
        if cur and cur.get("train_ref") is not None:
            ep = cur["epoch"]; m = cur["train_ref"]["mean"]
            ax.plot(ep, m, alpha=0.5, linewidth=1.5, label="train ref", )
        if cur and cur.get("train_t0") is not None:
            ep = cur["epoch"]; m = cur["train_t0"]["mean"]
            ax.plot(ep, m, linestyle="--", alpha=0.5, linewidth=1.5, label="train t0")
        ax.set_title(title)
        ax.set_xlabel("epoch"); ax.set_ylabel("acc")
        ax.grid(True, alpha=0.25)

        # Row 1: (optional) loss curves
        ax = axes[1, ci]
        have_any = False
        if cur and cur.get("val_loss") is not None:
            ep = cur["epoch"]; m = cur["val_loss"]["mean"]; s = cur["val_loss"]["sd"]
            ax.plot(ep, m, linewidth=2.0, label="val loss")
            ax.fill_between(ep, m - s, m + s, alpha=0.15)
            have_any = True
        if cur and cur.get("train_loss") is not None:
            ep = cur["epoch"]; m = cur["train_loss"]["mean"]
            ax.plot(ep, m, alpha=0.6, linewidth=1.5, label="train loss")
            have_any = True
        if not have_any:
            ax.text(0.5, 0.5, "no loss logged", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.grid(True, alpha=0.25)

        if ci==0:
            axes[0, ci].legend(fontsize=9, loc="lower right")
            axes[1, ci].legend(fontsize=9, loc="upper right")

    fig.suptitle(f"{dataset.upper()} — Phase 3 learning curves (ref solid, t0 dashed; val bold, train faint)", y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])
    out = figs_root / f"{dataset}_curves_grid.png"
    fig.savefig(out, dpi=200)
    out = figs_root / f"{dataset}_curves_grid.svg"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out

# ---------- best-T and summary roll-up ----------

def find_best_T(ts_dict, mode="acc"):
    """Return (T*, mean, sd) where T* maximizes acc or minimizes gap."""
    if ts_dict is None or ts_dict.get(mode) is None:
        return None
    T = ts_dict[mode]["T"]; mu = ts_dict[mode]["mean"]; sd = ts_dict[mode]["sd"]
    if mode == "acc":
        idx = int(np.nanargmax(mu))
    else:  # gap
        idx = int(np.nanargmin(mu))
    return float(T[idx]), float(mu[idx]), float(sd[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None, help="Filter to these datasets (e.g., mnist fashion)")
    ap.add_argument("--figs_dir", type=str, default="figs")
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"], help="Split for T-sweep")
    ap.add_argument("--summary_csv", type=str, default="figs/summary_stats.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_dir).resolve()
    figs_root = Path(args.figs_dir).resolve()
    figs_root.mkdir(parents=True, exist_ok=True)

    df = collect_runs(runs_root, args.recursive)
    if df.empty:
        print(f"[WARN] No runs found under: {runs_root}")
        return

    # dataset filter
    datasets = sorted(df["dataset"].unique().tolist())
    if args.datasets:
        datasets = [d for d in datasets if d in set(map(str.lower, args.datasets))]
    if not datasets:
        print("[WARN] No datasets match filter.")
        return

    all_rows = []  # for CSV summary

    for dataset in datasets:
        df_d = df[df["dataset"] == dataset].copy()
        if df_d.empty: 
            continue

        # group by condition family (strip seed)
        families = OrderedDict()
        for g in sorted(df_d["group_label"].unique()):
            fam = canonical_group_family(g)
            families.setdefault(fam, True)

        # Build per-group data slices (gather seeds)
        per_group = {}
        for fam in families.keys():
            sel = df_d[df_d["group_label"].str.startswith(fam)]
            # robust: also accept exact match (if names already seed-less)
            if sel.empty:
                sel = df_d[df_d["group_label"] == fam]
            per_group[fam] = sel.sort_values("seed")

        # T-sweep aggregation per group
        tsweep_by_group = {}
        for fam, runs in per_group.items():
            tsweep_by_group[fam] = aggregate_tsweep_for_group(runs, split=args.split)

        # learning curves aggregation per group
        curves_by_group = {}
        for fam, runs in per_group.items():
            curves_by_group[fam] = aggregate_curves_for_group(runs)

        # plot grids
        ordered_groups = list(per_group.keys())
        ts_fig = plot_tsweep_grid(dataset, ordered_groups, tsweep_by_group, figs_root, split=args.split)
        cr_fig = plot_curves_grid(dataset, ordered_groups, curves_by_group, figs_root)

        if ts_fig: print(f"✓ Saved {ts_fig}")
        if cr_fig: print(f"✓ Saved {cr_fig}")

        # key stats (phase2/phase3) + best T per group
        print(f"\n=== SUMMARY: {dataset.upper()} ===")
        for fam, runs in per_group.items():
            stats = summarize_key_stats(runs)
            ts = tsweep_by_group.get(fam)

            best_acc = find_best_T(ts, mode="acc")
            best_gap = find_best_T(ts, mode="gap")

            # Print concise lines
            print(f"\n  • Group: {fam}")
            if "phase2" in stats:
                p2 = stats["phase2"]
                m_acc, s_acc = p2["test_acc"]["mean"], p2["test_acc"]["std"]
                m_gap, s_gap = p2["gen_gap"]["mean"], p2["gen_gap"]["std"]
                print(f"    Phase2: test_acc={m_acc:.4f}±{s_acc:.4f}, gen_gap={m_gap:.4f}±{s_gap:.4f}")
            else:
                print("    Phase2: (no stats)")

            if "phase3" in stats:
                p3 = stats["phase3"]
                m_ref, s_ref = p3.get("test_acc_ref",{}).get("mean", np.nan), p3.get("test_acc_ref",{}).get("std", np.nan)
                m_t0 , s_t0  = p3.get("test_acc_t0",{}).get("mean", np.nan), p3.get("test_acc_t0",{}).get("std", np.nan)
                d_acc_m = p3.get("delta_test_acc",{}).get("mean", np.nan)
                m_gap_r, s_gap_r = p3.get("gen_gap_ref",{}).get("mean", np.nan), p3.get("gen_gap_ref",{}).get("std", np.nan)
                m_gap_t, s_gap_t = p3.get("gen_gap_t0",{}).get("mean", np.nan), p3.get("gen_gap_t0",{}).get("std", np.nan)
                d_gap_m = p3.get("delta_gen_gap",{}).get("mean", np.nan)
                print(f"    Phase3: test_acc_ref={m_ref:.4f}±{s_ref:.4f}, test_acc_t0={m_t0:.4f}±{s_t0:.4f}, Δacc={d_acc_m:.4f}")
                print(f"            gen_gap_ref={m_gap_r:.4f}±{s_gap_r:.4f}, gen_gap_t0={m_gap_t:.4f}±{s_gap_t:.4f}, Δgap={d_gap_m:.4f}")
            else:
                print("    Phase3: (no stats)")

            if best_acc:
                Ta, ya, sa = best_acc
                print(f"    Best T (max acc): T*={Ta:.4g}, acc={ya:.4f}±{sa:.4f}")
            if best_gap:
                Tg, yg, sg = best_gap
                print(f"    Best T (min gap): T*={Tg:.4g}, gap={yg:.4f}±{sg:.4f}")

            # Collect for CSV
            row = dict(dataset=dataset, group=fam)
            # Phase2
            p2 = stats.get("phase2", {})
            row["p2_test_acc_mean"] = p2.get("test_acc",{}).get("mean", np.nan)
            row["p2_test_acc_sd"]   = p2.get("test_acc",{}).get("std", np.nan)
            row["p2_gap_mean"]      = p2.get("gen_gap",{}).get("mean", np.nan)
            row["p2_gap_sd"]        = p2.get("gen_gap",{}).get("std", np.nan)
            # Phase3
            p3 = stats.get("phase3", {})
            row["p3_test_acc_ref_mean"] = p3.get("test_acc_ref",{}).get("mean", np.nan)
            row["p3_test_acc_ref_sd"]   = p3.get("test_acc_ref",{}).get("std", np.nan)
            row["p3_test_acc_t0_mean"]  = p3.get("test_acc_t0",{}).get("mean", np.nan)
            row["p3_test_acc_t0_sd"]    = p3.get("test_acc_t0",{}).get("std", np.nan)
            row["p3_delta_test_acc_mean"]= p3.get("delta_test_acc",{}).get("mean", np.nan)
            row["p3_gap_ref_mean"]      = p3.get("gen_gap_ref",{}).get("mean", np.nan)
            row["p3_gap_ref_sd"]        = p3.get("gen_gap_ref",{}).get("std", np.nan)
            row["p3_gap_t0_mean"]       = p3.get("gen_gap_t0",{}).get("mean", np.nan)
            row["p3_gap_t0_sd"]         = p3.get("gen_gap_t0",{}).get("std", np.nan)
            row["p3_delta_gap_mean"]    = p3.get("delta_gen_gap",{}).get("mean", np.nan)
            # Best T
            if best_acc:
                row["bestT_acc_T"] = best_acc[0]
                row["bestT_acc_mean"] = best_acc[1]
                row["bestT_acc_sd"] = best_acc[2]
            if best_gap:
                row["bestT_gap_T"] = best_gap[0]
                row["bestT_gap_mean"] = best_gap[1]
                row["bestT_gap_sd"] = best_gap[2]

            all_rows.append(row)

    # Save CSV summary
    if all_rows:
        out_csv = Path(args.summary_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"\n✓ Wrote overall summary CSV: {out_csv}")
    else:
        print("\n[WARN] No summary rows generated. Check your runs and splits.")

if __name__ == "__main__":
    main()
