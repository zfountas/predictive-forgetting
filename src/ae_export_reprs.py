"""
Export latent representations z_T for each refinement step T_eval.

Saves numpy arrays to: runs/<run>/repr/<split>/Teval_<k>/zT.npy
Required as input for ae_compute_mi.py.

Usage:
  python src/ae_export_reprs.py runs/mnist_s123
  python src/ae_export_reprs.py runs --replace-existing

By default, only runs without a repr/ subfolder are processed.
Pass --replace-existing to re-export.
"""

import argparse, os, glob, json
import numpy as np
import torch
from tqdm import tqdm

from api_adapter import get_dataloaders, load_models_from_run, max_refinement_T

def is_run_dir(d):
    return (
        os.path.isfile(os.path.join(d, "args.json"))
        and os.path.isfile(os.path.join(d, "phase3_summary.json"))
    )

def find_runs(path, replace_existing=False):
    """
    Return run directories under `path`.

    If replace_existing is False (default), only return runs that
    do NOT already have a `repr/` subfolder.
    """
    path = os.path.normpath(path)

    # Case 1: path itself is a run dir
    if os.path.isdir(path) and is_run_dir(path):
        if (not replace_existing) and os.path.isdir(os.path.join(path, "repr")):
            # repr exists -> skip this run unless we replace
            return []
        return [path]

    # Case 2: path is a prefix / parent (e.g., "runs/mnist_fixed_cf50")
    candidates = [p for p in glob.glob(os.path.join(path, "*")) if os.path.isdir(p)]
    runs = [p for p in candidates if is_run_dir(p)]

    if not replace_existing:
        # Filter out runs that already have repr/
        runs = [r for r in runs if not os.path.isdir(os.path.join(r, "repr"))]

    return sorted(runs)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

@torch.no_grad()
def collect_split(encode_fn, refine_fn, loader, device, T_eval):
    zs = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        z0 = encode_fn(xb)
        if isinstance(z0, tuple):  # e.g., (mu, logvar)
            z0 = z0[0]
        zT = refine_fn(z0, T_eval=T_eval) if T_eval > 0 else z0
        zs.append(zT.detach().cpu().numpy())
    return np.concatenate(zs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Run dir or run-family prefix")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tmax", type=int, default=None,
                    help="Override max T_eval (default: from args.json or Tsweep)")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--replace-existing", action="store_true",
                    help="Replace existing repr/ folders (default: skip them entirely)")
    args = ap.parse_args()

    # ✅ Only collect runs that match the "no repr/" condition (unless replacing)
    runs = find_runs(args.path, replace_existing=args.replace_existing)
    if not runs:
        if args.replace_existing:
            print(f"[WARN] No valid runs under: {args.path}")
        else:
            print(f"[WARN] No runs without repr/ under: {args.path} "
                  f"(use --replace-existing to include them).")
        return

    print("Found runs:")
    for r in runs:
        print("  ", r)

    for run_dir in runs:
        print(f"\n=== Exporting reps for: {run_dir} ===")

        # repr/ existence is already filtered in find_runs() when not replacing,
        # but keep this as a safety check.
        repr_dir = os.path.join(run_dir, "repr")
        if (not args.replace_existing) and os.path.isdir(repr_dir):
            print("  [SKIP] repr/ already exists (shouldn't happen if find_runs filtered correctly).")
            continue

        # read dataset name for dataloaders
        with open(os.path.join(run_dir, "args.json"), "r") as f:
            a = json.load(f)
        dataset = a.get("dataset", "mnist")
        batch_size = int(a.get("batch_size", args.batch))
        device = torch.device(args.device)

        encode_fn, refine_fn, meta = load_models_from_run(run_dir, device)
        loaders = get_dataloaders(dataset, batch_size=batch_size)

        T_max = args.tmax if args.tmax is not None else max_refinement_T(run_dir)
        T_max = int(T_max)

        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            if split not in loaders:
                print(f"  [skip] No loader for split '{split}'")
                continue
            for k in range(T_max + 1):
                out_dir = os.path.join(run_dir, "repr", split, f"Teval_{k}")
                ensure_dir(out_dir)
                out_path = os.path.join(out_dir, "zT.npy")
                if os.path.isfile(out_path):
                    print(f"  [{split}] Teval={k}: exists, skipping")
                    continue
                print(f"  [{split}] Teval={k}: computing…")
                zT = collect_split(encode_fn, refine_fn, loaders[split], device, T_eval=k)
                np.save(out_path, zT)
        print("  ✓ Done.")

if __name__ == "__main__":
    main()
