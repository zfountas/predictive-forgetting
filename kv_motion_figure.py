#!/usr/bin/env python3
"""
kv_motion_figure.py

Make a 3-panel figure for a single GSM8K log:
(A) PCA of V cache->processed with arrows (color = relative move)
(B) PCA of ΔV (update vectors)
(C) Mean relative movement of K vs V across refinement steps

Usage examples:
  python kv_motion_figure.py 17 --data_dir sample_data --layer 0 --step -1 --out kv_motion_log17

  python kv_motion_figure.py --log_path sample_data/log_17 --layer 0 --step -1
"""

import argparse
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def pca_2d_torch(X: torch.Tensor):
    """
    PCA to 2D using torch SVD.
    X: [N, D] float tensor on CPU
    Returns: Z [N,2], components [2,D], explained_variance_ratio [2]
    """
    X = X.float()
    X = X - X.mean(dim=0, keepdim=True)
    # SVD: X = U S Vh
    # principal directions are rows of Vh
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    comps = Vh[:2, :]  # [2, D]
    Z = X @ comps.T    # [N,2]
    # explained variance ratio
    # eigenvalues proportional to S^2/(N-1)
    N = X.shape[0]
    eigvals = (S**2) / max(N - 1, 1)
    evr = eigvals[:2] / eigvals.sum().clamp_min(1e-12)
    return Z, comps, evr


def load_log(log_path: Path):
    data = torch.load(log_path, map_location="cpu", weights_only=False)
    if not isinstance(data, (list, tuple)) or len(data) == 0:
        raise ValueError(f"Unexpected log format in {log_path}")
    return data


def get_step_layer(data, step_idx: int, layer_idx: int):
    T = len(data)
    if step_idx < 0:
        step_idx = T + step_idx
    if step_idx < 0 or step_idx >= T:
        raise IndexError(f"step_idx={step_idx} out of range (steps={T})")

    step = data[step_idx]
    if not isinstance(step, (list, tuple)):
        raise ValueError("Expected each step to be a list over layers")

    L = len(step)
    if layer_idx < 0:
        layer_idx = L + layer_idx
    if layer_idx < 0 or layer_idx >= L:
        raise IndexError(f"layer_idx={layer_idx} out of range (layers={L})")

    layer = step[layer_idx]
    return step_idx, layer_idx, layer


def flatten_tokens(x: torch.Tensor, mask: torch.Tensor):
    """
    x: [B, H, S, D] (float16 ok)
    mask: [B, S] (0/1 or bool), selects active token positions
    Returns:
      X: [N, H*D] float32
      idx: token indices selected (list)
    """
    if x.dim() != 4:
        raise ValueError(f"Expected [B,H,S,D], got {tuple(x.shape)}")
    B, H, S, D = x.shape
    if B != 1:
        # still handle, but take first batch
        x = x[:1]
        mask = mask[:1]
        B = 1

    m = mask[0].to(torch.bool)
    idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        raise ValueError("Mask selects zero tokens")

    # [1,H,S,D] -> [H, N, D] -> [N, H*D]
    x_sel = x[0, :, idx, :]          # [H, N, D]
    X = x_sel.permute(1, 0, 2).contiguous().view(idx.numel(), H * D)
    return X.float(), idx


def rel_move(cache: torch.Tensor, update: torch.Tensor, eps: float = 1e-8):
    """
    cache, update: [N, D]
    returns per-row relative movement: ||update|| / (||cache|| + eps)
    """
    num = torch.linalg.norm(update, dim=1)
    den = torch.linalg.norm(cache, dim=1).clamp_min(eps)
    return (num / den).cpu()


def compute_stepwise_moves(data, layer_idx: int):
    """
    For a fixed layer, compute mean relative move for K and V per step.
    Returns lists: steps, mean_move_K, mean_move_V
    """
    meanK, meanV = [], []
    steps = list(range(len(data)))

    for t in steps:
        _, _, layer = get_step_layer(data, t, layer_idx)
        mask = layer.get("mask", None)
        if mask is None:
            raise KeyError("No 'mask' in layer dict")

        k_cache = layer["k_cache"]
        k_upd   = layer["k_update"]
        v_cache = layer["v_cache"]
        v_upd   = layer["v_update"]

        Kc, _ = flatten_tokens(k_cache, mask)
        Ku, _ = flatten_tokens(k_upd, mask)
        Vc, _ = flatten_tokens(v_cache, mask)
        Vu, _ = flatten_tokens(v_upd, mask)

        meanK.append(rel_move(Kc, Ku).mean().item())
        meanV.append(rel_move(Vc, Vu).mean().item())

    return steps, meanK, meanV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_index", nargs="?", type=int, default=None,
                    help="Log index i for file log_i (ignored if --log_path is set)")
    ap.add_argument("--log_path", type=str, default=None,
                    help="Absolute path to a log file (overrides log_index + data_dir)")
    ap.add_argument("--data_dir", type=str,
                    default="./sample_data",
                    help="Directory containing log_{i} files")
    ap.add_argument("--layer", type=int, default=0, help="Layer index")
    ap.add_argument("--step", type=int, default=-1, help="Step index to visualise (default: final step)")
    ap.add_argument("--max_arrows", type=int, default=200, help="Max arrows to draw (subsample if too many tokens)")
    ap.add_argument("--out", type=str, default=None, help="Output prefix (without extension)")
    args = ap.parse_args()

    if args.log_path:
        log_path = Path(args.log_path)
        log_name = log_path.name
    else:
        if args.log_index is None:
            raise SystemExit("Provide log_index or --log_path")
        log_path = Path(args.data_dir) / f"log_{args.log_index}"
        log_name = f"log_{args.log_index}"

    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    data = load_log(log_path)
    step_idx, layer_idx, layer = get_step_layer(data, args.step, args.layer)

    # Extract tensors for the chosen step/layer
    mask = layer.get("mask", None)
    if mask is None:
        raise KeyError("No 'mask' in layer dict")

    v_cache = layer["v_cache"]
    v_upd   = layer["v_update"]
    v_proc  = v_cache + v_upd

    # Flatten token vectors
    Vc, tok_idx = flatten_tokens(v_cache, mask)   # [N, H*D]
    Vp, _       = flatten_tokens(v_proc,  mask)   # [N, H*D]
    dV = Vp - Vc

    rmove = rel_move(Vc, dV)  # [N]

    # Subsample for arrows if too dense
    N = Vc.shape[0]
    if N > args.max_arrows:
        # pick evenly spaced indices (deterministic)
        sel = torch.linspace(0, N - 1, steps=args.max_arrows).long()
    else:
        sel = torch.arange(N)

    # PCA for panel A: fit on concatenated cache+processed for stable basis
    Z_all, _, evr_all = pca_2d_torch(torch.cat([Vc, Vp], dim=0))
    Zc = Z_all[:N]
    Zp = Z_all[N:]

    # PCA for panel B: on delta vectors
    ZdV, _, evr_dv = pca_2d_torch(dV)

    # Panel C: stepwise K vs V mean relative move
    steps, meanK, meanV = compute_stepwise_moves(data, layer_idx=args.layer)

    # Output prefix
    if args.out is None:
        args.out = f"kv_motion_{log_name}_layer{args.layer}_step{step_idx}"
    out_prefix = Path(args.out)

    # --- Plot ---
    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(Zc[:, 0], Zc[:, 1], c=rmove.numpy(), s=16, alpha=0.85)
    ax1.scatter(Zp[:, 0], Zp[:, 1], c=rmove.numpy(), s=10, alpha=0.55)

    # arrows (subsample)
    dx = (Zp[sel, 0] - Zc[sel, 0]).numpy()
    dy = (Zp[sel, 1] - Zc[sel, 1]).numpy()
    ax1.quiver(Zc[sel, 0].numpy(), Zc[sel, 1].numpy(),
               dx, dy, angles='xy', scale_units='xy', scale=1,
               width=0.003, alpha=0.55)

    ax1.set_title(f"(a) V PCA with motion\n{log_name}, step {step_idx}, layer {args.layer}\nEVR: {evr_all[0]:.2f}, {evr_all[1]:.2f}")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("relative move  ||ΔV|| / ||V||")

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(ZdV[:, 0], ZdV[:, 1], c=rmove.numpy(), s=16, alpha=0.85)
    ax2.set_title(f"(b) ΔV PCA (update field)\nEVR: {evr_dv[0]:.2f}, {evr_dv[1]:.2f}")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    # Panel C
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps, meanK, marker="o", linewidth=1.5, label="K mean rel-move")
    ax3.plot(steps, meanV, marker="o", linewidth=1.5, label="V mean rel-move")
    ax3.set_title(f"(c) Stepwise movement (layer {args.layer})")
    ax3.set_xlabel("refinement step")
    ax3.set_ylabel("mean ||Δ|| / ||cache||")
    ax3.legend(frameon=True)

    fig.suptitle("Cache consolidation dynamics: stable Keys, coherent Value rewriting", y=1.02, fontsize=12)
    fig.tight_layout()

    png_path = out_prefix.with_suffix(".png")
    svg_path = out_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
