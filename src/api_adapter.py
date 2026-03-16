"""
Adapter to load trained AE models and generate latent representations.

Bridges ae_export_reprs.py to ae_experiment.py classes and checkpoints.
Provides encode_fn, refine_fn, and metadata from a saved run directory.
"""

import os, json
import torch

# ---------- DATA ----------
def get_dataloaders(dataset_name: str, batch_size: int = 256):
    """
    Delegates to ae_experiment.get_dataloaders so splits match your training.
    Returns dict(train/val/test) of DataLoaders.
    """
    import ae_experiment  # must be importable from current working dir / PYTHONPATH
    dl_train, dl_val, dl_test, in_ch, n_classes, img_size = ae_experiment.get_dataloaders(
        dataset=dataset_name, batch_size=batch_size
    )
    return {
        "train": dl_train,
        "val":   dl_val,
        "test":  dl_test,
    }

# ---------- MODELS ----------
def load_models_from_run(run_dir: str, device: torch.device):
    """
    Returns:
      encode_fn(x: FloatTensor[B,C,H,W]) -> z0: FloatTensor[B,D]
      refine_fn(z0: FloatTensor[B,D], T_eval: int) -> zT: FloatTensor[B,D]
      meta: dict (e.g., {"latent": D, "mode": "...", "dataset": "...", "in_ch": C, "img_size": H})
    Uses your ae_experiment.py classes and the checkpoints in run_dir.
    """
    import ae_experiment

    # --- read args ---
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path, "r") as f:
        args = json.load(f)

    latent     = int(args.get("latent", 32))
    ref_hidden = int(args.get("p3_ref_hidden", 256))
    mode       = args.get("mode", "fixed")
    dataset    = str(args.get("dataset", "mnist"))
    vae_flag   = bool(args.get("vae", False))

    # --- ask ae_experiment for canonical in_ch and img_size ---
    # This guarantees we reconstruct *exactly* the architecture used in training
    _, _, _, in_ch, n_classes, img_size = ae_experiment.get_dataloaders(
        dataset=dataset,
        batch_size=1,          # small, we only care about shapes
    )

    print(f"[api_adapter] Rebuilding AE for run {run_dir}")
    print(f"  dataset={dataset}, in_ch={in_ch}, img_size={img_size}, latent={latent}, mode={mode}, vae={vae_flag}")

    # --- rebuild AE and load phase1 encoder ---
    ae = ae_experiment.Autoencoder(
        in_ch=in_ch,
        latent=latent,
        vae=vae_flag,
        img_size=img_size,
    ).to(device)

    p1_ckpt_path = os.path.join(run_dir, "phase1_best.pt")
    state = torch.load(p1_ckpt_path, map_location=device)
    ae.load_state_dict(state)   # should now match exactly
    ae.eval()
    encoder = ae.encoder
    encoder.eval()

    # --- rebuild refiner and load Phase3 refiner weights ---
    if mode == "fixed":
        refiner = ae_experiment.RefinerFixed(dim=latent, hidden=ref_hidden).to(device)
    elif mode == "ponder":
        refiner = ae_experiment.RefinerPonder(dim=latent, hidden=ref_hidden).to(device)
    else:
        raise ValueError(f"Unknown mode in args.json: {mode}")

    p3_ref = os.path.join(run_dir, "phase3_refiner_best.pt")
    if not os.path.isfile(p3_ref):
        raise FileNotFoundError(f"Missing {p3_ref} in {run_dir}")
    ref_ckpt = torch.load(p3_ref, map_location=device)
    refiner.load_state_dict(ref_ckpt)
    refiner.eval()

    @torch.no_grad()
    def encode_fn(x):
        # deterministic encoding path used for head/refiner
        return encoder.encode_deterministic(x)

    @torch.no_grad()
    def refine_fn(z0, T_eval: int):
        """
        For fixed: returns z_T after T_eval steps.
        For ponder: returns expected representation over steps 1..T_eval
                    (renormalised mixture), and z0 when T_eval==0.
        """
        if mode == "fixed":
            zs = refiner(z0, T=T_eval)           # list length T_eval+1
            return zs[-1]
        else:
            zs, q = refiner(z0, T=T_eval if T_eval > 0 else 1)
            if T_eval == 0:
                return zs[0]
            probs = q[:, :T_eval]
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B,T_eval]
            z_stack = torch.stack(zs[1:T_eval+1], dim=1)                    # [B,T_eval,D]
            z_mix = (probs.unsqueeze(-1) * z_stack).sum(dim=1)              # [B,D]
            return z_mix

    meta = {
        "latent": latent,
        "mode": mode,
        "dataset": dataset,
        "in_ch": in_ch,
        "img_size": img_size,
    }
    return encode_fn, refine_fn, meta



# ---------- MAX T ----------
def max_refinement_T(run_dir: str) -> int:
    """
    Prefer T from args.json; fallback to max T_eval in phase3_Tsweep.csv; default 3.
    """
    args_path = os.path.join(run_dir, "args.json")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            args = json.load(f)
        if "T" in args:
            try:
                return int(args["T"])
            except Exception:
                pass

    ts_csv = os.path.join(run_dir, "phase3_Tsweep.csv")
    if os.path.isfile(ts_csv):
        import pandas as pd
        df = pd.read_csv(ts_csv)
        if "T_eval" in df.columns:
            try:
                return int(df["T_eval"].max())
            except Exception:
                pass

    return 3
