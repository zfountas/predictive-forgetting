"""
Generate online vs replay latent representations for Figure 4 plotting.

Saves latents_online.npy and latents_replay.npy in the run directory,
which are then used by lpc_plot_fig4.py.

Usage:
  python src/lpc_generate_latents_for_plotting.py runs/cifar10_lat512_s123
"""
import torch
import os, sys, json
import numpy as np
from torch.utils.data import DataLoader
from lpc_experiment import LPCModel, device_auto, get_dataloaders

# Usage: python src/lpc_generate_latents_for_plotting.py ./runs/lpc_exp1_s121
run_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs/lpc_exp1_s121"

def generate_latents():
    print(f"--- Regenerating Latents for {run_dir} ---")
    
    # 1. Load Config & Model
    config_path = os.path.join(run_dir, "args.json")
    if not os.path.exists(config_path):
        print("Error: args.json not found.")
        return

    with open(config_path, "r") as f: args = json.load(f)
    device = device_auto()
    
    in_ch = 1 if args['dataset'] in ['mnist', 'fashion', 'emnist'] else 3
    img_size = 28 if args['dataset'] in ['mnist', 'fashion', 'emnist'] else 32
    lpc = LPCModel(in_ch=in_ch, latent=args['latent'], img_size=img_size).to(device)
    
    ckpt = os.path.join(run_dir, "phase1_lpc_best.pt")
    if not os.path.exists(ckpt):
        print("Error: Phase 1 checkpoint not found.")
        return
        
    lpc.load_state_dict(torch.load(ckpt, map_location=device))
    lpc.eval()
    
    # 2. Get Data (Use Test Set for clean validation of the mechanism)
    _, _, dl_test, _, _, _ = get_dataloaders(args['dataset'], batch_size=128)
    
    # 3. Define Protocols (MUST MATCH YOUR "WINNING" CONFIG)
    # Wake Settings
    WAKE_STEPS = args.get('p3_T_wake', 5)
    WAKE_NOISE = 0.2
    WAKE_BETA = 0.001
    
    # Sleep Settings
    SLEEP_STEPS = 10
    SLEEP_NOISE = 0.05
    SLEEP_BETA = 5.0
    
    online_latents = []
    replay_latents = []
    
    print("Processing batches...")
    for i, (x, y) in enumerate(dl_test):
        x = x.to(device)
        
        # --- Generate ONLINE (Wake) ---
        lpc.prior_beta = WAKE_BETA
        z0, _, _, _ = lpc.amortized_encode(x)
        z_on = lpc.refine_latent(
            x, steps=WAKE_STEPS, step_size=0.1, 
            noise_std=WAKE_NOISE, detach=True, z_init=z0
        )[-1]
        online_latents.append(z_on.cpu().numpy())
        
        # --- Generate REPLAY (Sleep) ---
        # Start from the noisy z_on (simulating hippocampal trace)
        lpc.prior_beta = SLEEP_BETA
        with torch.no_grad():
            x_hat = torch.sigmoid(lpc.decoder(z_on)) # Dream
            
            z_rep = lpc.refine_latent(
                x_hat, steps=SLEEP_STEPS, step_size=0.2, 
                noise_std=SLEEP_NOISE, detach=True, z_init=z_on
            )[-1]
        replay_latents.append(z_rep.cpu().numpy())
        
        if i % 10 == 0: print(f"Batch {i}/{len(dl_test)}")

    # 4. Save Files
    Z_on = np.concatenate(online_latents, axis=0)
    Z_rep = np.concatenate(replay_latents, axis=0)
    
    np.save(os.path.join(run_dir, "latents_online.npy"), Z_on)
    np.save(os.path.join(run_dir, "latents_replay.npy"), Z_rep)
    
    print(f"Saved latents_online.npy ({Z_on.shape})")
    print(f"Saved latents_replay.npy ({Z_rep.shape})")
    
    # Quick check
    diff = np.linalg.norm(Z_on, axis=1).mean() - np.linalg.norm(Z_rep, axis=1).mean()
    print(f"Compression Check (Wake Norm - Sleep Norm): {diff:.4f}")
    if diff > 0: print("SUCCESS: Compression verified.")
    else: print("WARNING: No compression detected.")

if __name__ == "__main__":
    generate_latents()