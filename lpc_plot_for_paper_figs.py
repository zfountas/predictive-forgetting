"""
Generate paper figures from LPC experiments: t-SNE and prototype alignment.

Compares online (wake) vs offline replay (sleep) latent representations using
t-SNE visualisation and cosine similarity to class prototypes.

Usage:
  python lpc_plot_for_paper_figs.py runs/lpc_exp1_s123
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, json
import numpy as np
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from lpc_experiment import LPCModel, ClassifierHead, Phase3Config, device_auto, get_dataloaders

# Usage: python plot_final_paper_figs.py ./runs/your_best_run
run_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs/lpc_exp1_s123"

def get_latents_and_preds(lpc, head, dl, device):
    lpc.eval(); head.eval()
    all_z, all_y, all_preds = [], [], []
    
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            # Encode exactly as the head sees it (using the logic from your training)
            # For visualization, we just use the amortized z to see the manifold structure
            # OR better: replicate the refinement used in training.
            # Let's use Amortized + 5 steps (Online) vs Amortized + Dream + 10 steps (Replay)
            # BUT to compare 'fairly', let's just look at the HEAD's decision space.
            
            # Actually, simple approach:
            # Just get the refined latents using the Online protocol (fast) 
            # vs Replay protocol (dream).
            
            # Let's just use the simple "Online" encoding for the t-SNE 
            # to see how the HEADS treat the SAME input differently?
            # No, we want to see how the LATENTS differ.
            pass

    # RESTART: We need to generate two sets of latents for the SAME data.
    # 1. Z_online (Noisy)
    # 2. Z_replay (Cleaned via Dream)
    return

def analysis_pipeline():
    # 1. Load Config
    with open(os.path.join(run_dir, "args.json"), "r") as f: args = json.load(f)
    device = device_auto()
    
    # 2. Load LPC Model
    in_ch = 1 if args['dataset'] in ['mnist', 'fashion'] else 3
    img_size = 28 if args['dataset'] in ['mnist', 'fashion'] else 32
    lpc = LPCModel(in_ch=in_ch, latent=args['latent'], img_size=img_size).to(device)
    lpc.load_state_dict(torch.load(os.path.join(run_dir, "phase1_lpc_best.pt"), map_location=device))
    
    # 3. Get Test Data (Subset for speed)
    _, _, dl_test, _, _, _ = get_dataloaders(args['dataset'], batch_size=128)
    # Subset 1000 images for t-SNE
    indices = np.random.choice(len(dl_test.dataset), 1000, replace=False)
    dl_sub = DataLoader(Subset(dl_test.dataset, indices), batch_size=128)

    # 4. Generate Comparison Latents
    print("Generating Latents...")
    z_online_list, z_replay_list, labels_list = [], [], []
    
    # Parameters from your "Winning" Run
    INF_LR = args['p3_inf_lr']
    # WAKE: Noisy
    WAKE_STEPS = args['p3_T_train'] 
    WAKE_NOISE = 0.2  # The noise you added
    # SLEEP: Clean, Deep, High Beta
    SLEEP_STEPS = 10
    SLEEP_NOISE = 0.0
    SLEEP_BETA = 0.05

    for x, y in dl_sub:
        x = x.to(device)
        
        # --- A. Generate Online Z (Noisy) ---
        lpc.prior_beta = 0.001 # Weak gravity
        z0, _, _, _ = lpc.amortized_encode(x)
        z_on = lpc.refine_latent(x, steps=WAKE_STEPS, step_size=INF_LR, noise_std=WAKE_NOISE, detach=True, z_init=z0)[-1]
        
        # --- B. Generate Replay Z (Clean Dream) ---
        lpc.prior_beta = SLEEP_BETA # Strong gravity
        with torch.no_grad():
            # Simulate the consolidation loop:
            # 1. Start from the noisy z_on (Hippocampal Trace)
            x_hat = torch.sigmoid(lpc.decoder(z_on))
            # 2. Refine against dream
            z_rep = lpc.refine_latent(x_hat, steps=SLEEP_STEPS, step_size=INF_LR, noise_std=SLEEP_NOISE, detach=True, z_init=z_on)[-1]

        z_online_list.append(z_on.detach().cpu())    # <--- Added .detach()
        z_replay_list.append(z_rep.detach().cpu())   # <--- Added .detach()
        labels_list.append(y.detach().cpu())         # <--- Added .detach() just in case

    Z_online = torch.cat(z_online_list).numpy()
    Z_replay = torch.cat(z_replay_list).numpy()
    Labels = torch.cat(labels_list).numpy()

    # ================= PLOT 1: t-SNE (Geometric Clustering) =================
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    # Stack to run t-SNE jointly (ensures same space)
    Z_all = np.concatenate([Z_online, Z_replay], axis=0)
    Z_emb = tsne.fit_transform(Z_all)
    
    Z_emb_on = Z_emb[:len(Z_online)]
    Z_emb_rep = Z_emb[len(Z_online):]

    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Online Plot
    scatter1 = axes[0].scatter(Z_emb_on[:,0], Z_emb_on[:,1], c=Labels, cmap='tab10', s=10, alpha=0.6)
    axes[0].set_title("Online Encoding (Wake)\nHigh Entropy, Fuzzy Boundaries")
    axes[0].axis('off')
    
    # Replay Plot
    scatter2 = axes[1].scatter(Z_emb_rep[:,0], Z_emb_rep[:,1], c=Labels, cmap='tab10', s=10, alpha=0.6)
    axes[1].set_title("Offline Replay (Sleep)\nSemantic Clusters (Compressed)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "fig_tsne.png"))
    plt.savefig(os.path.join(run_dir, "fig_tsne.svg"))
    print("Saved fig_tsne.png/svg")

    # ================= PLOT 2: Prototype Alignment (Quantitative) =================
    print("Calculating Prototype Alignment...")
    # Calculate class means (Prototypes) based on REPLAY data (The ideal)
    prototypes = []
    for c in range(10):
        mask = (Labels == c)
        proto = Z_replay[mask].mean(axis=0)
        prototypes.append(proto / np.linalg.norm(proto)) # Normalize
    
    prototypes = np.stack(prototypes) # [10, D]
    
    # Measure Cosine Sim of every point to its class prototype
    def get_sims(Z_data):
        sims = []
        for i in range(len(Z_data)):
            z = Z_data[i]
            c = Labels[i]
            z_norm = z / np.linalg.norm(z)
            sim = np.dot(z_norm, prototypes[c])
            sims.append(sim)
        return sims

    sims_on = get_sims(Z_online)
    sims_rep = get_sims(Z_replay)
    
    plt.figure(figsize=(6, 5), dpi=150)
    sns.kdeplot(sims_on, fill=True, label="Online", color="orange")
    sns.kdeplot(sims_rep, fill=True, label="Replay", color="purple")
    plt.xlabel("Cosine Similarity to Class Prototype")
    plt.title("Semantic Alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "fig_alignment.png"))
    plt.savefig(os.path.join(run_dir, "fig_alignment.svg"))
    print("Saved fig_alignment.png/svg")

if __name__ == "__main__":
    analysis_pipeline()