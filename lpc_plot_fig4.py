"""
Generate Figure 4: capacity-dependency analysis.

Panel A: Generalisation gap (online vs replay).
Panel B: Representational energy (latent norm distributions).

Usage:
  python lpc_plot_fig4.py runs/cifar10_lat512_s123
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Usage: python plot_paper_figure4.py ./runs/lpc_exp1_s123
run_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs/lpc_exp1_s121"

def load_history(mode):
    path = os.path.join(run_dir, f"history_{mode}.json")
    if not os.path.exists(path): return None
    with open(path, "r") as f: return json.load(f)

def load_latents(mode):
    path = os.path.join(run_dir, f"latents_{mode}.npy")
    if not os.path.exists(path): return None
    return np.load(path)

def make_figure4():
    # 1. Load Data
    h_on = load_history("online")
    h_rep = load_history("replay")
    
    z_on = load_latents("online")
    z_rep = load_latents("replay")

    if not (h_on and h_rep and z_on is not None and z_rep is not None):
        print("Error: Missing history or latent files.")
        return

    # 2. Setup Plot (Nature Style)
    # Width: 180mm (Full page) or 89mm (Single column). 
    # Let's go for a wide 2-panel figure.
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=300)

    # --- PANEL A: GENERALISATION GAP ---
    epochs = [x['epoch'] for x in h_on]
    gap_on = [x['gap_acc'] * 100 for x in h_on] # Convert to %
    gap_rep = [x['gap_acc'] * 100 for x in h_rep]

    # Smooth curves slightly for readability if noisy
    def smooth(y, box_pts=3):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    ax1.plot(epochs, gap_on, label="Online (Baseline)", color="#FF8C00", linestyle="--", linewidth=1.5)
    ax1.plot(epochs, gap_rep, label="Offline Replay", color="#800080", linewidth=2)
    
    ax1.set_title("A. Generalisation Gap", loc='left', fontweight='bold')
    ax1.set_xlabel("Readout Training Epochs")
    ax1.set_ylabel("Gap (Train% - Test%)")
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.2)
    
    # Add annotation if the gap is distinct
    final_gap_diff = gap_on[-1] - gap_rep[-1]
    if final_gap_diff > 0:
        ax1.annotate(f'Δ Gap: {final_gap_diff:.1f}%', 
                     xy=(epochs[-1], gap_rep[-1]), 
                     xytext=(epochs[-1]-5, gap_rep[-1]-2),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))

    # --- PANEL B: COMPRESSION (HISTOGRAM) ---
    # Calculate Norms
    norms_on = np.linalg.norm(z_on, axis=1)
    norms_rep = np.linalg.norm(z_rep, axis=1)

    sns.kdeplot(norms_on, fill=True, color="#FF8C00", alpha=0.3, label="Wake (Noisy)", ax=ax2)
    sns.kdeplot(norms_rep, fill=True, color="#800080", alpha=0.5, label="Sleep (Compressed)", ax=ax2)
    
    # Vertical lines for means
    ax2.axvline(norms_on.mean(), color='#FF8C00', linestyle='--', alpha=0.8)
    ax2.axvline(norms_rep.mean(), color='#800080', linestyle='--', alpha=0.8)

    ax2.set_title("B. Representational Energy", loc='left', fontweight='bold')
    ax2.set_xlabel("Latent Norm ||z||")
    ax2.set_ylabel("Density")
    ax2.legend(frameon=False, loc='upper right')
    ax2.grid(True, alpha=0.2)

    # Final Layout
    plt.tight_layout()
    save_path = os.path.join(run_dir, "fig4_combined.png")
    plt.savefig(save_path, format='png', bbox_inches='tight')
    save_path = os.path.join(run_dir, "fig4_combined.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    save_path = os.path.join(run_dir, "fig4_combined.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    print(f"Success! Saved Figure 4 to {save_path}")
    plt.show()

if __name__ == "__main__":
    make_figure4()