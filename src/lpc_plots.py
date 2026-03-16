"""
Plot LPC experiment results: generalisation gap, test accuracy, and latent norms.

Usage:
  python src/lpc_plots.py runs/lpc_exp1_s121
"""
import json, os, sys
import matplotlib.pyplot as plt

run_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs/lpc_exp1_s121"

def load_hist(mode):
    return json.load(open(os.path.join(run_dir, f"history_{mode}.json")))

def plot_final_analysis():
    h_on = load_hist("online")
    h_rep = load_hist("replay")
    
    epochs = [x['epoch'] for x in h_on]
    
    # PLOT 1: Generalisation Gap (Acc based to match AE)
    gap_on = [x['gap_acc'] for x in h_on]
    gap_rep = [x['gap_acc'] for x in h_rep]
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, gap_on, label="Online (Baseline)", color="orange", linestyle="--")
    plt.plot(epochs, gap_rep, label="Offline Replay", color="purple")
    plt.title("Generalisation Gap\n(Train Acc - Test Acc)")
    plt.xlabel("Readout Training Epochs")
    plt.ylabel("Gap")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # PLOT 2: Test Accuracy Over Epochs
    test_acc_on = [x['test_acc'] for x in h_on]
    test_acc_rep = [x['test_acc'] for x in h_rep]
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, test_acc_on, label="Online (Baseline)", color="orange", linestyle="--")
    plt.plot(epochs, test_acc_rep, label="Offline Replay", color="purple")
    plt.title("Test Accuracy Over Epochs")
    plt.xlabel("Readout Training Epochs")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # PLOT 3: Latent Energy (Geometric Proof of Compression)
    norm_on = [x['z_norm'] for x in h_on]
    norm_rep = [x['z_norm'] for x in h_rep]
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, norm_on, label="Online Z", color="orange", linestyle="--")
    plt.plot(epochs, norm_rep, label="Replay Z", color="purple")
    plt.title("Latent Energy (Compression)\nAvg Norm ||z||")
    plt.xlabel("Epochs")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "fig4_analysis.png"))
    plt.savefig(os.path.join(run_dir, "fig4_analysis.svg"))
    print("Saved fig4_analysis.png/svg")
    plt.show()

if __name__ == "__main__": plot_final_analysis()