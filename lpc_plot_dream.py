"""
Visualise wake vs sleep generative replay (Figure 3c).

Shows real inputs (wake), dream reconstructions (sleep), and the difference
(forgotten noise) for qualitative assessment of consolidation.

Usage:
  python lpc_plot_dream.py runs/fashion_lpc_s123
"""
import torch
import matplotlib.pyplot as plt
import os, sys, json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lpc_experiment import LPCModel, device_auto

# Usage: python plot_dreams.py ./runs/fashion_debug
run_dir = sys.argv[1] if len(sys.argv) > 1 else "./runs/fashion_debug"

def plot_generative_replay():
    # 1. Load Config & Model
    config_path = os.path.join(run_dir, "args.json")
    if not os.path.exists(config_path):
        print(f"Error: args.json not found in {run_dir}")
        return
        
    with open(config_path, "r") as f:
        args = json.load(f)

    device = device_auto()
    img_size = 28 if args['dataset'] in ['mnist', 'fashion', 'emnist'] else 32
    in_ch = 1 if args['dataset'] in ['mnist', 'fashion', 'emnist'] else 3
    
    model = LPCModel(in_ch=in_ch, latent=args['latent'], img_size=img_size).to(device)
    ckpt = os.path.join(run_dir, "phase1_lpc_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # 2. Get Real Data
    transform = transforms.ToTensor()
    if args['dataset'] == 'fashion':
        ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    elif args['dataset'] == 'cifar10':
        ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    else:
        ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    x_real, _ = next(iter(loader))
    x_real = x_real.to(device)

    # 3. Generate "Dreams" (The Inverse Step)
    # We first encode to z_fast (Online Perception)
    z0, _, _, _ = model.amortized_encode(x_real)
    z_fast = model.refine_latent(x_real, steps=5, step_size=0.1, detach=True, z_init=z0)[-1]
    
    # Then we Decode (Generative Replay)
    with torch.no_grad():
        x_dream_logits = model.decoder(z_fast)
        x_dream = torch.sigmoid(x_dream_logits)

    # 4. Plot Comparison with Residuals
    plt.style.use('seaborn-v0_8-white')
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))  # Changed to 3 rows
    
    for i in range(8):
        # Row 1: Real
        img = x_real[i].cpu().permute(1, 2, 0).squeeze()
        if in_ch == 1: axes[0, i].imshow(img, cmap='gray')
        else: axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Wake (Input)", fontsize=10, fontweight='bold')

        # Row 2: Dream
        img_d = x_dream[i].cpu().permute(1, 2, 0).squeeze()
        if in_ch == 1: axes[1, i].imshow(img_d, cmap='gray')
        else: axes[1, i].imshow(img_d)
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Sleep (Replay)", fontsize=10, fontweight='bold', color='purple')

        # Row 3: The Difference (What was forgotten)
        # We normalize it to make the noise visible
        diff = torch.abs(x_real[i].cpu() - x_dream[i].cpu()).permute(1, 2, 0).squeeze()
        if in_ch == 1: axes[2, i].imshow(diff, cmap='inferno') # 'inferno' makes noise pop
        else: axes[2, i].imshow(diff)
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title("Forgotten Noise\n|Input - Replay|", fontsize=10, fontweight='bold', color='red')

    plt.tight_layout()
    save_p = os.path.join(run_dir, "fig3_dreams_diff.png")
    plt.savefig(save_p, dpi=300)
    save_p = os.path.join(run_dir, "fig3_dreams_diff.svg")
    plt.savefig(save_p, dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_generative_replay()