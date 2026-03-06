"""
Langevin Predictive Coding (LPC) consolidation experiment (Figures 3 and 4).

Three-phase pipeline:
  Phase 1: Train an LPC model (encoder + decoder with gradient-based inference).
  Phase 2: Train a supervised classifier head on frozen encoder latents.
  Phase 3: Consolidation via wake/sleep replay -- compares online (baseline)
           vs offline replay (consolidation with boosted priors).

Supports datasets: MNIST, Fashion-MNIST, EMNIST, CIFAR-10, SVHN.

Usage:
  python lpc_experiment.py --dataset cifar10 --latent 512 --seed 123 --out runs/cifar10_s123
"""
import argparse, os, math, random, json, time, csv
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

# Try to import wandb, but allow running without it
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    pass  # wandb not installed; use --no_wandb or install via: pip install wandb

# --------------- Utils ---------------

def safe_wandb_log(data_dict):
    """Safely log to wandb only if it's available and initialized."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(data_dict)

def safe_wandb_log_artifact(artifact):
    """Safely log artifact to wandb only if it's available and initialized."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log_artifact(artifact)

def set_seed(seed: int = 123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == y).float().mean().item()

def error_rate(logits: torch.Tensor, y: torch.Tensor) -> float:
    return 1.0 - accuracy(logits, y)

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# --------------- Data ---------------

def get_dataloaders(dataset:str="mnist", batch_size:int=128, data_root:str="./data", val_pct:float=0.1):
    tfm = transforms.Compose([
        transforms.ToTensor(),  # [0,1]
        # Normalise around 0.5 helps BCEWithLogits; we’ll handle inside model if needed
    ])
    if dataset.lower() == "mnist":
        train_full = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
        in_ch, n_classes, img_size = 1, 10, 28
    elif dataset.lower() == "fashion":
        train_full = datasets.FashionMNIST(data_root, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(data_root, train=False, download=True, transform=tfm)
        in_ch, n_classes, img_size = 1, 10, 28
    elif dataset.lower() == "cifar10" or dataset.lower() == "cifar-10":
        # CIFAR-10: use [0,1] normalization like MNIST for consistent BCE loss
        # (Standard ImageNet normalization causes issues with BCEWithLogitsLoss)
        train_full = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
        test = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)
        in_ch, n_classes, img_size = 3, 10, 32
    elif dataset.lower() == "svhn":
        # SVHN: Street View House Numbers (RGB, 32x32, 10 classes: digits 0-9)
        # Use [0,1] normalization for consistent BCE loss
        train_full = datasets.SVHN(data_root, split='train', download=True, transform=tfm)
        test = datasets.SVHN(data_root, split='test', download=True, transform=tfm)
        in_ch, n_classes, img_size = 3, 10, 32
    elif dataset.lower() == "emnist":
        # EMNIST: Extended MNIST with letters (grayscale, 28x28, 47 classes)
        # Using 'balanced' split: 47 balanced classes (digits + uppercase + lowercase)
        train_full = datasets.EMNIST(data_root, split='balanced', train=True, download=True, transform=tfm)
        test = datasets.EMNIST(data_root, split='balanced', train=False, download=True, transform=tfm)
        in_ch, n_classes, img_size = 1, 47, 28
    else:
        raise ValueError("dataset must be 'mnist', 'fashion', 'cifar10', 'svhn', or 'emnist'")

    val_size = int(len(train_full) * val_pct)
    train_size = len(train_full) - val_size
    train, val = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    dl_train = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val   = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    dl_test  = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return dl_train, dl_val, dl_test, in_ch, n_classes, img_size

# --------------- Models ---------------

class ConvEncoder(nn.Module):
    """Conv encoder: flexible for 28x28 (MNIST) or 32x32 (CIFAR-10) inputs.
    Architecture adapts based on input channels - deeper for RGB images.
    """
    def __init__(self, in_ch:int=1, latent:int=32, img_size:int=28):
        super().__init__()
        self.img_size = img_size
        
        # Use deeper/wider architecture for RGB images (CIFAR-10)
        if in_ch >= 3:  # Color images need more capacity
            # Deeper architecture: 32x32 -> 16x16 -> 8x8 -> 4x4
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),  nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=2, padding=1),     nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 16x16
                nn.Conv2d(64, 128, 3, stride=1, padding=1),    nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),   nn.BatchNorm2d(128), nn.ReLU(inplace=True), # 8x8
                nn.Conv2d(128, 256, 3, stride=1, padding=1),   nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=2, padding=1),   nn.BatchNorm2d(256), nn.ReLU(inplace=True), # 4x4
            )
            # After 3 stride-2 layers: 32 -> 16 -> 8 -> 4
            spatial_dim = img_size // 8
            self.flat_dim = 256 * spatial_dim * spatial_dim
        else:  # Grayscale images (MNIST) - keep original simple architecture
            # Original architecture: 2 stride-2 convs
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # img_size//2
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),     # img_size//4
                nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True),     # img_size//4
            )
            spatial_dim = img_size // 4
            self.flat_dim = 64 * spatial_dim * spatial_dim
        
        self.fc = nn.Linear(self.flat_dim, latent)
        
        # Initialize weights properly for deep networks
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for deep networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z

    @torch.no_grad()
    def encode_deterministic(self, x: torch.Tensor):
        """Deterministic encoding: returns z used for head training."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class ConvDecoder(nn.Module):
    """Mirror decoder with convtranspose; outputs logits in [B, out_ch, img_size, img_size]."""
    def __init__(self, out_ch:int=1, latent:int=32, img_size:int=28):
        super().__init__()
        self.img_size = img_size
        
        # Use architecture matching encoder
        if out_ch >= 3:  # Color images - deeper decoder
            # Start from 4x4, upsample to 8x8, 16x16, 32x32
            spatial_dim = img_size // 8  # 4 for 32x32 images
            self.spatial_dim = spatial_dim
            self.n_channels = 256  # Number of channels after fc layer
            
            self.fc = nn.Linear(latent, 256 * spatial_dim * spatial_dim)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), # 8x8
                nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(inplace=True), # 16x16
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, out_ch, 4, stride=2, padding=1),                                           # 32x32
                # no activation: we'll use BCEWithLogitsLoss
            )
        else:  # Grayscale - original simple architecture
            spatial_dim = img_size // 4
            self.spatial_dim = spatial_dim
            self.n_channels = 64  # Number of channels after fc layer
            
            self.fc = nn.Linear(latent, 64 * spatial_dim * spatial_dim)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True), # spatial_dim
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True), # spatial_dim*2
                nn.ConvTranspose2d(32, out_ch, 4, stride=2, padding=1),                    # spatial_dim*4 = img_size
                # no activation: we'll use BCEWithLogitsLoss
            )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for deep networks."""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor):
        h = self.fc(z)
        h = h.view(h.size(0), self.n_channels, self.spatial_dim, self.spatial_dim)
        x_logits = self.deconv(h)
        return x_logits

class LPCModel(nn.Module):
    """Simplified Langevin Predictive Coding model with amortised encoder + decoder."""

    def __init__(self, in_ch:int=1, latent:int=32, img_size:int=28, prior_beta:float=1e-3):
        super().__init__()
        self.encoder = ConvEncoder(in_ch, latent, img_size=img_size)
        self.decoder = ConvDecoder(in_ch, latent, img_size=img_size)
        self.prior_beta = prior_beta

    def amortized_encode(self, x: torch.Tensor):
        """Return stochastic sample, deterministic code, and optional (mu, logvar)."""
        z = self.encoder(x)
        return z, z, None, None

    def energy(self, z: torch.Tensor, x: torch.Tensor):
        recon = self.decoder(z)
        recon_loss = bce_recon_loss(recon, x)
        prior = 0.5 * z.pow(2).mean()
        total = recon_loss + self.prior_beta * prior
        return total, recon, recon_loss, prior

    def refine_latent(self, x: torch.Tensor, steps:int=3, step_size:float=0.1,
                      noise_std:float=0.0, detach:bool=True, z_init:Optional[torch.Tensor]=None):
        """Run LPC inference (gradient-based refinement) starting from amortized z0."""
        if z_init is None:
            z_init = self.encoder.encode_deterministic(x)
        z = z_init
        latents = [z]
        with torch.enable_grad():
            for _ in range(steps):
                z = z.requires_grad_(True)
                energy, _, _, _ = self.energy(z, x)
                grad = torch.autograd.grad(energy, z, create_graph=not detach)[0]
                z = z - step_size * grad
                if noise_std > 0.0:
                    z = z + noise_std * torch.randn_like(z)
                if detach:
                    z = z.detach()
                latents.append(z)
        return latents

    def forward(self, x: torch.Tensor, steps:int=3, step_size:float=0.1,
                noise_std:float=0.0):
        z0, _, _, _ = self.amortized_encode(x)
        latents = self.refine_latent(x, steps=steps, step_size=step_size,
                                     noise_std=noise_std, detach=False, z_init=z0)
        z_final = latents[-1]
        energy, recon, recon_loss, prior = self.energy(z_final, x)
        return recon, latents, energy, recon_loss, prior

class ClassifierHead(nn.Module):
    def __init__(self, in_dim:int, n_classes:int, hidden:int=128, p_drop:float=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, z: torch.Tensor):
        return self.net(z)

def single_langevin_step(model: LPCModel, x: torch.Tensor, z: torch.Tensor,
                         step_size:float, noise_std:float, detach:bool):
    """Utility to perform one LPC refinement step."""
    with torch.enable_grad():
        z = z.requires_grad_(True)
        energy, _, _, _ = model.energy(z, x)
        grad = torch.autograd.grad(energy, z, create_graph=not detach)[0]
    z = z - step_size * grad
    if noise_std > 0.0:
        z = z + noise_std * torch.randn_like(z)
    if detach:
        z = z.detach()
    return z

# --------------- Training Phases ---------------

@dataclass
class Phase1Config:
    epochs:int=20
    lr:float=1e-3
    wd:float=1e-4
    patience:int=5
    inference_steps:int=5
    inference_lr:float=0.1
    inference_noise:float=0.01
    prior_beta:float=1e-3
    amort_weight:float=0.1

@dataclass
class Phase2Config:
    epochs:int=15
    lr:float=5e-4
    wd:float=5e-4
    patience:int=5
    freeze_encoder:bool=True
    head_hidden:int=128
    dropout:float=0.1

@dataclass
class Phase3Config:
    T_wake:int=3
    T_sleep:int=10
    inference_noise_wake:float=0.01
    inference_noise_sleep:float=0.01
    beta_wake:float=0.001
    beta_sleep:float=0.1
    inference_lr:float=0.1
    lr:float=5e-4
    wd:float=5e-4
    epochs:int=20
    patience:int=6
    head_hidden:int=256
    dropout:float=0.1
    label_smooth:float=0.0

def bce_recon_loss(x_logits, x):
    # BCEWithLogitsLoss on [0,1] targets
    return F.binary_cross_entropy_with_logits(x_logits, x, reduction="mean")

def train_phase1_lpc(model: LPCModel, dl_train, dl_val, cfg: Phase1Config, device, out_dir):
    model = model.to(device)
    model.prior_beta = cfg.prior_beta
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )

    best_val = float("inf")
    patience = cfg.patience
    history = []

    def refine_forward(x, detach: bool):
        z0, z_det, mu, logvar = model.amortized_encode(x)
        latents = model.refine_latent(
            x,
            steps=cfg.inference_steps,
            step_size=cfg.inference_lr,
            noise_std=cfg.inference_noise,
            detach=detach,
            z_init=z0,
        )
        zT = latents[-1]
        energy, recon_logits, recon_loss, prior = model.energy(zT, x)
        return {
            "energy": energy,
            "recon": recon_loss,
            "prior": prior,
            "z_det": z_det,
            "z_final": zT,
            "mu": mu,
            "logvar": logvar,
        }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr = {"loss": 0.0, "recon": 0.0, "prior": 0.0, "amort": 0.0}

        for x, _ in dl_train:
            x = x.to(device)
            opt.zero_grad()
            out = refine_forward(x, detach=False)
            loss = out["recon"] + cfg.prior_beta * out["prior"]
            if cfg.amort_weight > 0:
                amort = F.mse_loss(out["z_det"], out["z_final"].detach())
                loss = loss + cfg.amort_weight * amort
            else:
                amort = torch.tensor(0.0, device=device)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            B = x.size(0)
            tr["loss"] += loss.item() * B
            tr["recon"] += out["recon"].item() * B
            tr["prior"] += out["prior"].item() * B
            tr["amort"] += amort.item() * B

        for k in tr:
            tr[k] /= len(dl_train.dataset)

        if not math.isfinite(tr["loss"]):
            raise ValueError(f"Training diverged with loss={tr['loss']}")

        model.eval()
        va = {"loss": 0.0, "recon": 0.0, "prior": 0.0, "amort": 0.0}
        with torch.no_grad():
            for x, _ in dl_val:
                x = x.to(device)
                out = refine_forward(x, detach=True)
                loss = out["recon"] + cfg.prior_beta * out["prior"]
                if cfg.amort_weight > 0:
                    amort = F.mse_loss(out["z_det"], out["z_final"].detach())
                else:
                    amort = torch.tensor(0.0, device=device)
                B = x.size(0)
                va["loss"] += loss.item() * B
                va["recon"] += out["recon"].item() * B
                va["prior"] += out["prior"].item() * B
                va["amort"] += amort.item() * B

        for k in va:
            va[k] /= len(dl_val.dataset)

        log_dict = {
            "phase1/epoch": epoch,
            "phase1/train_loss": tr["loss"],
            "phase1/val_loss": va["loss"],
            "phase1/train_recon": tr["recon"],
            "phase1/val_recon": va["recon"],
            "phase1/train_prior": tr["prior"],
            "phase1/val_prior": va["prior"],
            "phase1/train_amort": tr["amort"],
            "phase1/val_amort": va["amort"],
        }
        safe_wandb_log(log_dict)

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        })

        print(
            f"[P1][{epoch:02d}] train loss {tr['loss']:.4f} | val loss {va['loss']:.4f}"
            f" (recon {va['recon']:.4f}, prior {va['prior']:.4f})"
        )

        if va["loss"] + 1e-6 < best_val:
            best_val = va["loss"]
            patience = cfg.patience
            torch.save(model.state_dict(), os.path.join(out_dir, "phase1_lpc_best.pt"))
        else:
            patience -= 1
            if patience <= 0:
                print("[P1] Early stopping.")
                break

        scheduler.step()

    best_path = os.path.join(out_dir, "phase1_lpc_best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    with open(os.path.join(out_dir, "phase1_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, best_val, history


def train_phase2_head(encoder: ConvEncoder, head: ClassifierHead, dl_train, dl_val, dl_test, cfg: Phase2Config, device, out_dir):
    if cfg.freeze_encoder:
        for p in encoder.parameters(): p.requires_grad = False
    encoder.eval()
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    best_val = float("inf")
    patience = cfg.patience
    ce = nn.CrossEntropyLoss()
    history = []

    def run_epoch(dl, train=False):
        (head.train() if train else head.eval())
        total_loss = 0.0
        all_logits = []
        all_labels = []
        with torch.set_grad_enabled(train):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    z = encoder.encode_deterministic(x)
                logits = head(z)
                loss = ce(logits, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                    opt.step()
                total_loss += loss.item() * x.size(0)
                all_logits.append(logits.detach())
                all_labels.append(y.detach())
        total_loss /= len(dl.dataset)
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        acc = accuracy(logits, labels)
        err = 1.0 - acc
        return total_loss, acc, err

    for epoch in range(1, cfg.epochs+1):
        tr_loss, tr_acc, tr_err = run_epoch(dl_train, train=True)
        va_loss, va_acc, va_err = run_epoch(dl_val, train=False)
        
        # Log to W&B
        safe_wandb_log({
            "phase2/epoch": epoch,
            "phase2/train_loss": tr_loss,
            "phase2/train_acc": tr_acc,
            "phase2/train_err": tr_err,
            "phase2/val_loss": va_loss,
            "phase2/val_acc": va_acc,
            "phase2/val_err": va_err,
        })
        
        history.append({"epoch":epoch, "train_loss":tr_loss, "val_loss":va_loss, "train_acc":tr_acc, "val_acc":va_acc})
        print(f"[P2][{epoch:02d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            patience = cfg.patience
            torch.save(head.state_dict(), os.path.join(out_dir, "phase2_head_best.pt"))
        else:
            patience -= 1
            if patience <= 0:
                print("[P2] Early stopping.")
                break

    head.load_state_dict(torch.load(os.path.join(out_dir, "phase2_head_best.pt"), map_location=device))
    tr_loss, tr_acc, tr_err = run_epoch(dl_train, train=False)
    te_loss, te_acc, te_err = run_epoch(dl_test,  train=False)
    gap = te_err - tr_err
    print(f"[P2] FINAL  train acc {tr_acc:.3f} err {tr_err:.3f} | test acc {te_acc:.3f} err {te_err:.3f} | gen gap {gap:.3f}")

    # Log final metrics to W&B
    safe_wandb_log({
        "phase2/final_train_acc": tr_acc,
        "phase2/final_train_err": tr_err,
        "phase2/final_test_acc": te_acc,
        "phase2/final_test_err": te_err,
        "phase2/final_gen_gap": gap,
    })
    
    # Log model as artifact
    if WANDB_AVAILABLE and wandb.run is not None:
        artifact = wandb.Artifact(name=f"phase2_head", type="model")
        artifact.add_file(os.path.join(out_dir, "phase2_head_best.pt"))
        wandb.log_artifact(artifact)

    summary_path = os.path.join(out_dir, "phase2_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "train_acc": tr_acc, "train_err": tr_err,
            "test_acc": te_acc, "test_err": te_err,
            "gen_gap": gap
        }, f, indent=2)
    print(f"[P2] Saved summary: {summary_path}")
    return head


# --------------- Phase 3: Replay & Consolidation ---------------

def collect_hippocampal_traces(lpc_model, dl, device, inference_steps=5, inference_lr=0.1, noise_std=0.05): # Added noise arg
    """
    Simulate 'Wake' phase. Pass data through the frozen model with limited inference steps.
    These represent the noisy, approximate posteriors formed during rapid perception.
    """
    lpc_model.eval()
    buffer_z = []
    buffer_y = []
    
    print(f"[Wake] Collecting traces (Online Encoding, T={inference_steps}, Noise={noise_std})...")
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            # 1. Amortized Encode
            z0, _, _, _ = lpc_model.amortized_encode(x)

            # 2. Short Online Inference (The "Fast" encoding)
            # INJECT NOISE HERE to simulate sensory uncertainty
            latents = lpc_model.refine_latent(
                x, steps=inference_steps, step_size=inference_lr, 
                noise_std=noise_std,  # <--- CRITICAL CHANGE
                detach=True, z_init=z0
            )
            z_fast = latents[-1]
            
            buffer_z.append(z_fast.cpu())
            buffer_y.append(y)
            
    return torch.cat(buffer_z), torch.cat(buffer_y)

class ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, z_tensor, y_tensor):
        self.z = z_tensor
        self.y = y_tensor
    def __len__(self): return len(self.z)
    def __getitem__(self, idx): return self.z[idx], self.y[idx]

def train_phase3_consolidation(mode, lpc_model, z_mem, y_mem, dl_val, dl_test, cfg: Phase3Config, device, out_dir):
    print(f"\n--- Starting Phase 3 Training: MODE = {mode.upper()} ---")

    original_beta = lpc_model.prior_beta
    
    # If we are in 'replay' (Sleep) mode, we boost the prior strength.
    # This forces the 'energy minimization' to care more about the prior (0) than the reconstruction.
    if mode == "replay":
        lpc_model.prior_beta = cfg.beta_sleep # Perhaps stronger gravity during sleep!
        print(f"[Config] Sleep Mode: Boosted prior_beta to {lpc_model.prior_beta}")
    else:
        # Ensure baseline uses the weak prior (standard perception)
        lpc_model.prior_beta = cfg.beta_wake
        print(f"[Config] Wake Mode: Using prior_beta = {lpc_model.prior_beta}")


    # 1. Setup Data
    replay_ds = ReplayDataset(z_mem, y_mem)
    replay_dl = DataLoader(replay_ds, batch_size=128, shuffle=True, num_workers=0)
    
    # 2. Initialize Head
    latent_dim = z_mem.shape[1]
    n_classes = int(y_mem.max().item()) + 1
    head = ClassifierHead(latent_dim, n_classes, hidden=cfg.head_hidden, p_drop=cfg.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)

    # 3. Evaluation Helper
    def evaluate_full(loader):
        head.eval()
        loss_sum, corr, count = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                # Standard recognition path: x -> z_fast -> head
                z0, _, _, _ = lpc_model.amortized_encode(x)
                z_eval = lpc_model.refine_latent(x, steps=cfg.T_wake, step_size=cfg.inference_lr, detach=True, z_init=z0)[-1]
                logits = head(z_eval)
                loss_sum += ce(logits, y).item() * x.size(0)
                corr += (logits.argmax(1) == y).sum().item()
                count += x.size(0)
        return loss_sum/count, corr/count

    # 4. Training Loop
    history = []
    
    for epoch in range(1, cfg.epochs + 1):
        head.train()
        tr_loss_sum, tr_corr, tr_count = 0.0, 0, 0
        
        # Stats for Geometry Analysis
        avg_z_norm = 0.0
        
        for z_batch, y_batch in replay_dl:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)
            
            if mode == "online":
                z_input = z_batch # Noisy
            elif mode == "replay":
                # The Inverse-Forward Loop
                with torch.no_grad():
                    x_hat_logits = lpc_model.decoder(z_batch) # Dream
                    # CRITICAL FIX: Convert logits to probabilities [0,1]
                    # The refinement loss (BCEWithLogits) expects targets in [0,1]
                    x_hat = torch.sigmoid(x_hat_logits) 
                
                latents = lpc_model.refine_latent(
                    x_hat, 
                    steps=cfg.T_sleep,  # Deep refinement
                    step_size=cfg.inference_lr, 
                    noise_std=cfg.inference_noise_sleep,
                    detach=True, 
                    z_init=z_batch 
                )
                z_input = latents[-1] # Refined

            # Track Geometry (Compression)
            avg_z_norm += torch.norm(z_input, dim=1).mean().item()

            # Update Head
            logits = head(z_input)
            loss = ce(logits, y_batch)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            tr_loss_sum += loss.item() * z_batch.size(0)
            tr_corr += (logits.argmax(1) == y_batch).sum().item()
            tr_count += z_batch.size(0)
            
        train_loss = tr_loss_sum / tr_count
        train_acc = tr_corr / tr_count
        avg_z_norm /= len(replay_dl)
        
        # --- CRITICAL CHANGE: EVALUATE TEST SET EVERY EPOCH ---
        # This gives you the consistent "Generalisation Gap" plot
        test_loss, test_acc = evaluate_full(dl_test)
        
        # Gap = Test Error - Train Error (using accuracy for consistency with AE)
        # Train Error = 1 - Train Acc
        # Test Error = 1 - Test Acc
        # Gap = (1 - Test Acc) - (1 - Train Acc) = Train Acc - Test Acc
        # (Alternatively, use Loss Gap: Test Loss - Train Loss)
        gen_gap_acc = train_acc - test_acc
        gen_gap_loss = test_loss - train_loss
        
        print(f"[{mode.upper()}] Ep {epoch}: TrLoss {train_loss:.3f} | TeLoss {test_loss:.3f} | GAP(Loss) {gen_gap_loss:.4f} | Z_Norm {avg_z_norm:.2f}")
        
        safe_wandb_log({
            f"p3_{mode}/epoch": epoch,
            f"p3_{mode}/train_loss": train_loss,
            f"p3_{mode}/test_loss": test_loss,
            f"p3_{mode}/gen_gap_loss": gen_gap_loss,
            f"p3_{mode}/gen_gap_acc": gen_gap_acc,
            f"p3_{mode}/z_norm": avg_z_norm
        })
        
        history.append({
            "epoch": epoch, 
            "train_loss": train_loss, "test_loss": test_loss, 
            "train_acc": train_acc, "test_acc": test_acc,
            "gap_loss": gen_gap_loss, "gap_acc": gen_gap_acc,
            "z_norm": avg_z_norm
        })

    with open(os.path.join(out_dir, f"history_{mode}.json"), "w") as f:
        json.dump(history, f, indent=2)


    # Reset it back at the end if you reuse the model
    lpc_model.prior_beta = original_beta
    print(f"[Config] Reset prior_beta to original value: {lpc_model.prior_beta}")

    return head

# --------------- Main Experiment Script ---------------

def main():
    ap = argparse.ArgumentParser(description="LPC -> Head -> LPC refinement experiment")
    ap.add_argument("--dataset", choices=["mnist","fashion","cifar10","svhn","emnist"], default="mnist")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="./runs/exp1")
    # Phase1
    ap.add_argument("--p1_epochs", type=int, default=20)
    ap.add_argument("--p1_lr", type=float, default=1e-3)
    ap.add_argument("--p1_wd", type=float, default=1e-4)
    ap.add_argument("--p1_patience", type=int, default=5)
    ap.add_argument("--p1_inf_steps", type=int, default=5)
    ap.add_argument("--p1_inf_lr", type=float, default=0.1)
    ap.add_argument("--p1_inf_noise", type=float, default=0.01)
    ap.add_argument("--p1_prior_beta", type=float, default=1e-3)
    ap.add_argument("--p1_amort_weight", type=float, default=0.1)
    # Phase2
    ap.add_argument("--p2_epochs", type=int, default=15)
    ap.add_argument("--p2_lr", type=float, default=5e-4)
    ap.add_argument("--p2_wd", type=float, default=5e-4)
    ap.add_argument("--p2_patience", type=int, default=5)
    ap.add_argument("--p2_head_hidden", type=int, default=128)
    ap.add_argument("--p2_dropout", type=float, default=0.1)
    # Phase3
    ap.add_argument("--p3_T_wake", type=int, default=3, help="Number of inference steps during wake encoding")
    ap.add_argument("--p3_T_sleep", type=int, default=10, help="Number of inference steps during sleep replay")
    ap.add_argument("--p3_noise_wake", type=float, default=0.01, help="Inference noise during wake encoding (sensory uncertainty)") # <--- Ensure this is > 0.0
    ap.add_argument("--p3_noise_sleep", type=float, default=0.01, help="Inference noise during sleep replay (clean processing)")
    ap.add_argument("--p3_beta_wake", type=float, default=0.001, help="Prior strength during wake (weak - trust sensory input)")
    ap.add_argument("--p3_beta_sleep", type=float, default=0.1, help="Prior strength during sleep (strong - compress toward 0)")
    ap.add_argument("--p3_inf_lr", type=float, default=0.1)
    ap.add_argument("--p3_epochs", type=int, default=20)
    ap.add_argument("--p3_lr", type=float, default=5e-4)
    ap.add_argument("--p3_wd", type=float, default=5e-4)
    ap.add_argument("--p3_patience", type=int, default=6)
    ap.add_argument("--p3_head_hidden", type=int, default=256)
    ap.add_argument("--p3_dropout", type=float, default=0.1)
    ap.add_argument("--p3_label_smooth", type=float, default=0.0, help="label smoothing for CE in phase 3")
    ap.add_argument("--p3_crossfit_ratio", type=float, default=0.5,
                    help="fraction of train data reserved to train Phase 3 head (rest used for gap eval). Set 0 to disable.")
    # W&B arguments
    ap.add_argument("--wandb_project", type=str, default="consolidation-experiments", help="W&B project name")
    ap.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (username or team)")
    ap.add_argument("--wandb_group", type=str, default=None, help="W&B group for multi-seed runs (auto-generated if not provided)")
    ap.add_argument("--wandb_tags", type=str, default="", help="Comma-separated W&B tags")
    ap.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    args = ap.parse_args()

    set_seed(args.seed)
    device = device_auto()
    os.makedirs(args.out, exist_ok=True)
    
    args_path = os.path.join(args.out, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[SETUP] Output directory: {args.out}")
    print(f"[SETUP] Saved args: {args_path}")

    dl_train, dl_val, dl_test, in_ch, n_classes, img_size = get_dataloaders(dataset=args.dataset, batch_size=args.batch_size)

    # Initialize W&B
    use_wandb = not args.no_wandb and WANDB_AVAILABLE
    
    mode_tag = "lpc"

    if use_wandb:
        # Auto-generate group name for multi-seed experiments if not provided
        if args.wandb_group is None:
            # Group name: dataset_mode_latent (without seed)
            args.wandb_group = f"{args.dataset}_{mode_tag}_lat{args.latent}"

        # Parse tags
        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        tags.extend([args.dataset, mode_tag, "ae"])

        # Create run name: includes seed for identification
        run_name = f"{args.dataset}_{mode_tag}_s{args.seed}"
        
        # Initialize W&B
        try:
            print("[W&B] Logging in...")
            wandb.login(relogin=True, timeout=30)
            
            # Create a dedicated wandb directory to avoid scanning large output dirs
            wandb_dir = os.path.join(args.out, ".wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            
            print("[W&B] Initializing run...")
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                group=args.wandb_group,
                tags=tags,
                config=vars(args),
                dir=wandb_dir,  # Use dedicated subdir instead of output dir
                settings=wandb.Settings(
                    _disable_stats=True,      # Disable system stats collection
                    _disable_meta=True,       # Disable metadata collection
                    start_method="thread",    # Use threading instead of fork
                    init_timeout=60,          # Timeout for initialization
                ),
            )
            
            print("[W&B] Updating config...")
            # Log system info
            wandb.config.update({
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "img_size": img_size,
                "in_channels": in_ch,
                "n_classes": n_classes,
            })
            print(f"✓ W&B initialized: {run_name} (group: {args.wandb_group})")
        except Exception as e:
            print(f"Warning: W&B initialization failed: {e}")
            print("Continuing without W&B logging...")
            use_wandb = False
    elif not args.no_wandb and not WANDB_AVAILABLE:
        print("Warning: W&B not available. Install with: pip install wandb")
        print("Continuing without W&B logging...")
    
    # Store whether to use wandb in args for access in training functions
    args._use_wandb = use_wandb

    # ----- Cross-fit split for Phase 2 vs Phase 3 -----
    # We split the original training set into A (for head & 'train' metric) and B (for refiner training).
    # If ratio=0, we reuse the same loader for both (no cross-fit).
    from torch.utils.data import Subset
    base_train_ds = dl_train.dataset  # underlying Subset of original torchvision dataset

    ratio = getattr(args, "p3_crossfit_ratio", 0.5)
    if ratio > 0 and 0 < ratio < 1.0:
        N = len(base_train_ds)
        nB = int(round(N * ratio))     # B: for refiner training
        nA = N - nB                    # A: for head and "train" metrics
        gen = torch.Generator().manual_seed(args.seed + 999)
        idx = torch.randperm(N, generator=gen).tolist()
        idxB = idx[:nB]
        idxA = idx[nB:]
        dsA = Subset(base_train_ds, idxA)
        dsB = Subset(base_train_ds, idxB)
        # Make loaders
        dlA = DataLoader(dsA, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        dlB = DataLoader(dsB, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        dlA_eval = DataLoader(dsA, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        dlA = dl_train
        dlB = dl_train
        dlA_eval = DataLoader(dl_train.dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # -----------------------------------------------

    # Phase 1: LPC
    p1cfg = Phase1Config(
        epochs=args.p1_epochs,
        lr=args.p1_lr,
        wd=args.p1_wd,
        patience=args.p1_patience,
        inference_steps=args.p1_inf_steps,
        inference_lr=args.p1_inf_lr,
        inference_noise=args.p1_inf_noise,
        prior_beta=args.p1_prior_beta,
        amort_weight=args.p1_amort_weight,
    )
    lpc = LPCModel(in_ch=in_ch, latent=args.latent, img_size=img_size, prior_beta=args.p1_prior_beta)
    n_params = count_params(lpc)
    enc_params = count_params(lpc.encoder)
    dec_params = count_params(lpc.decoder)
    print(f"[P1] Model params: LPC {n_params:,} (encoder: {enc_params:,}, decoder: {dec_params:,})")
    print(f"[P1] Architecture: {in_ch}ch {img_size}x{img_size} -> latent {args.latent}")
    compression_ratio = (in_ch * img_size * img_size) / args.latent
    print(f"[P1] Compression ratio: {compression_ratio:.1f}:1")
    lpc, _, _ = train_phase1_lpc(lpc, dl_train, dl_val, p1cfg, device, args.out)

    # Phase 2: Head
    p2cfg = Phase2Config(
        epochs=args.p2_epochs, lr=args.p2_lr, wd=args.p2_wd, patience=args.p2_patience,
        freeze_encoder=True, head_hidden=args.p2_head_hidden, dropout=args.p2_dropout
    )
    head = ClassifierHead(in_dim=args.latent, n_classes=n_classes,
                          hidden=args.p2_head_hidden, p_drop=args.p2_dropout).to(device)
    head = train_phase2_head(lpc.encoder, head, dlA, dl_val, dl_test, p2cfg, device, args.out)

    # Phase 3: LPC refinement - The Consolidation Experiment 
    print("\n[Phase 3] Preparing for Consolidation...")
    
    # 1. Freeze LPC (The Generative Model)
    for p in lpc.parameters(): p.requires_grad_(False)
    
    # 2. Collect "Hippocampal" Buffer (Simulate Wake Encoding)
    # We use dlA (split A) as the 'source' of experiences
    z_mem, y_mem = collect_hippocampal_traces(
        lpc, dlA, device, 
        inference_steps=args.p3_T_wake, 
        inference_lr=args.p3_inf_lr,
        noise_std=args.p3_noise_wake  # Sensory uncertainty during wake
    )

    # 3. Define Config
    p3cfg = Phase3Config(
        T_wake=args.p3_T_wake,
        T_sleep=args.p3_T_sleep,
        inference_noise_wake=args.p3_noise_wake,
        inference_noise_sleep=args.p3_noise_sleep,
        beta_wake=args.p3_beta_wake,
        beta_sleep=args.p3_beta_sleep,
        inference_lr=args.p3_inf_lr,
        lr=args.p3_lr,
        wd=args.p3_wd,
        epochs=args.p3_epochs,
        patience=args.p3_patience,
        head_hidden=args.p3_head_hidden,
        dropout=args.p3_dropout,
        label_smooth=args.p3_label_smooth,
    )

    # 4. Run Condition A: ONLINE (Baseline)
    # Trains head directly on z_mem (noisy).
    head_online = train_phase3_consolidation(
        "online", lpc, z_mem, y_mem, dl_val, dl_test, p3cfg, device, out_dir=args.out
    )

    # 5. Run Condition B: REPLAY (Experiment)
    # Trains head on z_refined (via Dream loop).
    head_replay = train_phase3_consolidation(
        "replay", lpc, z_mem, y_mem, dl_val, dl_test, p3cfg, device, out_dir=args.out
    )
    

    print("\n[Done] Experiment Complete. Check wandb/json logs for Generalization Gap comparison.")


    # Finish W&B run
    if WANDB_AVAILABLE and wandb.run is not None and not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()