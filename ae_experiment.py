"""
Autoencoder consolidation experiment (Figures 2 and S1).

Three-phase pipeline:
  Phase 1: Train a convolutional autoencoder (AE or VAE).
  Phase 2: Train a supervised classifier head on frozen encoder latents.
  Phase 3: Train a refiner that iteratively consolidates latent representations
           (fixed-step or adaptive PonderNet halting).

Supports datasets: MNIST, Fashion-MNIST, EMNIST, CIFAR-10, SVHN.

Usage:
  python ae_experiment.py --dataset mnist --mode ponder --T 4 --seed 123 --out runs/mnist_s123
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
    print("Warning: wandb not installed. Run 'pip install wandb' to enable experiment tracking.")

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
    def __init__(self, in_ch:int=1, latent:int=32, vae:bool=False, img_size:int=28):
        super().__init__()
        self.vae = vae
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
        
        if vae:
            self.fc_mu     = nn.Linear(self.flat_dim, latent)
            self.fc_logvar = nn.Linear(self.flat_dim, latent)
        else:
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
        if self.vae:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h).clamp(-10, 10)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            z = self.fc(h)
            return z

    @torch.no_grad()
    def encode_deterministic(self, x: torch.Tensor):
        """Use mean for VAE; z for AE. This is the z used for head/refiner training."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        if hasattr(self, "fc_mu"):
            mu = self.fc_mu(h)
            return mu
        else:
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

class Autoencoder(nn.Module):
    """AE/VAE wrapper."""
    def __init__(self, in_ch:int=1, latent:int=32, vae:bool=False, img_size:int=28):
        super().__init__()
        self.encoder = ConvEncoder(in_ch, latent, vae=vae, img_size=img_size)
        self.decoder = ConvDecoder(in_ch, latent, img_size=img_size)
        self.vae = vae

    def forward(self, x: torch.Tensor):
        if self.vae:
            z, mu, logvar = self.encoder(x)
            x_logits = self.decoder(z)
            return x_logits, z, mu, logvar
        else:
            z = self.encoder(x)
            x_logits = self.decoder(z)
            return x_logits, z

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

class RefinerFixed(nn.Module):
    """Shared MLP applied T times: z_{t+1} = z_t + f(z_t), with LayerNorm + residual gate."""
    def __init__(self, dim:int, hidden:int=256):
        super().__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )
        self.gate = nn.Parameter(torch.zeros(1))  # scalar gate init 0

    def forward(self, z0: torch.Tensor, T:int=3) -> List[torch.Tensor]:
        zs = [z0]
        z = z0
        for _ in range(T):
            delta = self.f(z)
            z = z + torch.tanh(self.gate) * delta
            zs.append(z)
        return zs  # length T+1

class RefinerPonder(nn.Module):
    """
    Ponder-like halting:
      - shared residual block as above
      - per-step halting logits -> p_t = sigmoid(h(z_t))
      - q(t) = p_t * prod_{i < t} (1 - p_i); q(T) += leftover (truncate)
      - returns zs, q_t per step (B,T)
    """
    def __init__(self, dim:int, hidden:int=256):
        super().__init__()
        self.core = RefinerFixed(dim, hidden)
        self.halt = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, z0: torch.Tensor, T:int=3):
        zs = self.core(z0, T)  # list len T+1: z0..zT
        # compute halting from z1..zT (no halting at t=0)
        z_steps = zs[1:]  # length T, each [B,D]
        B = z0.size(0)
        p = []
        for z in z_steps:
            p_t = torch.sigmoid(self.halt(z)).squeeze(-1)  # [B]
            p.append(p_t)
        p = torch.stack(p, dim=1)  # [B, T]

        # posterior q over steps 1..T (truncated geometric)
        # q_t = p_t * prod_{i < t} (1 - p_i)
        one_m_p = (1 - p).clamp(1e-6, 1 - 1e-6)
        cumprod = torch.cumprod(one_m_p, dim=1)
        # shift cumprod right by 1 with 1 at start for prod i<1
        prefix = torch.ones_like(p)
        prefix[:, 1:] = cumprod[:, :-1]
        q = (p * prefix)  # [B,T]
        # make sure it sums to 1 by pushing leftover into last step:
        leftover = (1 - q.sum(dim=1, keepdim=True)).clamp(min=0.0)
        q[:, -1:] = q[:, -1:] + leftover
        return zs, q  # zs length T+1, q over steps 1..T

def geometric_prior(T:int, lam:float, device):
    """
    Truncated geometric prior over steps 1..T:
      p(t) = lam * (1-lam)^{t-1} for t < T
      p(T) = (1-lam)^{T-1}  (absorbs tail)
    """
    t = torch.arange(1, T+1, device=device, dtype=torch.float32)
    p = lam * (1 - lam) ** (t - 1)
    p[-1] = (1 - lam) ** (T - 1)
    p = p / p.sum()  # normalise (should already sum 1)
    return p  # [T]

# --------------- Training Phases ---------------

@dataclass
class Phase1Config:
    epochs:int=20
    lr:float=1e-3
    wd:float=1e-4
    patience:int=5
    vae:bool=False
    beta_kl:float=1.0

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
    mode:str="fixed"   # "fixed" or "ponder"
    T:int=3
    lr:float=5e-4
    wd:float=5e-4
    epochs:int=20
    patience:int=6
    ref_hidden:int=256
    finetune_head:bool=False
    ponder_prior_lambda:float=0.3
    ponder_beta:float=0.01  # weight for KL(q||prior)

def bce_recon_loss(x_logits, x):
    # BCEWithLogitsLoss on [0,1] targets
    return F.binary_cross_entropy_with_logits(x_logits, x, reduction="mean")

def vae_kl(mu, logvar):
    # 0.5 * sum (mu^2 + sigma^2 - logvar - 1)
    return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1.0)

def train_phase1_autoencoder(model: Autoencoder, dl_train, dl_val, cfg: Phase1Config, device, out_dir):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    
    # Cosine annealing scheduler helps converge better for difficult datasets like CIFAR-10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    
    best_val = float("inf")
    patience = cfg.patience
    criterion = bce_recon_loss
    history = []

    for epoch in range(1, cfg.epochs+1):
        model.train()
        tr_loss = 0.0
        tr_recon = 0.0 if cfg.vae else None
        tr_kl = 0.0 if cfg.vae else None
        
        for x, _ in dl_train:
            x = x.to(device)
            opt.zero_grad()
            if cfg.vae:
                x_logits, z, mu, logvar = model(x)
                recon = criterion(x_logits, x)
                kl = vae_kl(mu, logvar)
                loss = recon + cfg.beta_kl * kl
                tr_recon += recon.item() * x.size(0)
                tr_kl += kl.item() * x.size(0)
            else:
                x_logits, z = model(x)
                loss = criterion(x_logits, x)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(dl_train.dataset)
        if cfg.vae:
            tr_recon /= len(dl_train.dataset)
            tr_kl /= len(dl_train.dataset)
        
        # Sanity check for NaN
        if not torch.isfinite(torch.tensor(tr_loss)):
            print(f"[ERROR] NaN/Inf detected in training loss at epoch {epoch}!")
            print("This usually means:")
            print("  1. Learning rate is too high")
            print("  2. Data normalization is incompatible with loss function")
            print("  3. Gradient explosion (check your architecture)")
            raise ValueError(f"Training diverged with loss={tr_loss}")

        model.eval()
        va_loss = 0.0
        va_recon = 0.0 if cfg.vae else None
        va_kl = 0.0 if cfg.vae else None
        
        with torch.no_grad():
            for x, _ in dl_val:
                x = x.to(device)
                if cfg.vae:
                    x_logits, z, mu, logvar = model(x)
                    recon = criterion(x_logits, x)
                    kl = vae_kl(mu, logvar)
                    loss = recon + cfg.beta_kl * kl
                    va_recon += recon.item() * x.size(0)
                    va_kl += kl.item() * x.size(0)
                else:
                    x_logits, z = model(x)
                    loss = criterion(x_logits, x)
                va_loss += loss.item() * x.size(0)
        va_loss /= len(dl_val.dataset)
        if cfg.vae:
            va_recon /= len(dl_val.dataset)
            va_kl /= len(dl_val.dataset)

        # Log to W&B
        log_dict = {
            "phase1/epoch": epoch,
            "phase1/train_loss": tr_loss,
            "phase1/val_loss": va_loss,
        }
        if cfg.vae:
            log_dict.update({
                "phase1/train_recon": tr_recon,
                "phase1/train_kl": tr_kl,
                "phase1/val_recon": va_recon,
                "phase1/val_kl": va_kl,
            })
        
        # Log sample reconstructions every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                x_sample, _ = next(iter(dl_val))
                x_sample = x_sample[:8].to(device)
                if cfg.vae:
                    x_recon_logits, _, _, _ = model(x_sample)
                else:
                    x_recon_logits, _ = model(x_sample)
                x_recon = torch.sigmoid(x_recon_logits)
                
                # Create comparison image
                comparison = torch.cat([x_sample.cpu(), x_recon.cpu()], dim=0)
                if WANDB_AVAILABLE and wandb.run is not None:
                    log_dict["phase1/reconstructions"] = wandb.Image(
                        comparison,
                        caption=f"Epoch {epoch}: Top=Original, Bottom=Reconstructed"
                    )
        
        safe_wandb_log(log_dict)
        
        history.append({"epoch":epoch, "train_loss":tr_loss, "val_loss":va_loss})
        print(f"[P1][{epoch:02d}] train {tr_loss:.4f}  val {va_loss:.4f}  lr {scheduler.get_last_lr()[0]:.2e}")

        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            patience = cfg.patience
            torch.save(model.state_dict(), os.path.join(out_dir, "phase1_best.pt"))
        else:
            patience -= 1
            if patience <= 0:
                print("[P1] Early stopping.")
                break
        
        # Step the learning rate scheduler
        scheduler.step()

    # load best
    model.load_state_dict(torch.load(os.path.join(out_dir, "phase1_best.pt"), map_location=device))
    
    # Log best model as artifact
    if WANDB_AVAILABLE and wandb.run is not None:
        artifact = wandb.Artifact(name=f"phase1_model", type="model")
        artifact.add_file(os.path.join(out_dir, "phase1_best.pt"))
        safe_wandb_log_artifact(artifact)
    
    history_path = os.path.join(out_dir, "phase1_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[P1] Saved history: {history_path}")
    
    safe_wandb_log({"phase1/best_val_loss": best_val})
    return model

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


def train_phase3_refiner(encoder: ConvEncoder, head: ClassifierHead,
                         dl_train_for_metrics, dl_val, dl_test,
                         dl_reftrain,  # NEW: loader used to train refiner
                         cfg: Phase3Config, device, out_dir):

    # --- NO-OP when epochs==0: write minimal artifacts and return None ---
    if getattr(cfg, "epochs", 0) <= 0:
        # Evaluate t=0 only (no refinement)
        def eval_t0(dl):
            encoder.eval(); head.eval()
            import torch
            all_logits, all_y = [], []
            with torch.no_grad():
                for x, y in dl:
                    x, y = x.to(device), y.to(device)
                    z0 = encoder.encode_deterministic(x)
                    logits = head(z0)
                    all_logits.append(logits.cpu())
                    all_y.append(y.cpu())
            logits = torch.cat(all_logits, dim=0)
            y = torch.cat(all_y, dim=0)
            acc = (logits.argmax(-1) == y).float().mean().item()
            err = 1.0 - acc
            return acc, err

        tr_acc, tr_err = eval_t0(dl_train_for_metrics)
        te_acc, te_err = eval_t0(dl_test)

        # Minimal summary & T-sweep (T_eval=0 rows only)
        with open(os.path.join(out_dir, "phase3_summary.json"), "w") as f:
            json.dump({
                "skipped": True,
                "train_acc_t0": tr_acc,
                "test_acc_t0": te_acc,
                "gap_t0": te_err - tr_err
            }, f, indent=2)

        with open(os.path.join(out_dir, "phase3_Tsweep.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["T_eval", "split", "acc", "err", "gap"])
            wr.writerow([0, "train", f"{tr_acc:.6f}", f"{tr_err:.6f}", ""])
            wr.writerow([0, "test",  f"{te_acc:.6f}", f"{te_err:.6f}", f"{(te_err - tr_err):.6f}"])

        print("[P3] epochs=0 → skipped Phase 3. Wrote phase3_summary.json and phase3_Tsweep.csv (T=0 only).")
        return None



    # freeze encoder
    for p in encoder.parameters(): p.requires_grad = False
    encoder.eval()

    if cfg.mode == "fixed":
        refiner = RefinerFixed(dim=head.net[0].normalized_shape[0], hidden=cfg.ref_hidden).to(device)
    elif cfg.mode == "ponder":
        refiner = RefinerPonder(dim=head.net[0].normalized_shape[0], hidden=cfg.ref_hidden).to(device)
    else:
        raise ValueError("cfg.mode must be 'fixed' or 'ponder'")

    # optionally fine-tune head (off by default to isolate the effect of z refinement)
    for p in head.parameters(): p.requires_grad = cfg.finetune_head
    params = list(refiner.parameters()) + (list(head.parameters()) if cfg.finetune_head else [])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)

    #ce = nn.CrossEntropyLoss()
    ce = nn.CrossEntropyLoss(label_smoothing=getattr(cfg, "label_smooth", 0.0))
    best_val = float("inf")
    patience = cfg.patience
    history = []

    prior = geometric_prior(cfg.T, cfg.ponder_prior_lambda, device) if cfg.mode == "ponder" else None

    def forward_steps(z0):
        if cfg.mode == "fixed":
            zs = refiner(z0, T=cfg.T)  # list len T+1
            return zs, None
        else:
            zs, q = refiner(z0, T=cfg.T)  # zs len T+1, q [B,T]
            return zs, q

    def predict_logits(zs, q=None):
        """Return logits at t=0 and refined prediction (fixed: at T; ponder: mixture over steps)."""
        logits_t0 = head(zs[0])
        if cfg.mode == "fixed":
            logits_ref = head(zs[-1])
        else:
            # expectation over halting q: mix probabilities, not logits
            # safer: mix log-probs via log-sum-exp weighted, but we’ll mix probs:
            probs = []
            for t in range(1, len(zs)):  # steps 1..T
                logits_t = head(zs[t])
                probs_t = torch.softmax(logits_t, dim=-1)  # [B,C]
                probs.append(probs_t)
            probs = torch.stack(probs, dim=1)  # [B,T,C]
            q_expanded = q.unsqueeze(-1)       # [B,T,1]
            probs_mix = (q_expanded * probs).sum(dim=1)  # [B,C]
            logits_ref = torch.log(probs_mix.clamp(1e-8))  # log-probs for CE compatibility
        return logits_t0, logits_ref

    def ponder_kl(q, prior):
        # q: [B,T], prior: [T]
        q = q.clamp(1e-8, 1.0)
        p = prior.unsqueeze(0)  # [1,T]
        kl = (q * (q.log() - p.log())).sum(dim=1)  # per-sample
        return kl.mean()

    def run_epoch(dl, train=False):
        if train:
            refiner.train()
            if cfg.finetune_head: head.train()
            else: head.eval()
        else:
            refiner.eval()
            head.eval()
        total_loss, total_val = 0.0, 0.0
        all_logits_t0, all_logits_ref, all_y = [], [], []
        with torch.set_grad_enabled(train):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    z0 = encoder.encode_deterministic(x)  # [B, D]
                if train and getattr(cfg, "znoise", 0.0) > 0.0:
                    # small Gaussian noise encourages smoother refinements and narrows the gap
                    z0 = z0 + cfg.znoise * torch.randn_like(z0)
                zs, q = forward_steps(z0)
                logits_t0, logits_ref = predict_logits(zs, q)

                # training loss only uses refined prediction to push refiner;
                # evaluation reports both t=0 and refined numbers.
                if train:
                    # base loss on refined prediction
                    if cfg.mode == "fixed":
                        loss = ce(logits_ref, y)
                    else:
                        # use CE on mixed log-probs + KL to geometric prior
                        ce_loss = ce(logits_ref, y)
                        kl = ponder_kl(q, prior)
                        loss = ce_loss + cfg.ponder_beta * kl

                    # ---- Contractive step penalty: encourage small updates z_{t+1}-z_t ----
                    lam = getattr(cfg, "contract", 0.0)
                    if lam > 0.0:
                        # zs is [z0, z1, ..., zT]
                        deltas = [zs[t+1] - zs[t] for t in range(len(zs)-1)]
                        if cfg.mode == "fixed" or q is None:
                            # uniform across steps
                            step_pen = sum((d**2).mean() for d in deltas) / max(1, len(deltas))
                        else:
                            # weight by expected halting distribution q over steps 1..T
                            # broadcast q to match batch, penalise larger steps more if they are likely used
                            # q: [B,T]; deltas[t]: [B,D]
                            step_pen = 0.0
                            for t, d in enumerate(deltas):  # t=0..T-1 corresponds to step t+1
                                w = q[:, t].detach().unsqueeze(1)  # [B,1], stop gradient on weights
                                step_pen = step_pen + ((w * (d**2)).mean())
                            step_pen = step_pen / max(1, len(deltas))
                        loss = loss + lam * step_pen
                    # ----------------------------------------------------------------------
                        
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, 5.0)
                    opt.step()
                    total_loss += loss.item() * x.size(0)
                else:
                    # in eval we also accumulate a "val loss" on refined prediction
                    if cfg.mode == "fixed":
                        loss = ce(logits_ref, y)
                    else:
                        ce_loss = ce(logits_ref, y)
                        kl = ponder_kl(q, prior)
                        loss = ce_loss + cfg.ponder_beta * kl
                    total_loss += loss.item() * x.size(0)

                all_logits_t0.append(logits_t0.detach())
                all_logits_ref.append(logits_ref.detach())
                all_y.append(y.detach())

        total_loss /= len(dl.dataset)
        logits_t0 = torch.cat(all_logits_t0, dim=0)
        logits_ref = torch.cat(all_logits_ref, dim=0)
        y = torch.cat(all_y, dim=0)

        acc_t0  = accuracy(logits_t0, y)
        err_t0  = 1.0 - acc_t0
        acc_ref = accuracy(logits_ref, y)
        err_ref = 1.0 - acc_ref
        return total_loss, acc_t0, err_t0, acc_ref, err_ref

    for epoch in range(1, cfg.epochs+1):
        tr_loss, tr_acc0, tr_err0, tr_accr, tr_errr = run_epoch(dl_reftrain, train=True)
        va_loss, va_acc0, va_err0, va_accr, va_errr = run_epoch(dl_val,   train=False)
        
        # Log to W&B
        safe_wandb_log({
            "phase3/epoch": epoch,
            "phase3/train_loss": tr_loss,
            "phase3/train_acc_t0": tr_acc0,
            "phase3/train_err_t0": tr_err0,
            "phase3/train_acc_refined": tr_accr,
            "phase3/train_err_refined": tr_errr,
            "phase3/val_loss": va_loss,
            "phase3/val_acc_t0": va_acc0,
            "phase3/val_err_t0": va_err0,
            "phase3/val_acc_refined": va_accr,
            "phase3/val_err_refined": va_errr,
        })
        
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_acc_t0": tr_acc0, "train_acc_ref": tr_accr,
            "val_acc_t0": va_acc0,   "val_acc_ref":  va_accr
        })
        print(f"[P3][{epoch:02d}] train loss {tr_loss:.4f} acc0 {tr_acc0:.3f} accR {tr_accr:.3f} | "
              f"val loss {va_loss:.4f} acc0 {va_acc0:.3f} accR {va_accr:.3f}")

        # early stop on val refined loss
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            patience = cfg.patience
            torch.save(refiner.state_dict(), os.path.join(out_dir, "phase3_refiner_best.pt"))
            if cfg.finetune_head:
                torch.save(head.state_dict(), os.path.join(out_dir, "phase3_head_best.pt"))
        else:
            patience -= 1
            if patience <= 0:
                print("[P3] Early stopping.")
                break

    # load best
    

    #refiner.load_state_dict(torch.load(os.path.join(out_dir, "phase3_refiner_best.pt"), map_location=device))
    best_path = os.path.join(out_dir, "phase3_refiner_best.pt")
    if os.path.exists(best_path):
        refiner.load_state_dict(torch.load(best_path, map_location=device))

    if cfg.finetune_head and os.path.exists(os.path.join(out_dir, "phase3_head_best.pt")):
        head.load_state_dict(torch.load(os.path.join(out_dir, "phase3_head_best.pt"), map_location=device))

    # final report
    # 'Train' metrics measured on the head's training split (A), not the refiner's training data (B)
    tr_loss, tr_acc0, tr_err0, tr_accr, tr_errr = run_epoch(dl_train_for_metrics, train=False)
    te_loss, te_acc0, te_err0, te_accr, te_errr = run_epoch(dl_test,  train=False)
    gap0 = te_err0 - tr_err0
    gapr = te_errr - tr_errr
    print(f"[P3] t=0:    train acc {tr_acc0:.3f} err {tr_err0:.3f} | test acc {te_acc0:.3f} err {te_err0:.3f} | gap {gap0:.3f}")
    print(f"[P3] refine: train acc {tr_accr:.3f} err {tr_errr:.3f} | test acc {te_accr:.3f} err {te_errr:.3f} | gap {gapr:.3f}")

    # Log final Phase 3 metrics to W&B
    safe_wandb_log({
        "phase3/final_train_acc_t0": tr_acc0,
        "phase3/final_train_err_t0": tr_err0,
        "phase3/final_test_acc_t0": te_acc0,
        "phase3/final_test_err_t0": te_err0,
        "phase3/final_gen_gap_t0": gap0,
        "phase3/final_train_acc_refined": tr_accr,
        "phase3/final_train_err_refined": tr_errr,
        "phase3/final_test_acc_refined": te_accr,
        "phase3/final_test_err_refined": te_errr,
        "phase3/final_gen_gap_refined": gapr,
    })
    
    # Log refiner as artifact
    if WANDB_AVAILABLE and wandb.run is not None and os.path.exists(best_path):
        artifact = wandb.Artifact(name=f"phase3_refiner", type="model")
        artifact.add_file(best_path)
        wandb.log_artifact(artifact)

    summary_path = os.path.join(out_dir, "phase3_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "train_acc_t0": tr_acc0, "train_err_t0": tr_err0,
            "test_acc_t0": te_acc0,  "test_err_t0": te_err0,  "gen_gap_t0": gap0,
            "train_acc_ref": tr_accr,"train_err_ref": tr_errr,
            "test_acc_ref": te_accr, "test_err_ref": te_errr, "gen_gap_ref": gapr
        }, f, indent=2)
    print(f"[P3] Saved summary: {summary_path}")
    
    history_path = os.path.join(out_dir, "phase3_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[P3] Saved history: {history_path}")


    # --- Post-training: sweep T_eval = 0..T and dump metrics (acc + gap) ---
    import csv

    def _eval_with_T(T_eval:int, dl):
        refiner.eval(); head.eval()
        all_logits, all_y = [], []
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                z0 = encoder.encode_deterministic(x)
                if cfg.mode == "fixed":
                    # Use exactly T_eval refinement steps
                    zs = refiner(z0, T=T_eval)         # list len T_eval+1
                    z_use = zs[-1]
                    logits = head(z_use)
                else:
                    # Ponder: approximate by mixing the first T_eval steps, renormalised
                    zs, q = refiner(z0, T=cfg.T)       # zs len T+1, q [B,T]
                    if T_eval == 0:
                        logits = head(zs[0])
                    else:
                        probs = []
                        for t in range(1, T_eval+1):
                            probs_t = torch.softmax(head(zs[t]), dim=-1)
                            probs.append(probs_t)
                        probs = torch.stack(probs, dim=1)  # [B, T_eval, C]
                        q_cut = q[:, :T_eval]
                        q_cut = q_cut / q_cut.sum(dim=1, keepdim=True).clamp(1e-8)
                        probs_mix = (q_cut.unsqueeze(-1) * probs).sum(dim=1)
                        logits = torch.log(probs_mix.clamp(1e-8))
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
        logits = torch.cat(all_logits, dim=0)
        y = torch.cat(all_y, dim=0)
        acc = (logits.argmax(-1) == y).float().mean().item()
        err = 1.0 - acc
        return acc, err

    tsweep_csv = os.path.join(out_dir, "phase3_Tsweep.csv")
    tsweep_data = []
    with open(tsweep_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["T_eval", "split", "acc", "err", "gap"])
        for T_eval in range(0, cfg.T+1):
            tr_acc, tr_err = _eval_with_T(T_eval, dl_train_for_metrics)
            te_acc, te_err = _eval_with_T(T_eval, dl_test)
            gap = te_err - tr_err
            wr.writerow([T_eval, "train", f"{tr_acc:.6f}", f"{tr_err:.6f}", ""])
            wr.writerow([T_eval, "test",  f"{te_acc:.6f}", f"{te_err:.6f}", f"{gap:.6f}"])
            
            # Store for W&B logging
            tsweep_data.append({
                "T_eval": T_eval,
                "train_acc": tr_acc,
                "train_err": tr_err,
                "test_acc": te_acc,
                "test_err": te_err,
                "gen_gap": gap
            })
    print(f"[P3] Saved T-sweep CSV: {tsweep_csv}")
    print(f"[P3] Saved T-sweep to {tsweep_csv}")
    
    # Log T-sweep results to W&B as a table
    if WANDB_AVAILABLE and wandb.run is not None:
        tsweep_table = wandb.Table(
            columns=["T_eval", "train_acc", "train_err", "test_acc", "test_err", "gen_gap"],
            data=[[d["T_eval"], d["train_acc"], d["train_err"], d["test_acc"], d["test_err"], d["gen_gap"]] 
                  for d in tsweep_data]
        )
        safe_wandb_log({"phase3/T_sweep": tsweep_table})
    
    # Also log as line plots
    for d in tsweep_data:
        safe_wandb_log({
            f"phase3/tsweep_T_{d['T_eval']}/train_err": d["train_err"],
            f"phase3/tsweep_T_{d['T_eval']}/test_err": d["test_err"],
            f"phase3/tsweep_T_{d['T_eval']}/gen_gap": d["gen_gap"],
        })
    
    # Log T-sweep CSV as artifact
    if WANDB_AVAILABLE and wandb.run is not None:
        artifact = wandb.Artifact(name=f"phase3_tsweep", type="results")
        artifact.add_file(tsweep_csv)
        wandb.log_artifact(artifact)

    return refiner

# --------------- CLI ---------------

def main():
    ap = argparse.ArgumentParser(description="AE/VAE -> Head -> Refiner (Fixed/Ponder) pipeline")
    ap.add_argument("--dataset", choices=["mnist","fashion","cifar10","svhn","emnist"], default="mnist")
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="./runs/exp1")
    ap.add_argument("--vae", action="store_true")
    # Phase1
    ap.add_argument("--p1_epochs", type=int, default=20)
    ap.add_argument("--p1_lr", type=float, default=1e-3)
    ap.add_argument("--p1_wd", type=float, default=1e-4)
    ap.add_argument("--p1_patience", type=int, default=5)
    ap.add_argument("--p1_beta_kl", type=float, default=1.0)
    # Phase2
    ap.add_argument("--p2_epochs", type=int, default=15)
    ap.add_argument("--p2_lr", type=float, default=5e-4)
    ap.add_argument("--p2_wd", type=float, default=5e-4)
    ap.add_argument("--p2_patience", type=int, default=5)
    ap.add_argument("--p2_head_hidden", type=int, default=128)
    ap.add_argument("--p2_dropout", type=float, default=0.1)
    # Phase3
    ap.add_argument("--mode", choices=["fixed","ponder"], default="fixed")
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--p3_epochs", type=int, default=20)
    ap.add_argument("--p3_lr", type=float, default=5e-4)
    ap.add_argument("--p3_wd", type=float, default=5e-4)
    ap.add_argument("--p3_patience", type=int, default=6)
    ap.add_argument("--p3_ref_hidden", type=int, default=256)
    ap.add_argument("--p3_finetune_head", action="store_true")
    ap.add_argument("--ponder_lambda", type=float, default=0.3)
    ap.add_argument("--ponder_beta", type=float, default=0.01)
    ap.add_argument("--p3_znoise", type=float, default=0.05)
    ap.add_argument("--p3_contract", type=float, default=0.0, help="λ for contractive step penalty on refiner (e.g., 1e-3)")
    ap.add_argument("--p3_label_smooth", type=float, default=0.05, help="label smoothing for CE in phase 3")
    ap.add_argument("--p3_crossfit_ratio", type=float, default=0.5,
                    help="fraction of train data reserved to train refiner in Phase 3 (rest used for head/gap). Set 0 to disable.")
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
    
    if use_wandb:
        # Auto-generate group name for multi-seed experiments if not provided
        if args.wandb_group is None:
            # Group name: dataset_mode_latent (without seed)
            args.wandb_group = f"{args.dataset}_{args.mode}_lat{args.latent}"
            if args.vae:
                args.wandb_group += "_vae"
        
        # Parse tags
        tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
        tags.extend([args.dataset, args.mode, "vae" if args.vae else "ae"])
        
        # Create run name: includes seed for identification
        run_name = f"{args.dataset}_{args.mode}_s{args.seed}"
        
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

    # Phase 1: AE/VAE
    p1cfg = Phase1Config(
        epochs=args.p1_epochs, lr=args.p1_lr, wd=args.p1_wd,
        patience=args.p1_patience, vae=args.vae, beta_kl=args.p1_beta_kl
    )
    ae = Autoencoder(in_ch=in_ch, latent=args.latent, vae=args.vae, img_size=img_size)
    n_params = count_params(ae)
    enc_params = count_params(ae.encoder)
    dec_params = count_params(ae.decoder)
    print(f"[P1] Model params: AE/VAE {n_params:,} (encoder: {enc_params:,}, decoder: {dec_params:,})")
    print(f"[P1] Architecture: {in_ch}ch {img_size}x{img_size} -> latent {args.latent}")
    compression_ratio = (in_ch * img_size * img_size) / args.latent
    print(f"[P1] Compression ratio: {compression_ratio:.1f}:1")
    ae = train_phase1_autoencoder(ae, dl_train, dl_val, p1cfg, device, args.out)

    # Phase 2: Head
    p2cfg = Phase2Config(
        epochs=args.p2_epochs, lr=args.p2_lr, wd=args.p2_wd, patience=args.p2_patience,
        freeze_encoder=True, head_hidden=args.p2_head_hidden, dropout=args.p2_dropout
    )
    head = ClassifierHead(in_dim=args.latent, n_classes=n_classes,
                          hidden=args.p2_head_hidden, p_drop=args.p2_dropout).to(device)
    #head = train_phase2_head(ae.encoder, head, dl_train, dl_val, dl_test, p2cfg, device, args.out)
    head = train_phase2_head(ae.encoder, head, dlA, dl_val, dl_test, p2cfg, device, args.out)


    # Phase 3: Refiner
    p3cfg = Phase3Config(
        mode=args.mode, T=args.T, lr=args.p3_lr, wd=args.p3_wd, epochs=args.p3_epochs,
        patience=args.p3_patience, ref_hidden=args.p3_ref_hidden,
        finetune_head=args.p3_finetune_head,
        ponder_prior_lambda=args.ponder_lambda, ponder_beta=args.ponder_beta
    )
    p3cfg.znoise = args.p3_znoise  # attach dynamically
    p3cfg.contract = args.p3_contract
    p3cfg.label_smooth = args.p3_label_smooth
    _ = train_phase3_refiner(ae.encoder, head,
                            dlA_eval, dl_val, dl_test,
                            dlB,  # this is the refiner's private training data
                            p3cfg, device, args.out)
    
    # Finish W&B run
    if WANDB_AVAILABLE and wandb.run is not None and not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
