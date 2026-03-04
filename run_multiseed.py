#!/usr/bin/env python3
"""
Helper script to run multi-seed experiments with W&B integration.
Automatically groups runs by configuration for easy comparison.
"""

import argparse
import subprocess
import os

def run_experiment_with_seeds(
    dataset: str,
    seeds: list,
    mode: str = "fixed",
    latent: int = 32,
    T: int = 3,
    vae: bool = False,
    output_base: str = "./runs",
    wandb_project: str = "consolidation-experiments",
    wandb_group: str = None,
    extra_args: list = None
):
    """
    Run the same experiment configuration with multiple seeds.
    All runs will be grouped together in W&B for easy comparison.
    
    Args:
        dataset: Dataset name (mnist, fashion, cifar10)
        seeds: List of random seeds
        mode: Refiner mode (fixed or ponder)
        latent: Latent dimension
        T: Number of refinement steps
        vae: Whether to use VAE instead of AE
        output_base: Base directory for outputs
        wandb_project: W&B project name
        wandb_group: W&B group name (auto-generated if None)
        extra_args: Additional command-line arguments as list
    """
    
    # Auto-generate group name if not provided
    if wandb_group is None:
        wandb_group = f"{dataset}_{mode}_lat{latent}"
        if vae:
            wandb_group += "_vae"
    
    print(f"="*60)
    print(f"Running multi-seed experiment")
    print(f"  Dataset: {dataset}")
    print(f"  Seeds: {seeds}")
    print(f"  Mode: {mode}")
    print(f"  Latent: {latent}")
    print(f"  T: {T}")
    print(f"  VAE: {vae}")
    print(f"  W&B Group: {wandb_group}")
    print(f"="*60)
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Starting run with seed {seed}")
        print(f"{'='*60}\n")
        
        # Build output directory
        run_name = f"{dataset}_{mode}_s{seed}"
        if vae:
            run_name += "_vae"
        output_dir = os.path.join(output_base, run_name)
        
        # Build command
        cmd = [
            "python", "ae_experiment.py",
            "--dataset", dataset,
            "--seed", str(seed),
            "--mode", mode,
            "--latent", str(latent),
            "--T", str(T),
            "--out", output_dir,
            "--wandb_project", wandb_project,
            "--wandb_group", wandb_group,
        ]
        
        if vae:
            cmd.append("--vae")
        
        if extra_args:
            cmd.extend(extra_args)
        
        # Run experiment
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n✗ Run with seed {seed} failed!")
            print(f"  Consider checking the output directory: {output_dir}")
        else:
            print(f"\n✓ Run with seed {seed} completed successfully!")
    
    print(f"\n{'='*60}")
    print(f"All runs completed!")
    print(f"View results in W&B: https://wandb.ai/{wandb_project}")
    print(f"Group: {wandb_group}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed experiments with W&B tracking"
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["mnist", "fashion", "cifar10"],
                       help="Dataset to use")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in range(121, 141)),
                       help="Comma-separated list of seeds (default: 121-140)")
    parser.add_argument("--mode", type=str, default="fixed",
                       choices=["fixed", "ponder"],
                       help="Refiner mode")
    parser.add_argument("--latent", type=int, default=32,
                       help="Latent dimension")
    parser.add_argument("--T", type=int, default=3,
                       help="Number of refinement steps")
    parser.add_argument("--vae", action="store_true",
                       help="Use VAE instead of AE")
    parser.add_argument("--output_base", type=str, default="./runs",
                       help="Base directory for outputs")
    parser.add_argument("--wandb_project", type=str, default="consolidation-experiments",
                       help="W&B project name")
    parser.add_argument("--wandb_group", type=str, default=None,
                       help="W&B group name (auto-generated if not provided)")
    
    args, extra_args = parser.parse_known_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    run_experiment_with_seeds(
        dataset=args.dataset,
        seeds=seeds,
        mode=args.mode,
        latent=args.latent,
        T=args.T,
        vae=args.vae,
        output_base=args.output_base,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        extra_args=extra_args
    )

if __name__ == "__main__":
    main()
