#!/usr/bin/env bash
# Lightweight repository quality checks that do not require GPU dependencies.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Checking core files..."
for path in README.md requirements.txt LICENSE scripts/run_figure2.sh scripts/run_figure3.sh scripts/run_figure4.sh scripts/run_figure5.sh scripts/run_figureS1.sh src/ae_experiment.py src/lpc_experiment.py src/kv_motion_figure.py; do
    [[ -f "$path" ]] || { echo "Missing required file: $path"; exit 1; }
done

echo "[2/4] Validating shell scripts parse..."
for script in scripts/*.sh; do
    bash -n "$script"
done

echo "[3/4] Python syntax check (compileall)..."
python -m compileall -q .

echo "[4/4] README command consistency spot-check..."
# Ensure Figure 4 command matches the actual output naming in scripts/run_figure4.sh.
rg -q "capacity_cifar10_lat\\$\{lat\}_s124" README.md

# Ensure Figure 2/S1 compute note reports the correct model counts.
rg -q "Figures 2 and S1 together involve training 120 models" README.md

echo "All repository checks passed."
