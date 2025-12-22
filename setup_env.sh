#!/usr/bin/env sh
set -eu
# setup_env.sh - Install Python packages system-wide (intended for container build)
# Usage: set TORCH_VER and CUDA_VER via env (defaults below), then run this script.

# Defaults (matching setup_env.py)
TORCH_VER="${TORCH_VER:-2.8.0}"
CUDA_VER="${CUDA_VER:-cu129}"
PYTHON_BIN="python3"

echo "Starting setup_env.sh"
echo " - Torch: ${TORCH_VER}"
echo " - CUDA tag: ${CUDA_VER}"

echo "Upgrading pip, setuptools, wheel..."
$PYTHON_BIN -m pip install --break-system-packages --upgrade pip setuptools wheel

echo "Installing PyTorch (${TORCH_VER}, ${CUDA_VER})..."
$PYTHON_BIN -m pip install --break-system-packages "torch==${TORCH_VER}" --index-url "https://download.pytorch.org/whl/${CUDA_VER}"

echo "Installing PyG extension wheels (pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv)..."
$PYTHON_BIN -m pip install --break-system-packages pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html"

echo "Installing torch-geometric..."
$PYTHON_BIN -m pip install --break-system-packages torch-geometric

echo "Installing DeepSNAP from GitHub..."
$PYTHON_BIN -m pip install --break-system-packages git+https://github.com/snap-stanford/deepsnap.git

echo "Installing utility libraries..."
$PYTHON_BIN -m pip install --break-system-packages pandas tqdm colorama requests beautifulsoup4 scikit-learn pyyaml

echo "Cleaning pip cache (best-effort)..."
${PYTHON_BIN} -m pip cache purge || true

echo "setup_env.sh finished."
