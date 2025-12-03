import os
import subprocess
import venv
import argparse

def run(cmd, description=None):
    """Run a shell command with live output and optional description."""
    if description:
        print(f"\n🔹 {description}")
    print(f">>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # --- Argument parser ---
    parser = argparse.ArgumentParser(description="Create venv and install PyTorch + PyG + DeepSNAP + utilities")
    parser.add_argument("--venv-dir", default="venv", help="Folder where venv will be created")
    parser.add_argument("--torch-ver", default="2.8.0", help="Torch version (e.g. 2.8.0)")
    parser.add_argument("--cuda-ver", default="cu129", help="CUDA version (e.g. cpu, cu118, cu129)")
    # While torch supports ROCm now, PyG does not. 
    # In the event that it gets support you can set it to something like rocm6.4 or whichever is the latest version
    args = parser.parse_args()

    VENV_DIR = args.venv_dir
    TORCH_VER = args.torch_ver
    CUDA_VER = args.cuda_ver

    print("\n🚀 Starting environment setup...")
    print(f"   • Virtual environment directory: {VENV_DIR}")
    print(f"   • Torch version: {TORCH_VER}")
    print(f"   • CUDA version: {CUDA_VER}")

    # 1. Create venv if not exists
    if not os.path.exists(VENV_DIR):
        print(f"\n📦 Step 1: Creating virtual environment in '{VENV_DIR}'...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    else:
        print(f"\n📦 Step 1: Virtual environment '{VENV_DIR}' already exists, skipping creation.")

    # 2. Path to pip inside venv (cross-platform)
    if os.name == "nt":  # Windows
        python_path = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:  # Linux/macOS
        python_path = os.path.join(VENV_DIR, "bin", "python")

    print(f"\n📂 Step 2: Using Python interpreter at: {python_path}")

    # 3. Upgrade pip
    run(f"{python_path} -m pip install --upgrade pip", description="Step 3: Upgrading pip to latest version")

    # 4. Install PyTorch core package
    run(
        f"{python_path} -m pip install torch=={TORCH_VER} "
        f"-f https://download.pytorch.org/whl/{CUDA_VER}",
        description="Step 4: Installing PyTorch core package"
    )

    # 5. Install PyTorch Geometric extensions
    run(
        f"{python_path} -m pip install pyg_lib torch_scatter torch_sparse "
        f"torch_cluster torch_spline_conv "
        f"-f https://data.pyg.org/whl/torch-{TORCH_VER}+{CUDA_VER}.html",
        description="Step 5: Installing PyTorch Geometric extensions"
    )

    # 6. Install torch-geometric meta package
    run(f"{python_path} -m pip install torch-geometric", description="Step 6: Installing torch-geometric meta package")

    # 7. Install DeepSNAP from GitHub
    run(f"{python_path} -m pip install git+https://github.com/snap-stanford/deepsnap.git",
        description="Step 7: Installing DeepSNAP from GitHub")

    # 8. Other libraries
    run(f"{python_path} -m pip install pandas tqdm colorama requests beautifulsoup4 scikit-learn",
        description="Step 8: Installing utility libraries (pandas, tqdm, colorama, requests, bs4, scikit-learn)")

    print("\n✅ Environment setup complete!")
    print(f"Activate your venv with:\n  source {VENV_DIR}/bin/activate   (Linux/macOS)\n  {VENV_DIR}\\Scripts\\activate    (Windows)")

if __name__ == "__main__":
    main()
