#!/usr/bin/env bash
# build_apptainer.sh - Build the Aircraft-Delay-Prediction Apptainer container locally

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${SCRIPT_DIR}/apptainer.def"
OUTPUT_DIR="${SCRIPT_DIR}/dist"
OUTPUT_FILE="${OUTPUT_DIR}/aircraft-delay.sif"
TORCH_VER="${TORCH_VER:-2.8.0}"
CUDA_VER="${CUDA_VER:-cu129}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check prerequisites
check_apptainer() {
    if ! command -v apptainer &> /dev/null; then
        error "apptainer is not installed. Please install apptainer first."
    fi
    info "apptainer found: $(apptainer --version)"
}

check_files() {
    if [ ! -f "$DEF_FILE" ]; then
        error "apptainer.def not found at $DEF_FILE"
    fi
    if [ ! -f "$SCRIPT_DIR/setup_env.sh" ]; then
        error "setup_env.sh not found at $SCRIPT_DIR/setup_env.sh"
    fi
    info "Required files found"
}

# Main build function
build_sif() {
    info "Building Apptainer SIF image..."
    info "  Definition: $DEF_FILE"
    info "  Output: $OUTPUT_FILE"
    info "  PyTorch version: $TORCH_VER"
    info "  CUDA version: $CUDA_VER"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Remove existing SIF if present
    if [ -f "$OUTPUT_FILE" ]; then
        warn "Overwriting existing SIF at $OUTPUT_FILE"
        rm -f "$OUTPUT_FILE"
    fi
    
    # Build the container
    TORCH_VER="$TORCH_VER" CUDA_VER="$CUDA_VER" \
        apptainer build "$OUTPUT_FILE" "$DEF_FILE"
    
    if [ $? -eq 0 ]; then
        info "Build successful!"
        info "SIF image saved to: $OUTPUT_FILE"
        ls -lh "$OUTPUT_FILE"
    else
        error "Build failed!"
    fi
}

# Display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -t, --torch-ver VERSION    PyTorch version (default: 2.8.0)
    -c, --cuda-ver VERSION     CUDA version (default: cu129, use 'cpu' for CPU-only)
    -o, --output FILE          Output SIF file path (default: dist/aircraft-delay.sif)
    -h, --help                 Show this help message

Examples:
    # Build with defaults (PyTorch 2.8.0, CUDA 12.9)
    ./build_apptainer.sh

    # Build for CPU-only
    ./build_apptainer.sh --cuda-ver cpu

    # Build with specific versions
    ./build_apptainer.sh --torch-ver 2.7.0 --cuda-ver cu118

    # Custom output location
    ./build_apptainer.sh --output /tmp/my-container.sif

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--torch-ver)
            TORCH_VER="$2"
            shift 2
            ;;
        -c|--cuda-ver)
            CUDA_VER="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Main execution
info "Aircraft-Delay-Prediction Container Builder"
check_apptainer
check_files
build_sif

info "Done!"
