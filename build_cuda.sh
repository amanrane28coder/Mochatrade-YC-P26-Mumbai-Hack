#!/bin/bash
# Cross-platform build script for GPU-accelerated backtesting engine
# Usage: ./build_cuda.sh [--release|--debug]

set -e  # Exit on any error

# Default to release build
BUILD_TYPE="Release"
CLEAN_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--release|--debug] [--clean]"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Determine number of CPU cores for parallel build
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CPU_COUNT=$(sysctl -n hw.ncpu)
else
    # Linux
    CPU_COUNT=$(nproc)
fi

BUILD_DIR="$PROJECT_ROOT/build_gpu"

echo "Building GPU-Accelerated Backtesting Engine"
echo "=========================================="
echo "Build type: $BUILD_TYPE"
echo "CPU cores: $CPU_COUNT"
echo "Project root: $PROJECT_ROOT"
echo ""

# Clean previous build if requested
if [[ "$CLEAN_BUILD" == true ]]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake with CUDA enabled
echo "Configuring CMake with CUDA support..."
cmake "$PROJECT_ROOT" \
    -DUSE_CUDA=ON \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# Build the project
echo "Building with $CPU_COUNT parallel jobs..."
make -j"$CPU_COUNT"

# Copy the resulting module to project root for easy access
echo "Copying gpu_engine.so to project root..."
cp gpu_engine.so "$PROJECT_ROOT/"

echo ""
echo "Build successful! 🚀"
echo "Module location: $PROJECT_ROOT/gpu_engine.so"
echo ""
echo "To run tests:"
echo "  cd $PROJECT_ROOT"
echo "  python3 tests/test_engine.py"
echo ""
echo "For debug build next time:"
echo "  ./build_cuda.sh --debug"
echo ""
echo "For clean rebuild:"
echo "  ./build_cuda.sh --clean"