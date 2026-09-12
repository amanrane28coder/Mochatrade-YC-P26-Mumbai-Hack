#!/bin/bash
set -e  # Exit on any error

echo "=== Building GPU-Accelerated Backtesting Engine (CUDA) ==="

# Check if we are in the project directory (adjust as needed)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"

# Check for nvcc (CUDA compiler)
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA Toolkit and ensure nvcc is in your PATH."
    exit 1
fi

# Show CUDA version
nvcc --version

# Create build directory
BUILD_DIR="$PROJECT_DIR/build_cuda"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure and build with CUDA enabled
echo "Configuring with CMake (USE_CUDA=ON)..."
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release

echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build successful!"

# Run the test suite
echo "Running tests..."
cd "$PROJECT_DIR"
python3 tests/test_engine.py

echo "Tests passed!"

echo "=== Build and test completed successfully ==="