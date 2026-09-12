# Testing GPU-Accelerated Backtesting Engine on Linux

## Overview
This document provides instructions for testing the GPU-accelerated version of the backtesting engine in a Linux environment with NVIDIA CUDA support.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (Compute Capability 3.0+ recommended)
- Minimum 4GB GPU memory
- Modern x86_64 CPU

### Software Requirements
- Linux distribution (Ubuntu 20.04/22.04, CentOS 7/8, RHEL 8, or Amazon Linux 2)
- NVIDIA CUDA Toolkit 11.0 or higher
- CMake 3.18+
- GCC 7+ or Clang 6+
- Python 3.7+
- NumPy
- Git

## Installation Guide

### 1. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-pip python3-dev
```

#### CentOS/RHEL:
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git python3-pip python3-devel
```

### 2. Install NVIDIA CUDA Toolkit

#### Ubuntu (Preferred Method):
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# Set up environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Manual Installation:
1. Download CUDA Toolkit from https://developer.nvidia.com/cuda-downloads
2. Run the installer: `sudo sh cuda_*.run`
3. Follow the installation wizard
4. Add CUDA to PATH and LD_LIBRARY_PATH as shown above

### 3. Verify Installation
```bash
nvcc --version          # Should show CUDA version
nvidia-smi              # Should show GPU and driver info
```

## Building the GPU Version

### 1. Get the Source Code
```bash
git clone <repository-url>
cd gpu_engine
```

### 2. Create GPU Build Directory
```bash
mkdir -p build_cuda && cd build_cuda
```

### 3. Configure and Build
```bash
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 4. Verify Build Output
Successful build will produce:
- `libgpu_engine_static.a` - Static library
- `gpu_engine.so` - Python extension module

## Running Tests

### 1. Install Python Dependencies
```bash
pip3 install numpy
```

### 2. Execute Test Suite
```bash
# From project root
python3 tests/test_engine.py
```

### 3. Expected Successful Output
```
Successfully imported gpu_engine module
Testing GPU-Accelerated Backtesting Engine
==================================================
Generating test data...
Data shape: 500 points
Creating FastQuantEngine...
Running parameter sweep...
Got 66 results
Result 0: P&L=-0.00, Sharpe=-0.00, DD=0.00%, Trades=2, Win%=50.0%
Result 1: P&L=0.00, Sharpe=0.00, DD=0.00%, Trades=2, Win%=100.0%
Result 2: P&L=-0.00, Sharpe=-0.00, DD=225782.40%, Trades=2, Win%=50.0%
Basic functionality test passed!

Testing edge cases...
Empty data test passed
Single point test: 12 results
Single point test completed

All tests completed!
```

## Performance Validation

### CPU vs GPU Comparison (Optional)
To verify both backends produce equivalent results:

1. **Build CPU version** (separate directory):
   ```bash
   mkdir build_cpu && cd build_cpu
   cmake .. -DUSE_CUDA=OFF
   make
   cp gpu_engine.so ../gpu_engine_cpu.so
   ```

2. **Build GPU version**:
   ```bash
   mkdir build_gpu && cd build_gpu
   cmake .. -DUSE_CUDA=ON
   make
   cp gpu_engine.so ../gpu_engine_gpu.so
   ```

3. **Run comparison script**:
   ```python
   import numpy as np
   import sys
   import os
   
   # Test both versions
   for version, libname in [("CPU", "gpu_engine_cpu.so"), ("GPU", "gpu_engine_gpu.so")]:
       sys.path.insert(0, os.getcwd())
       # Load the appropriate module...
       # Run identical tests and compare results within tolerance
   ```

## Troubleshooting

### Common Build Issues

1. **"nvcc: command not found"**
   - Solution: Ensure `/usr/local/cuda/bin` is in PATH
   - `export PATH=/usr/local/cuda/bin:$PATH`

2. **"cannot find -lcudart"**
   - Solution: Ensure `/usr/local/cuda/lib64` is in LD_LIBRARY_PATH
   - `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

3. **"unsupported gpu architecture"**
   - Solution: Specify compute capability explicitly
   - Add to cmake: `-DCMAKE_CUDA_ARCHITECTURES=75` (adjust for your GPU)

4. **Memory allocation failures**
   - Solution: Reduce test data size or close other GPU applications

### Runtime Issues

1. **"invalid device function"**
   - Indicates PTX incompatibility with GPU architecture
   - Rebuild with correct `-DCMAKE_CUDA_ARCHITECTURES`

2. **"illegal memory access"**
   - Usually indicates buffer overflow in kernel
   - Check array bounds in kernel code

3. **Performance not as expected**
   - Ensure GPU is in persistence mode: `sudo nvidia-smi -pm 1`
   - Monitor with `nvidia-smi dmon -s u`

## Verification Checklist

When testing the GPU version, verify:

1. [ ] Module imports successfully: `import gpu_engine`
2. [ ] Engine instantiation works with NumPy arrays
3. [ ] Parameter sweep returns correct number of results
4. [ ] Basic statistics (P&L, Sharpe, drawdown, etc.) are computed
5. [ ] Edge cases (empty data, single point) handled properly
6. [ ] Results are numerically consistent with CPU implementation (within floating-point tolerance)
7. [ ] Performance scales appropriately with data size

## Cleaning Between Builds

To switch between CPU and GPU builds:
```bash
# Remove build directory
rm -rf build_cpu build_gpu

# Create fresh build directories as needed
mkdir build_cpu && cd build_cpu
cmake .. -DUSE_CUDA=OFF
make

mkdir ../build_gpu && cd ../build_gpu
cmake .. -DUSE_CUDA=ON
make
```

## Cloud Deployment Notes

For AWS/GCP/Azure GPU instances:
1. Use Deep Learning AMI or GPU-optimized images
2. Verify nvidia-smi works immediately
3. Follow same build process above
4. Consider using containerized environments (Docker with nvidia/runtime)

## Performance Expectations

For meaningful performance comparison:
- Use datasets of 10K+ data points
- Test with 50+ parameter combinations
- GPU advantage becomes pronounced with larger problem sizes
- Overhead of kernel launch dominates for very small problems

## Support

If encountering issues:
1. Check build logs for specific error messages
2. Search NVIDIA Developer Forums for error codes
3. Verify GPU compatibility with CUDA version
4. Ensure drivers match toolkit version