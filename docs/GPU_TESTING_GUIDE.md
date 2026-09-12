# GPU-Accelerated Backtesting Engine - Linux GPU Testing Guide

This guide provides instructions for testing the GPU-accelerated version of the backtesting engine in a Linux environment with NVIDIA CUDA support.

## Prerequisites

### System Requirements
- Linux distribution (Ubuntu 20.04/22.04, CentOS 7/8, or similar)
- NVIDIA GPU with CUDA capability (Compute Capability 3.0 or higher)
- At least 4GB of RAM
- 10GB+ free disk space

### Software Dependencies
1. **NVIDIA CUDA Toolkit** (version 11.0 or higher recommended)
2. **CMake** (version 3.18 or higher)
3. **GCC/G++** (version 7 or higher)
4. **Python** (3.7 or higher)
5. **NumPy** (for testing)
6. **Git** (to clone the repository)

## Installation Steps

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

#### Option A: Using Package Manager (Recommended for Ubuntu)
```bash
# Add the CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Option B: Using Runfile Installer
1. Download the appropriate runfile from https://developer.nvidia.com/cuda-downloads
2. Run the installer:
```bash
sudo sh cuda_XX.X_linux.run
```
3. Follow the prompts, accepting the EULA and installing the toolkit
4. Add CUDA to your PATH as shown above

### 3. Verify CUDA Installation
```bash
nvcc --version
# Should show CUDA version info

nvidia-smi
# Should show your GPU and driver information
```

## Building the GPU-Accelerated Engine

### 1. Clone the Repository
```bash
git clone <repository-url>
cd gpu_engine
```

### 2. Create Build Directory and Configure
```bash
mkdir -p build_cuda && cd build_cuda
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
```

### 3. Compile
```bash
make -j$(nproc)
```

### 4. Verify Build
After successful compilation, you should see:
- `libgpu_engine_static.a` - Static library
- `gpu_engine.so` - Python module
- Build completion message

## Running Tests

### 1. Install Python Dependencies
```bash
pip3 install numpy
```

### 2. Run the Test Suite
```bash
# From the project root directory
python3 tests/test_engine.py
```

### 3. Expected Output
You should see output similar to:
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

## Performance Benchmarking (Optional)

To compare CPU vs GPU performance:

### 1. Create a Benchmark Script
Create a file `benchmark.py`:
```python
import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import gpu_engine as ge

def generate_test_data(n_points=100000):
    np.random.seed(42)
    timestamps = np.arange(n_points, dtype=np.uint64) * 1000000000
    returns = np.random.normal(0, 0.01, n_points)
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.uniform(100, 1000, n_points)
    return timestamps, prices, volumes

def benchmark_engine(engine, description):
    start = time.time()
    results = engine.run_parameter_sweep((10, 30), (30, 60), 1.5, 0.5)
    end = time.time()
    print(f"{description}: {end-start:.2f} seconds for {len(results)} parameter combinations")
    print(f"  Performance: {len(results)/(end-start):.1f} combinations/second")

# Generate test data
print("Generating test data (100K points)...")
timestamps, prices, volumes = generate_test_data(100000)

# Test CPU version (rebuild with -DUSE_CUDA=OFF first)
print("\nTesting CPU version...")
cpu_engine = ge.FastQuantEngine(
    timestamps.ctypes.data,
    prices.ctypes.data,
    volumes.ctypes.data,
    len(timestamps)
)
benchmark_engine(cpu_engine, "CPU Backend")

# Test GPU version (rebuild with -DUSE_CUDA=ON)
print("\nTesting GPU version...")
gpu_engine = ge.FastQuantEngine(
    timestamps.ctypes.data,
    prices.ctypes.data,
    volumes.ctypes.data,
    len(timestamps)
)
benchmark_engine(gpu_engine, "GPU Backend")
```

### 2. Run the Benchmark
```bash
python3 benchmark.py
```

## Troubleshooting

### Common Issues

1. **"nvcc command not found"**
   - Ensure CUDA is installed and `/usr/local/cuda/bin` is in your PATH
   - Try: `export PATH=/usr/local/cuda/bin:$PATH`

2. **"cannot find -lcudart" or similar linker errors**
   - Ensure CUDA libraries are in your LD_LIBRARY_PATH
   - Try: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`

3. **"undefined reference to `__cudaRegisterLinkedBinary__"`**
   - This indicates separable compilation issues
   - Try adding `-DCUDA_SEPARABLE_COMPILATION=OFF` to cmake command

4. **"invalid device function" or "no kernel image is available for execution"**
   - Your GPU architecture may not be supported by the default PTX compilation
   - Try specifying architecture: `-DCMAKE_CUDA_ARCHITECTURES=75` (adjust for your GPU)

5. **Memory allocation errors**
   - Ensure you have sufficient GPU memory
   - Try reducing test data size temporarily

### Getting Help
- Check the build output for specific error messages
- Consult NVIDIA CUDA documentation: https://docs.nvidia.com/cuda/
- Search for specific error messages in CUDA forums or Stack Overflow

## Verification Notes

When running tests on GPU vs CPU:
1. Results should be numerically equivalent within floating-point tolerance
2. Performance should be significantly better on GPU for larger datasets
3. Both versions should pass the same test suite

## Cleaning Build Artifacts

To switch between CPU and GPU builds:
```bash
# To switch from GPU to CPU build
rm -rf build_cuda
mkdir build && cd build
cmake .. -DUSE_CUDA=OFF
make

# To switch from CPU to GPU build
rm -rf build
mkdir -p build_cuda && cd build_cuda
cmake .. -DUSE_CUDA=ON
make
```

## Summary

Once you have successfully built and tested the GPU-accelerated version:
1. Your development environment on macOS can use the CPU fallback (`-DUSE_CUDA=OFF`)
2. Your production/deployment environment on Linux can use the GPU acceleration (`-DUSE_CUDA=ON`)
3. Both versions expose the identical Python API, ensuring code compatibility