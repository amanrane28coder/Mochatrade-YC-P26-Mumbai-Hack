# Testing GPU-Accelerated Backtesting Engine on Windows

## Overview
This document provides instructions for testing the GPU-accelerated version of the backtesting engine in a Windows environment with NVIDIA CUDA support.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (Compute Capability 3.0+ recommended)
- Minimum 4GB GPU memory
- Modern x86_64 CPU

### Software Requirements
- Windows 10/11 (64-bit)
- NVIDIA CUDA Toolkit 11.0 or higher
- Visual Studio 2019 or 2022 (with Desktop development with C++ workload)
- CMake 3.18+
- Python 3.7+
- NumPy
- Git

## Installation Guide

### 1. Install System Dependencies

#### Install Visual Studio
- Download Visual Studio 2022 Community (or Professional/Enterprise) from https://visualstudio.microsoft.com/
- During installation, select the "Desktop development with C++" workload
- Ensure the Windows 10/11 SDK and C++ CMake tools are included

#### Install Git
- Download from https://git-scm.com/download/win
- Install with default options

#### Install Python
- Download Python 3.12 from https://www.python.org/downloads/windows/
- During installation, check "Add Python to PATH"
- Install for all users or just current user as preferred

#### Install NumPy
Open Command Prompt or PowerShell and run:
```bash
pip install numpy
```

### 2. Install NVIDIA CUDA Toolkit
#### Option A: Using NVIDIA Installer (Recommended)
1. Go to https://developer.nvidia.com/cuda-downloads
2. Select:
   - Operating System: Windows
   - Architecture: x86_64
   - Version: Windows 10/11
   - Installer Type: exe (local)
3. Download and run the installer
4. Follow the installation wizard (express installation is fine)
5. The installer will automatically set up environment variables

#### Option B: Using Zip File
1. Download the CUDA Toolkit zip archive from the same page
2. Extract to a directory (e.g., `C:\cuda`)
3. Manually set environment variables:
   - Add `C:\cuda\bin` to PATH
   - Add `C:\cuda\libnvvp` to PATH (optional, for NVPROF)
   - Add `C:\cuda\lib\x64` to LIB environment variable
   - Add `C:\cuda\include` to INCLUDE environment variable

### 3. Verify Installation
Open Command Prompt and run:
```bash
nvcc --version          # Should show CUDA version
nvidia-smi              # Should show GPU and driver info
where cmake             # Should show cmake.exe location
python --version        # Should show Python version
```

## Building the GPU Version

### 1. Get the Source Code
```bash
git clone <repository-url>
cd gpu_engine
```

### 2. Create GPU Build Directory
```bash
mkdir build_cuda
cd build_cuda
```

### 3. Configure and Build
```bash
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -- /maxcpucount
```
*Note: The `/maxcpucount` flag tells MSBuild to use all available CPU cores.*

### 4. Verify Build Output
Successful build will produce:
- `libgpu_engine_static.lib` - Static library
- `gpu_engine.pyd` - Python extension module (on Windows, `.pyd` instead of `.so`)

## Running Tests

### 1. Install Python Dependencies
```bash
pip install numpy
```

### 2. Execute Test Suite
```bash
# From project root
python tests\test_engine.py
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

## Performance Validation (Optional)

### CPU vs GPU Comparison
To verify both backends produce equivalent results:

1. **Build CPU version** (separate directory):
   ```bash
   mkdir build_cpu && cd build_cpu
   cmake .. -DUSE_CUDA=OFF
   cmake --build . --config Release -- /maxcpucount
   copy gpu_engine.pyd ..\gpu_engine_cpu.pyd
   ```

2. **Build GPU version**:
   ```bash
   cd ..\build_cuda
   cmake --build . --config Release -- /maxcpucount
   copy gpu_engine.pyd ..\gpu_engine_gpu.pyd
   ```

3. **Run comparison script**:
   Create a file `compare.py`:
   ```python
   import numpy as np
   import sys
   import os

   sys.path.insert(0, os.getcwd())

   # Test both versions
   for version, libname in [("CPU", "gpu_engine_cpu.pyd"), ("GPU", "gpu_engine_gpu.pyd")]:
       print(f"\nTesting {version} version...")
       # Load the appropriate module...
       # Run identical tests and compare results within tolerance
   ```
   Then run: `python compare.py`

## Troubleshooting

### Common Build Issues

1. **"nvcc: command not found"**
   - Solution: Ensure CUDA `bin` directory is in PATH
   - Restart Command Prompt after installation

2. **"cannot find -lcudart" or similar linker errors**
   - Solution: Ensure CUDA `lib\x64` directory is in LIB environment variable
   - Or specify explicitly in cmake: `-DCUDA_TOOLKIT_ROOT_DIR="C:\cuda"`

3. **"unsupported gpu architecture"**
   - Solution: Specify compute capability explicitly
   - Add to cmake: `-DCMAKE_CUDA_ARCHITECTURES=75` (adjust for your GPU)
   - Find your GPU's CC: https://developer.nvidia.com/cuda-gpus

4. **CMake cannot find Python**
   - Solution: Ensure Python is in PATH and development headers are installed
   - Or specify: `-DPython_ROOT_DIR="C:\Users\YourUser\AppData\Local\Programs\Python\Python312"`

5. **Nanobind not found**
   - Solution: Ensure nanobind is installed via pip in your Python environment
   - CMake looks for nanobind in the Python site-packages directory

### Runtime Issues

1. **"DLL load failed" when importing gpu_engine**
   - Solution: Ensure CUDA runtime DLLs are in PATH
   - Copy `cudart64_*.dll` from `C:\cuda\bin` to your project directory or add to PATH

2. **"illegal memory access"**
   - Usually indicates buffer overflow in kernel
   - Check array bounds in kernel code
   - Reduce test data size temporarily

3. **Performance not as expected**
   - Ensure GPU is in persistence mode: `nvidia-smi -pm 1`
   - Monitor with `nvidia-smi dmon -s u`
   - Close other GPU-intensive applications

## Verification Checklist

When testing the GPU version, verify:

1. [ ] Module imports successfully: `import gpu_engine`
2. [ ] Engine instantiation works with NumPy arrays
3. [ ] Parameter sweep returns correct number of results
4. [ ] Basic statistics (P&L, Sharpe, drawdown, etc.) are computed
5. [ ] Edge cases (empty data, single point) handled properly
6. [ ] Results are numerically consistent with CPU implementation (within floating-point tolerance)
7. [ ] Performance scales appropriately with data size

## Cloud Deployment Notes

### Azure GPU Instances
1. Use NCv3, NCv4, or ND-series VMs with NVIDIA GPUs
2. Select Windows Server 2019/2022 or Windows 10/11 VM image
3. Verify nvidia-smi works immediately after VM provisioning
4. Follow same build process above
5. Consider using Azure Batch for large-scale parameter sweeps

### AWS EC2 Windows Instances
1. Use G4dn, G5, or P3/P4 instances with NVIDIA GPUs
2. Use Windows Server 2019/2022 AMI with GPU drivers pre-installed
3. Follow same build process
4. Consider using EC2 Fleet or Spot Instances for cost optimization

## Cleaning Between Builds

To switch between CPU and GPU builds:
```bash
# Remove build directories
rmdir /s /q build_cpu build_cuda

# Create fresh CPU build directory
mkdir build_cpu && cd build_cpu
cmake .. -DUSE_CUDA=OFF
cmake --build . --config Release -- /maxcpucount

# Create fresh GPU build directory
mkdir ..\build_cuda && cd ..\build_cuda
cmake .. -DUSE_CUDA=ON
cmake --build . --config Release -- /maxcpucount
```

## Summary

Once you have successfully built and tested the GPU-accelerated version:
1. Your development environment on Windows can use the GPU acceleration (`-DUSE_CUDA=ON`)
2. For systems without NVIDIA GPUs, use the CPU fallback (`-DUSE_CUDA=OFF`)
3. Both versions expose the identical Python API, ensuring code compatibility
4. The engine is ready for high-frequency strategy testing with millisecond-level performance

## Important Notes

- The Windows build uses MSVC (Microsoft Visual C++) compiler
- CUDA Toolkit integrates seamlessly with MSVC via custom build rules
- Python extension on Windows uses `.pyd` extension instead of `.so`
- All source code remains identical between Windows, Linux, and macOS builds
- The CPU fallback (`-DUSE_CUDA=OFF`) works on Windows with multi-threading via `std::async`