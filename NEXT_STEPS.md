# Next Steps: GPU Testing Preparation

## ✅ Completed: CPU Fallback Verification on macOS
- Built successfully with `-DUSE_CUDA=OFF`
- All tests in `tests/test_engine.py` pass
- Module imports correctly and exposes expected Python API

## 🔜 Next: Linux GPU Testing Setup

### 1. Environment Preparation
You'll need a Linux system with:
- NVIDIA GPU (Compute Capability 3.0+ recommended)
- NVIDIA CUDA Toolkit 11.0+
- Python 3.7+ with NumPy
- CMake 3.18+
- GCC 7+ or Clang 6+

### 2. Quick Start Guide
```bash
# Clone repository (if not already done)
git clone <repository-url>
cd gpu_engine

# Install dependencies
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-pip python3-dev
pip3 install numpy

# Install CUDA Toolkit (Ubuntu preferred method):
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

### 3. Build GPU Version
```bash
mkdir -p build_gpu && cd build_gpu
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 4. Run Tests
```bash
# From project root
python3 tests/test_engine.py
```

### 5. Performance Comparison (Optional)
```bash
# Build CPU version
mkdir build_cpu && cd build_cpu
cmake .. -DUSE_CUDA=OFF
make
cp gpu_engine.so ../gpu_engine_cpu.so

# Build GPU version (from build_gpu directory)
cd ../build_gpu
make
cp gpu_engine.so ../gpu_engine_gpu.so

# Run comparison (see docs/GPU_TESTING_GUIDE.md for full benchmark script)
```

### 6. Verification Checklist
When testing GPU version, ensure:
- [ ] Module imports: `import gpu_engine`
- [ ] Engine works with NumPy arrays
- [ ] Correct number of results from parameter sweep
- [ ] Reasonable P&L, Sharpe, drawdown values
- [ ] Edge cases handled (empty data, single point)
- [ ] Results match CPU version within floating-point tolerance
- [ ] Performance scales with data size

## 📚 Reference Documents
- `docs/LINUX_GPU_TESTING.md` - Detailed Linux GPU testing instructions
- `docs/GPU_TESTING_GUIDE.md` - Concise GPU testing guide
- `docs/HPC_DEPLOYMENT_GUIDE.md` - High-performance deployment notes

## 🐛 Troubleshooting Common Issues
See the troubleshooting sections in the reference documents for:
- "nvcc: command not found" → Add CUDA to PATH
- "cannot find -lcudart" → Add CUDA libs to LD_LIBRARY_PATH
- "unsupported gpu architecture" → Specify `-DCMAKE_CUDA_ARCHITECTURES`
- Memory allocation failures → Reduce test data size

## 🚀 Cloud Deployment
For AWS/GCP/Azure GPU instances:
1. Use Deep Learning AMI or GPU-optimized images
2. Verify `nvidia-smi` works immediately
3. Follow same build process
4. Consider Docker with nvidia/runtime

## 📝 Important Notes
- CPU fallback (`-DUSE_CUDA=OFF`) and GPU (`-DUSE_CUDA=ON`) use identical source
- Both versions expose identical Python API via nanobind
- Code developed on macOS CPU fallback works unchanged on Linux GPU
- Memory layout and algorithms are identical between implementations