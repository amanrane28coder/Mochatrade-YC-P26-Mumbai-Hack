# GPU-Accelerated Backtesting Engine

An institutional-grade, GPU-accelerated backtesting engine written in modern C++ and CUDA, exposed to Python via low-overhead nanobind bindings. Empowers quants to test high-frequency strategies across millions of ticks in milliseconds.

## 🚀 Overview

This engine solves the CPU bottleneck in algorithmic trading research by leveraging extreme parallel compute power of NVIDIA hardware while maintaining accessibility through Python scripting. It implements:

- **Core C++20 & CUDA Engine**: Processes millions of market events simultaneously
- **Structure of Arrays (SoA) Layout**: Maximizes GPU memory bandwidth with CUDA Unified Memory
- **Zero-Copy API Bridge**: Nanobind passes NumPy arrays directly without memory duplication
- **N+1 Execution Lock**: Prevents look-ahead bias by enforcing temporal barriers
- **Configurable Microstructure**: Models latency, bid-ask spread, and transaction fees
- **Dual-Mode Build**: CPU fallback (`-DUSE_CUDA=OFF`) and GPU acceleration (`-DUSE_CUDA=ON`)

## 📊 Features

- **Massive Parallelism**: Test 10,000+ parameter combinations in milliseconds
- **Institutional Execution**: N+1 lock prevents curve-fitting from look-ahead bias
- **Realistic Market Mechanics**: Configurable slippage and commission (basis points)
- **Zero-Copy Python Interface**: Direct NumPy array access via nanobind
- **Cross-Platform**: Develop on macOS/Windows/Linux (CPU), deploy on Linux/Windows (GPU)
- **Identical API**: Same Python interface for CPU fallback and GPU acceleration

## 📁 Project Structure

```
gpu_engine/
├── include/                # Header files
│   ├── engine.cuh          # FastQuantEngine interface
│   └── market_data.cuh     # TickDataSoA structure
├── src/                    # Source files
│   ├── engine.cpp          # Host implementation & CPU fallback
│   ├── bindings.cpp        # Nanobind Python bindings
│   └── kernels.cu          # CUDA kernels (when USE_CUDA=ON)
├── tests/                  # Python test suite
│   └── test_engine.py      # Validation tests
├── docs/                   # Platform-specific guides
│   ├── LINUX_GPU_TESTING.md
│   ├── GPU_TESTING_GUIDE.md
│   └── WINDOWS_GPU_TESTING.md
├── notebooks/              # Jupyter notebooks for demonstration
├── scripts/                # Build and deployment scripts
├── BUILD_VERIFICATION.md   # CPU fallback verification on macOS
├── GPU_TESTING_SUMMARY.md  # Overview and next steps
├── NEXT_STEPS.md           # Quick start for Linux GPU testing
├── CMakeLists.txt          # Build configuration
└── README.md               # This file
```

## ⚙️ Installation & Usage

### Prerequisites
- C++20-compatible compiler (AppleClang, GCC 7+, or Clang 6+)
- CMake 3.18+
- Python 3.7+
- NumPy
- NVIDIA CUDA Toolkit 11.0+ (for GPU acceleration only)

### CPU Fallback Development (macOS/Linux/Windows)
*Recommended for initial development and testing*

```bash
# Clone repository
git clone <repository-url>
cd gpu_engine

# Build CPU fallback version
rm -rf build && mkdir build && cd build
cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # or -j$(sysctl -n hw.ncpu) on macOS

# Install Python module
cp build/gpu_engine.so .  # Linux/macOS
# On Windows: copy build\gpu_engine.pyd . 

# Run tests
python3 tests/test_engine.py
```

### GPU Acceleration Deployment (Linux/Windows with NVIDIA GPU)
*For production deployment and maximum performance*

```bash
# Build GPU version
mkdir -p build_gpu && cd build_gpu
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python module
cp build/gpu_engine.so .  # Linux/macOS (.so)
# On Windows: copy build\gpu_engine.pyd .  (.pyd)

# Run tests (should match CPU results)
python3 tests/test_engine.py
```

## 📖 Documentation

- **[BUILD_VERIFICATION.md](BUILD_VERIFICATION.md)** - CPU fallback verification on macOS Apple Silicon
- **[GPU_TESTING_SUMMARY.md](GPU_TESTING_SUMMARY.md)** - Overview and preparation for GPU testing
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - Quick start guide for Linux GPU testing
- **[docs/LINUX_GPU_TESTING.md](docs/LINUX_GPU_TESTING.md)** - Comprehensive Linux GPU testing instructions
- **[docs/GPU_TESTING_GUIDE.md](docs/GPU_TESTING_GUIDE.md)** - Concise GPU testing reference
- **[docs/WINDOWS_GPU_TESTING.md](docs/WINDOWS_GPU_TESTING.md)** - Complete Windows GPU testing guide
- **[tests/test_engine.py](tests/test_engine.py)** - Python validation test suite

## 🔧 Usage Example

```python
import numpy as np
import gpu_engine as ge

# Generate or load your market data
# Format: timestamps (uint64_t), prices (double), volumes (double)
timestamps = np.arange(10000, dtype=np.uint64) * 1000000000  # 1-second intervals
prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 10000)))
volumes = np.random.uniform(100, 1000, 10000)

# Create engine — pass NumPy arrays directly (data is copied into unified memory
# for CPU↔GPU zero-copy transfer; the Python→engine handoff involves a memcpy)
engine = ge.FastQuantEngine(timestamps, prices, volumes)

# Run parameter sweep (only valid fast < slow combinations are evaluated)
results = engine.run_parameter_sweep(
    fast_window_range=(5, 50),    # Test fast MA windows 5-50
    slow_window_range=(10, 100),  # Test slow MA windows 10-100
    slippage_bps=1.5,             # 0.015% slippage per trade
    commission_bps=0.5            # 0.005% commission per trade
)

# Access results
for i, result in enumerate(results):
    print(f"Combo {i}: P&L={result.total_pnl:.2f}, "
          f"Sharpe={result.sharpe_ratio:.2f}, "
          f"DD={result.max_drawdown:.2f}%, "
          f"Trades={result.total_trades}, Win%={result.win_rate:.1f}")
```

## 🎯 Performance Expectations

| Data Points | CPU Time (approx) | GPU Time (approx) | Speedup |
|-------------|-------------------|-------------------|---------|
| 1K          | 50-100 ms         | 5-10 ms           | 5-10x   |
| 10K         | 500-1000 ms       | 10-20 ms          | 25-50x  |
| 100K        | 5-10 seconds      | 50-100 ms         | 50-100x |
| 1M          | 50-100 seconds    | 200-500 ms        | 100-200x|

*Actual performance depends on GPU model, data size, and parameter combinations.*

## 🔄 Development Workflow

1. **Develop/Test**: Use CPU fallback (`-DUSE_CUDA=OFF`) on any system (macOS/Windows/Linux)
2. **Validate**: Ensure results are correct and strategies are sound
3. **Deploy**: Switch to GPU acceleration (`-DUSE_CUDA=ON`) on Linux/NVIDIA or Windows/NVIDIA
4. **Scale**: Process larger datasets and more parameter combinations with minimal code changes

## ⚠️ Important Notes

- **This engine generates trading signals only** - you must add:
  - Risk management (position limits, VAR, etc.)
  - Order execution systems
  - Portfolio construction
  - Compliance checks
- **Backtesting ≠ Live Trading**: Always account for latency, order book dynamics, and market impact
- **Start with Paper Trading**: Validate with simulation before risking capital
- **Numerical Equivalence**: CPU and GPU results should match within floating-point tolerance

## 📈 Performance Validation

To compare CPU vs GPU performance on the same system:

```bash
# Build CPU version
mkdir build_cpu && cd build_cpu
cmake .. -DUSE_CUDA=OFF
make
cp gpu_engine.so ../gpu_engine_cpu.so  # Linux/macOS
# Windows: copy build\gpu_engine.pyd ..\gpu_engine_cpu.pyd

# Build GPU version (requires NVIDIA GPU)
cd ../build_gpu
make
cp gpu_engine.so ../gpu_engine_gpu.so  # Linux/macOS
# Windows: copy build\gpu_engine.pyd ..\gpu_engine_gpu.pyd

# Run benchmark comparison (see docs/GPU_TESTING_GUIDE.md)
```

## 🤝 Contributing

This project follows standard C++ and Python practices. Please ensure:
- Code matches existing style and conventions
- All tests pass before submitting changes
- Documentation is updated for new features
- Build system modifications are tested on all target platforms

## 📄 License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## 🙏 Acknowledgments

- Nanobind team for excellent Python-C++ bindings
- NVIDIA for CUDA parallel computing platform
- Open-source quantitative finance community

## 👥 Team

- **Aman Rane** - Lead Developer
- **Harsh Gosavi** - Contributor
- **Anirudh Jatav** - Contributor
- **Yash Kushwah** - Contributor

---

**Ready for high-frequency strategy research and development.** Start with the CPU fallback on your current system, then scale to GPU acceleration when you need maximum performance.