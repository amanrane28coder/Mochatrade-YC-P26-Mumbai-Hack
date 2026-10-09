# Void Quant — GPU-Accelerated Backtesting Engine

A quantitative research and developer platform for exploring trading strategies across historical market data. The project combines a C++20 engine, optional CUDA acceleration, and Python bindings so researchers can work in Python while the engine handles backtest computation.

> **Project stage:** Early-stage, unincorporated engineering project begun in September 2026. Development used Claude Code in the terminal with free Opus 5 access; this is development-tool use, not a Claude API integration. The CPU fallback has been built and exercised on macOS Apple Silicon. GPU validation and performance benchmarking remain open work; estimates below are not measured results.

## The problem

Strategy research often involves repeatedly evaluating parameter choices over time-series market data. Python makes research accessible, but repeated computation can become a bottleneck as datasets and parameter sweeps grow. Backtests also need explicit execution assumptions: ignoring timing, spreads, fees, or information that would not yet have been available can make results misleading.

## The approach

This engine provides a Python-facing workflow backed by a C++20 implementation, with a CUDA build option for NVIDIA GPUs. It evaluates parameter sweeps and models selected execution costs, while an **N+1 execution lock** is intended to enforce temporal barriers and help avoid look-ahead bias. A CPU fallback supports development on systems without CUDA.

The engine produces research results and trading signals. It is not a live trading system and does not provide portfolio construction, risk management, order routing, or compliance controls.

## Architecture

- **C++20 core with optional CUDA kernels:** shared engine code can be built with CPU fallback (-DUSE_CUDA=OFF) or GPU acceleration (-DUSE_CUDA=ON).
- **Python interface via nanobind:** exposes the engine to Python and NumPy-based research workflows. Input data is copied into unified memory by the engine; the Python-to-engine handoff is not zero-copy.
- **Structure of Arrays (SoA):** market data is stored in separate arrays (timestamps, prices, and volumes) to support the engine's data-processing layout.
- **N+1 execution lock:** applies temporal barriers during execution to reduce look-ahead bias.
- **Microstructure settings:** supports configurable latency, bid-ask spread/slippage, and transaction fees/commission.
- **Shared Python API:** CPU and GPU builds are designed to expose the same interface. GPU builds target Linux or Windows systems with NVIDIA CUDA; CPU development is documented for macOS, Linux, and Windows.

## Features

- Parameter sweeps over fast/slow window ranges; invalid fast >= slow combinations are excluded.
- Backtest result fields include total P&L, Sharpe ratio, maximum drawdown, trade count, and win rate.
- Configurable slippage and commission in basis points.
- CPU fallback for development and testing without CUDA.
- CUDA build path for GPU deployment.
- Python test suite covering basic execution and edge cases.

## Current status

The repository documents a successful CPU fallback build and test run on macOS Apple Silicon using -DUSE_CUDA=OFF. The test run exercises module import, engine construction, a parameter sweep, and basic edge cases.

The repository also contains CUDA build and GPU testing instructions. Its own testing notes list GPU build verification, CPU/GPU numerical equivalence, performance benchmarks, and memory checks as next steps. The published test output includes an extremely large drawdown result for one case, so these checks should not be treated as validation of financial correctness. GPU performance and numerical equivalence have not been established in the repository.

## Roadmap

1. Build and run the CUDA configuration on supported NVIDIA hardware.
2. Compare CPU and GPU outputs within defined floating-point tolerances, including edge cases.
3. Investigate and test numerical behavior in drawdown and other reported statistics.
4. Publish reproducible CPU/GPU benchmark methodology and measured results across documented data sizes and hardware.
5. Expand validation around execution timing, costs, and parameter-sweep behavior.
6. Improve developer setup and package the Python interface for a repeatable research workflow.

## Potential Claude/API integration

There is no Claude integration in the engine today. If added, Claude or another language-model API could support optional research and developer workflows such as:

- Turning a natural-language research question into a draft parameter-sweep configuration for a user to review.
- Summarizing and comparing result tables produced by a run.
- Explaining configuration choices, engine outputs, and test failures.
- Helping developers draft strategy experiments or documentation.

Any such integration would need to keep the engine's inputs and outputs explicit and verifiable. It would not replace backtest validation or provide live trading, investment, or risk decisions.

## Project structure

```
gpu_engine/
├── include/                # Engine and market-data headers
├── src/                    # Host implementation, bindings, and CUDA kernels
├── tests/                  # Python validation tests
├── docs/                   # Platform-specific build and GPU testing guides
├── notebooks/              # Demonstration notebooks
├── scripts/                # Build and deployment scripts
├── BUILD_VERIFICATION.md   # CPU fallback verification on macOS
├── GPU_TESTING_SUMMARY.md  # GPU testing status and next steps
├── NEXT_STEPS.md           # Linux GPU testing preparation
└── CMakeLists.txt          # Build configuration
```

## Build and run

### Requirements

- C++20-compatible compiler
- CMake 3.18+
- Python 3.7+ and NumPy
- NVIDIA CUDA Toolkit 11.0+ for GPU builds

### CPU fallback

```bash
git clone https://github.com/amanrane28coder/Mochatrade-YC-P26-Mumbai-Hack.git
cd Mochatrade-YC-P26-Mumbai-Hack
cmake -S . -B build -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Install or copy the built Python extension as appropriate for your platform, then run the repository's test script:

```bash
python3 tests/test_engine.py
```

### NVIDIA GPU build

On a supported Linux or Windows environment with the CUDA toolkit installed:

```bash
cmake -S . -B build_gpu -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_gpu --parallel
```

See [GPU testing guides](docs/GPU_TESTING_GUIDE.md) and [Linux GPU testing instructions](docs/LINUX_GPU_TESTING.md) for platform-specific steps. The GPU build path is documented, but GPU validation is still an open roadmap item.

## Python usage

```python
import numpy as np
import gpu_engine as ge

timestamps = np.arange(10_000, dtype=np.uint64) * 1_000_000_000
prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 10_000)))
volumes = np.random.uniform(100, 1_000, 10_000)

engine = ge.FastQuantEngine(timestamps, prices, volumes)
results = engine.run_parameter_sweep(
    fast_window_range=(5, 50),
    slow_window_range=(10, 100),
    slippage_bps=1.5,
    commission_bps=0.5,
)
```

The engine's current example reports result fields such as P&L, Sharpe ratio, drawdown, trades, and win rate. Treat these as research outputs that require independent validation.

## Performance expectations — unvalidated estimates

The figures below are estimates retained from the project notes, not benchmark results. The repository does not provide measured runs, hardware details, or a reproducible benchmark result supporting these numbers. Actual performance has not yet been established and will depend on hardware, workload, data transfer, and parameter combinations.

| Data points | CPU time (estimate) | GPU time (estimate) | Speedup (estimate) |
|-------------:|--------------------:|--------------------:|-------------------:|
| 1K           | 50–100 ms           | 5–10 ms             | 5–10×              |
| 10K          | 500–1,000 ms        | 10–20 ms            | 25–50×             |
| 100K         | 5–10 s              | 50–100 ms           | 50–100×            |
| 1M           | 50–100 s            | 200–500 ms           | 100–200×            |

## Scope and limitations

- The engine generates backtest results and trading signals only. Risk management, portfolio construction, live order execution, and compliance are outside the current scope.
- Backtests do not establish live-trading performance. Results depend on data quality and assumptions about timing, costs, liquidity, and market impact.
- CPU/GPU numerical equivalence is a validation goal, not a result claimed here.
- Performance estimates are unvalidated; use reproducible measurements before making performance decisions.

## Documentation

- [CPU fallback build verification](BUILD_VERIFICATION.md)
- [GPU testing summary](GPU_TESTING_SUMMARY.md)
- [Next steps for GPU testing](NEXT_STEPS.md)
- [GPU testing guide](docs/GPU_TESTING_GUIDE.md)
- [Linux GPU testing guide](docs/LINUX_GPU_TESTING.md)
- [Windows GPU testing guide](docs/WINDOWS_GPU_TESTING.md)
- [Python test suite](tests/test_engine.py)

## Contributing

Contributions to the engine, tests, and documentation are welcome. Please follow the existing C++ and Python style, update documentation when behavior changes, and run the relevant tests for your build configuration.

## License

**TODO:** Choose and add a project license before reuse or redistribution terms are stated.

## Team

- Aman Rane
- Harsh Gosavi
- Yash Kushwaha
- Anirudh Jatav
