#ifndef FAST_QUANT_ENGINE_CUH
#define FAST_QUANT_ENGINE_CUH

#include "market_data.cuh"
#include "metrics.cuh"
#include <vector>
#include <cstdint>

#ifdef USE_CUDA
/**
 * @brief Forward declaration of the CUDA backtest kernel (single-dispatch)
 *
 * Each thread processes one parameter combination from the parameter arrays.
 */
__global__ void backtest_kernel(
    const uint64_t* timestamps,
    const double* prices,
    const double* volumes,
    int size,
    const int* fast_windows,
    const int* slow_windows,
    int num_params,
    double slippage_bps,
    double commission_bps,
    double* ma_buffers,
    PerformanceMetrics* results
);
#endif

/**
 * @brief GPU-Accelerated Backtesting Engine
 *
 * This class provides a high-performance backtesting engine that executes
 * trading strategies on GPU using CUDA. It supports parameter sweeps and zero-copy data
 * transfer via CUDA Unified Memory.
 */
class FastQuantEngine {
public:
    /**
     * @brief Construct a new FastQuantEngine object
     *
     * @param timestamps Array of timestamps (nanoseconds since epoch)
     * @param prices Array of prices
     * @param volumes Array of volumes
     * @param size Number of elements in each array
     */
    FastQuantEngine(const uint64_t* timestamps,
                    const double* prices,
                    const double* volumes,
                    size_t size);

    /**
     * @brief Destroy the FastQuantEngine object
     */
    ~FastQuantEngine();

    // Delete copy constructor and assignment
    FastQuantEngine(const FastQuantEngine&) = delete;
    FastQuantEngine& operator=(const FastQuantEngine&) = delete;

    // Allow move semantics
    FastQuantEngine(FastQuantEngine&& other) noexcept;
    FastQuantEngine& operator=(FastQuantEngine&& other) noexcept;

    /**
     * @brief Run a parameter sweep across fast and slow window ranges
     *
     * @param fast_window_range Pair of (min, max) fast window values to test
     * @param slow_window_range Pair of (min, max) slow window values to test
     * @param slippage_bps Slippage in basis points per trade
     * @param commission_bps Commission in basis points per trade
     * @return std::vector<PerformanceMetrics> Results for each parameter combination
     */
    std::vector<PerformanceMetrics> run_parameter_sweep(
        std::pair<int, int> fast_window_range,
        std::pair<int, int> slow_window_range,
        double slippage_bps = 1.5,
        double commission_bps = 0.5);

private:
    TickDataSoA tick_data_;  ///< Internal tick data storage
};

#endif // FAST_QUANT_ENGINE_CUH