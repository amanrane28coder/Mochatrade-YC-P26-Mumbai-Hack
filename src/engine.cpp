#include "../include/engine.cuh"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
#include <thread>
#include <future>
#include <deque>
#endif
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <utility>
#include <cmath>
#include <vector>

namespace {
    // CPU fallback implementation of the backtest for a single parameter set
    PerformanceMetrics compute_metrics_cpu(const TickDataSoA& data,
                                          int fast_window,
                                          int slow_window,
                                          double slippage_bps,
                                          double commission_bps) {
        size_t size = data.size();
        if (size == 0) {
            return PerformanceMetrics{0.0, 0.0, 0.0, 0, 0.0};
        }

        // Initialize backtest state
        int position = 0; // 0: out, 1: long
        double entry_price = 0.0;
        double equity = 0.0;
        double peak_equity = 0.0;
        double max_drawdown = 0.0; // stored as percentage, capped at 100%
        int winning_trades = 0;
        int total_trades = 0;
        double total_pnl = 0.0;

        // Use int signal to avoid floating-point equality comparisons (Issue #16)
        int signal_to_execute = 0; // -1: bearish, 0: no signal, 1: bullish

        // For moving averages: sliding window sums
        double fast_sum = 0.0;
        double slow_sum = 0.0;
        std::deque<double> fast_window_prices;
        std::deque<double> slow_window_prices;

        // Iterate over each bar
        for (size_t i = 0; i < size; ++i) {
            double price = data.prices()[i];

            // Update fast window
            fast_window_prices.push_back(price);
            fast_sum += price;
            if (fast_window_prices.size() > static_cast<size_t>(fast_window)) {
                fast_sum -= fast_window_prices.front();
                fast_window_prices.pop_front();
            }

            // Update slow window
            slow_window_prices.push_back(price);
            slow_sum += price;
            if (slow_window_prices.size() > static_cast<size_t>(slow_window)) {
                slow_sum -= slow_window_prices.front();
                slow_window_prices.pop_front();
            }

            // Compute current signal (for this bar) if we have enough data for both windows
            int signal_curr = 0;
            if (fast_window_prices.size() == static_cast<size_t>(fast_window) &&
                slow_window_prices.size() == static_cast<size_t>(slow_window)) {
                double fast_ma = fast_sum / fast_window;
                double slow_ma = slow_sum / slow_window;
                if (fast_ma > slow_ma) {
                    signal_curr = 1;
                } else if (fast_ma < slow_ma) {
                    signal_curr = -1;
                }
            }

            // If we have a signal to execute from the previous bar and we are in a position, we close at this bar's price
            if (signal_to_execute != 0 && position == 1) {
                // We only exit on signal_to_execute == -1 (as per the original strategy)
                if (signal_to_execute == -1) {
                    double execution_price = price;
                    double cost = execution_price * (slippage_bps + commission_bps) / 10000.0;
                    double exit_price = execution_price - cost; // we pay slippage and commission when selling
                    double trade_pnl = exit_price - entry_price;
                    total_pnl += trade_pnl;
                    if (trade_pnl > 0) {
                        winning_trades++;
                    }
                    total_trades++;

                    equity += trade_pnl;
                    if (equity > peak_equity) {
                        peak_equity = equity;
                    }
                    if (peak_equity > 0.0) {
                        double drawdown_pct = (peak_equity - equity) / peak_equity * 100.0;
                        drawdown_pct = std::min(drawdown_pct, 100.0); // Cap at 100% (Issue #1)
                        if (drawdown_pct > max_drawdown) {
                            max_drawdown = drawdown_pct;
                        }
                    }
                    position = 0;
                }
                // Note: we ignore signal_to_execute == 1 when in position (we are only long and don't reverse)
            }

            // If we have a signal to execute from the previous bar and we are not in a position, we open a long position if the signal is 1
            if (signal_to_execute != 0 && position == 0) {
                if (signal_to_execute == 1) {
                    double execution_price = price;
                    double cost = execution_price * (slippage_bps + commission_bps) / 10000.0;
                    entry_price = execution_price + cost; // we pay slippage and commission when buying
                    position = 1;
                }
            }

            // Prepare the signal to execute on the next bar: it is the current bar's signal
            signal_to_execute = signal_curr;
        }

        // After processing all bars, if we are still in a position, close at the last price
        // (This ensures we don't leave an open position; the original CUDA kernel did not execute the signal from the last bar,
        // but for realism we close at the last available price.)
        if (position == 1) {
            double execution_price = data.prices()[size - 1];
            double cost = execution_price * (slippage_bps + commission_bps) / 10000.0;
            double exit_price = execution_price - cost; // we pay slippage and commission when selling
            double trade_pnl = exit_price - entry_price;
            total_pnl += trade_pnl;
            if (trade_pnl > 0) {
                winning_trades++;
            }
            total_trades++;

            equity += trade_pnl;
            if (equity > peak_equity) {
                peak_equity = equity;
            }
            if (peak_equity > 0.0) {
                double drawdown_pct = (peak_equity - equity) / peak_equity * 100.0;
                drawdown_pct = std::min(drawdown_pct, 100.0); // Cap at 100% (Issue #1)
                if (drawdown_pct > max_drawdown) {
                    max_drawdown = drawdown_pct;
                }
            }
            position = 0;
        }

        // Compute performance metrics
        // NOTE: This is a placeholder Sharpe ratio approximation (Issue #5).
        // A true Sharpe ratio requires per-trade return standard deviation:
        //   sharpe = (mean_return - risk_free_rate) / std_dev(returns) * sqrt(252)
        // This placeholder grows unboundedly with trade count and should not be
        // used for cross-strategy comparison.
        double sharpe_ratio = 0.0;
        if (total_trades > 1) {
            double avg_trade_pnl = total_pnl / total_trades;
            sharpe_ratio = avg_trade_pnl * std::sqrt(static_cast<double>(total_trades));
        }
        double win_rate = (total_trades > 0) ? (static_cast<double>(winning_trades) / total_trades) * 100.0 : 0.0;

        PerformanceMetrics metrics;
        metrics.total_pnl = total_pnl;
        metrics.sharpe_ratio = sharpe_ratio;
        metrics.max_drawdown = max_drawdown; // already in percentage, capped at 100%
        metrics.total_trades = static_cast<uint64_t>(total_trades);
        metrics.win_rate = win_rate;

        return metrics;
    }
}

FastQuantEngine::FastQuantEngine(const uint64_t* timestamps,
                                 const double* prices,
                                 const double* volumes,
                                 size_t size)
    : tick_data_(size) {
    if (size > 0) {
        // Copy data to unified memory (zero-copy accessible from both CPU and GPU)
        // Use std::memcpy for host-to-host copy (works with unified memory)
        std::memcpy(tick_data_.timestamps(), timestamps, size * sizeof(uint64_t));
        std::memcpy(tick_data_.prices(), prices, size * sizeof(double));
        std::memcpy(tick_data_.volumes(), volumes, size * sizeof(double));
    }
}

FastQuantEngine::~FastQuantEngine() {
    // Destructor handled by TickDataSoA
}

FastQuantEngine::FastQuantEngine(FastQuantEngine&& other) noexcept
    : tick_data_(std::move(other.tick_data_)) {
}

FastQuantEngine& FastQuantEngine::operator=(FastQuantEngine&& other) noexcept {
    if (this != &other) {
        tick_data_ = std::move(other.tick_data_);
    }
    return *this;
}

std::vector<PerformanceMetrics> FastQuantEngine::run_parameter_sweep(
    std::pair<int, int> fast_window_range,
    std::pair<int, int> slow_window_range,
    double slippage_bps,
    double commission_bps) {

    size_t size = tick_data_.size();
    if (size == 0) {
        return {};
    }

    // Calculate number of parameter combinations
    int fast_min = fast_window_range.first;
    int fast_max = fast_window_range.second;
    int slow_min = slow_window_range.first;
    int slow_max = slow_window_range.second;

    // Build list of valid parameter combinations (Issue #4: filter out fast >= slow)
    std::vector<std::pair<int, int>> valid_params;
    for (int fast_window = fast_min; fast_window <= fast_max; ++fast_window) {
        for (int slow_window = slow_min; slow_window <= slow_max; ++slow_window) {
            if (fast_window < slow_window) {
                valid_params.emplace_back(fast_window, slow_window);
            }
        }
    }

    if (valid_params.empty()) {
        return {};
    }

    int param_count = static_cast<int>(valid_params.size());

#ifdef USE_CUDA
    // Build parameter arrays for single-dispatch kernel (Issue #14)
    int* d_fast_windows = nullptr;
    int* d_slow_windows = nullptr;
    PerformanceMetrics* d_results = nullptr;
    double* d_ma_buffers = nullptr;

    auto cuda_check = [](cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string(msg) + ": " + cudaGetErrorString(err));
        }
    };

    cuda_check(cudaMallocManaged(&d_fast_windows, param_count * sizeof(int)),
               "Failed to allocate fast_windows");
    cuda_check(cudaMallocManaged(&d_slow_windows, param_count * sizeof(int)),
               "Failed to allocate slow_windows");
    cuda_check(cudaMallocManaged(&d_results, param_count * sizeof(PerformanceMetrics)),
               "Failed to allocate results");
    cudaMemset(d_results, 0, param_count * sizeof(PerformanceMetrics));

    for (int i = 0; i < param_count; ++i) {
        d_fast_windows[i] = valid_params[i].first;
        d_slow_windows[i] = valid_params[i].second;
    }

    // Allocate global memory for per-thread MA buffers
    // Each thread needs 2 * size doubles (fast_ma + slow_ma)
    // Warning: for large sweeps this can be very large
    // (e.g., 10K params × 100K points = 16 GB)
    size_t ma_buffer_bytes = static_cast<size_t>(param_count) * 2 * size * sizeof(double);
    cuda_check(cudaMallocManaged(&d_ma_buffers, ma_buffer_bytes),
               "Failed to allocate MA buffers (try reducing parameter range or data size)");

    // Single kernel launch for all parameter combinations
    const int block_size = 256;
    const int grid_size = (param_count + block_size - 1) / block_size;

    backtest_kernel<<<grid_size, block_size>>>(
        tick_data_.timestamps(),
        tick_data_.prices(),
        tick_data_.volumes(),
        static_cast<int>(size),
        d_fast_windows,
        d_slow_windows,
        param_count,
        slippage_bps,
        commission_bps,
        d_ma_buffers,
        d_results
    );

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy results back to host vector
    std::vector<PerformanceMetrics> h_results(param_count);
    cudaMemcpy(h_results.data(), d_results, param_count * sizeof(PerformanceMetrics), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_fast_windows);
    cudaFree(d_slow_windows);
    cudaFree(d_ma_buffers);
    cudaFree(d_results);

    return h_results;
#else
    // CPU fallback: use std::async to run each valid parameter set in parallel
    std::vector<std::future<PerformanceMetrics>> futures;
    futures.reserve(param_count);

    for (const auto& [fast_window, slow_window] : valid_params) {
        futures.push_back(std::async(std::launch::async, compute_metrics_cpu,
                                     std::cref(tick_data_),
                                     fast_window, slow_window,
                                     slippage_bps, commission_bps));
    }

    // Collect results
    std::vector<PerformanceMetrics> h_results;
    h_results.reserve(param_count);
    for (auto& f : futures) {
        h_results.push_back(f.get());
    }

    return h_results;
#endif
}