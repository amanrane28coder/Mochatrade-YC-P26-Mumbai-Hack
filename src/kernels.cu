#include "metrics.cuh"
#include <cmath>

/**
 * @brief CUDA device function to compute moving average into a pre-allocated buffer
 *
 * @param prices Input prices array
 * @param size Number of elements
 * @param window Window size for moving average
 * @param ma Output moving average array (must be pre-allocated, one per thread)
 */
__device__ void compute_moving_average(const double* prices, int size, int window, double* ma) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += prices[i];
        if (i >= window) {
            sum -= prices[i - window];
        }
        if (i >= window - 1) {
            ma[i] = sum / window;
        } else {
            ma[i] = 0.0; // Not enough data yet
        }
    }
}

/**
 * @brief CUDA kernel for backtesting multiple parameter sets in a single dispatch
 *
 * Each thread processes one parameter combination from the fast_windows/slow_windows
 * arrays and computes performance metrics by iterating through the entire time series.
 * Uses pre-allocated global memory buffers for moving averages to avoid shared memory
 * race conditions (Issue #3 fix).
 *
 * @param timestamps Array of timestamps (unused in this simple strategy)
 * @param prices Array of prices
 * @param volumes Array of volumes (unused in this simple strategy)
 * @param size Number of time steps
 * @param fast_windows Array of fast MA window values (one per param combo)
 * @param slow_windows Array of slow MA window values (one per param combo)
 * @param num_params Total number of parameter combinations
 * @param slippage_bps Slippage in basis points per trade
 * @param commission_bps Commission in basis points per trade
 * @param ma_buffers Pre-allocated global memory for MA computation (num_params * 2 * size doubles)
 * @param results Pointer to array of PerformanceMetrics (one per param combo)
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
) {
    // Each thread handles one parameter combination
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check against number of parameter combinations (Issue #6 fix)
    if (idx >= num_params) return;

    int fast_window = fast_windows[idx];
    int slow_window = slow_windows[idx];

    // Each thread gets its own MA buffers in global memory (Issue #3 fix: no shared memory races)
    double* fast_ma = ma_buffers + static_cast<size_t>(idx) * 2 * size;
    double* slow_ma = fast_ma + size;

    // Compute moving averages
    compute_moving_average(prices, size, fast_window, fast_ma);
    compute_moving_average(prices, size, slow_window, slow_ma);

    // Initialize backtest variables
    int position = 0; // 0: out, 1: long
    double entry_price = 0.0;
    double equity = 0.0;
    double peak_equity = 0.0;
    double max_drawdown = 0.0; // fraction, capped at 1.0
    int winning_trades = 0;
    int total_trades = 0;
    double total_pnl = 0.0;

    // We need to wait for both MAs to be valid
    int start_idx = (fast_window > slow_window) ? fast_window - 1 : slow_window - 1;

    for (int i = start_idx; i < size - 1; ++i) { // -1 because we execute at i+1
        int signal = 0;
        if (fast_ma[i] > slow_ma[i]) {
            signal = 1; // Bullish
        } else if (fast_ma[i] < slow_ma[i]) {
            signal = -1; // Bearish
        }

        // Execute at next bar's open (we'll use close of i+1 as execution price for simplicity)
        double execution_price = prices[i+1];

        // Apply slippage and commission
        double slippage = execution_price * (slippage_bps / 10000.0);
        double commission = execution_price * (commission_bps / 10000.0);
        double cost = slippage + commission;

        if (position == 0 && signal == 1) {
            // Enter long position
            position = 1;
            entry_price = execution_price + cost; // We pay slippage and commission when buying
        } else if (position == 1 && signal == -1) {
            // Exit long position
            position = 0;
            double exit_price = execution_price - cost; // We receive less due to slippage and commission
            double trade_pnl = exit_price - entry_price;
            total_pnl += trade_pnl;

            if (trade_pnl > 0) {
                winning_trades++;
            }
            total_trades++;

            // Update equity and drawdown
            equity += trade_pnl;
            if (equity > peak_equity) {
                peak_equity = equity;
            }
            if (peak_equity > 0.0) {
                double drawdown = (peak_equity - equity) / peak_equity;
                if (drawdown > 1.0) drawdown = 1.0; // Cap at 100% (Issue #1 fix)
                if (drawdown > max_drawdown) {
                    max_drawdown = drawdown;
                }
            }
        }
    }

    // If we are still in a position at the end, close it at the last price
    if (position == 1) {
        double exit_price = prices[size-1];
        double slippage = exit_price * (slippage_bps / 10000.0);
        double commission = exit_price * (commission_bps / 10000.0);
        double cost = slippage + commission;
        double exit_price_net = exit_price - cost;
        double trade_pnl = exit_price_net - entry_price;
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
            double drawdown = (peak_equity - equity) / peak_equity;
            if (drawdown > 1.0) drawdown = 1.0; // Cap at 100% (Issue #1 fix)
            if (drawdown > max_drawdown) {
                max_drawdown = drawdown;
            }
        }
    }

    // Compute performance metrics
    // NOTE: This is a placeholder Sharpe ratio approximation (Issue #5).
    // A true Sharpe ratio requires per-trade return standard deviation:
    //   sharpe = (mean_return - risk_free_rate) / std_dev(returns) * sqrt(252)
    double sharpe_ratio = 0.0;
    if (total_trades > 1) {
        double avg_trade_pnl = total_pnl / total_trades;
        sharpe_ratio = avg_trade_pnl * sqrt(static_cast<double>(total_trades)); // Placeholder
    }

    double win_rate = (total_trades > 0) ? (static_cast<double>(winning_trades) / total_trades) * 100.0 : 0.0;
    double max_drawdown_pct = max_drawdown * 100.0; // Convert to percentage (already capped at 100%)

    // Store results
    results[idx].total_pnl = total_pnl;
    results[idx].sharpe_ratio = sharpe_ratio;
    results[idx].max_drawdown = max_drawdown_pct;
    results[idx].total_trades = static_cast<uint64_t>(total_trades);
    results[idx].win_rate = win_rate;
}