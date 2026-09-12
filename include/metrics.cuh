#ifndef METRICS_CUH
#define METRICS_CUH

#include <cstdint>

/**
 * @brief Structure to hold performance metrics for a single parameter set
 */
struct PerformanceMetrics {
    double total_pnl;           ///< Total profit and loss
    double sharpe_ratio;        ///< Annualized Sharpe ratio (risk-free rate = 0)
    double max_drawdown;        ///< Maximum drawdown as percentage (positive value)
    uint64_t total_trades;      ///< Total number of trades executed
    double win_rate;            ///< Win rate as percentage (0-100)
};

/**
 * @brief Structure to hold backtest configuration parameters
 */
struct BacktestConfig {
    double slippage_bps;        ///< Slippage in basis points per trade
    double commission_bps;      ///< Commission in basis points per trade
    uint64_t lookback_window;   ///< Lookback window for indicator calculation
};

#endif // METRICS_CUH