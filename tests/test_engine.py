#!/usr/bin/env python3
"""
Test script for the GPU-Accelerated Backtesting Engine
"""

import numpy as np
import sys
import os

# Add the build directory to path so we can import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import gpu_engine as ge
    print("Successfully imported gpu_engine module")
except ImportError as e:
    print(f"Failed to import gpu_engine: {e}")
    print("Make sure the module is built and in the Python path")
    sys.exit(1)

def generate_test_data(n_points=1000):
    """Generate synthetic test data"""
    np.random.seed(42)

    # Generate timestamps (nanoseconds)
    timestamps = np.arange(n_points, dtype=np.uint64) * 1000000000  # 1 second intervals

    # Generate prices with some random walk
    returns = np.random.normal(0, 0.01, n_points)  # 1% volatility
    prices = 100 * np.exp(np.cumsum(returns))  # Start at $100

    # Generate volumes
    volumes = np.random.uniform(100, 1000, n_points)

    return timestamps, prices, volumes

def test_basic_functionality():
    """Test basic engine functionality"""
    print("Generating test data...")
    timestamps, prices, volumes = generate_test_data(500)

    print(f"Data shape: {len(timestamps)} points")

    # Create engine — pass numpy arrays directly (no more .ctypes.data)
    print("Creating FastQuantEngine...")
    engine = ge.FastQuantEngine(timestamps, prices, volumes)

    # Run a small parameter sweep
    print("Running parameter sweep...")
    results = engine.run_parameter_sweep(
        fast_window_range=(5, 10),
        slow_window_range=(15, 25),
        slippage_bps=1.5,
        commission_bps=0.5
    )

    print(f"Got {len(results)} results")

    # Print first few results
    for i, result in enumerate(results[:3]):
        print(f"Result {i}: P&L={result.total_pnl:.2f}, Sharpe={result.sharpe_ratio:.2f}, "
              f"DD={result.max_drawdown:.2f}%, Trades={result.total_trades}, Win%={result.win_rate:.1f}%")

    # Basic validation
    assert len(results) > 0, "Should have results"
    assert all(isinstance(r.total_pnl, float) for r in results), "P&L should be float"
    assert all(isinstance(r.total_trades, int) for r in results), "Trades should be int"

    # Verify no result has drawdown > 100% (Issue #1 fix validation)
    for i, r in enumerate(results):
        assert r.max_drawdown <= 100.0, f"Result {i}: drawdown {r.max_drawdown}% exceeds 100%"

    # Verify no zero-result placeholders from skipped combos (Issue #4 fix validation)
    # All results should correspond to valid fast < slow combinations
    expected_count = sum(1 for f in range(5, 11) for s in range(15, 26) if f < s)
    assert len(results) == expected_count, \
        f"Expected {expected_count} valid combos, got {len(results)}"

    print("Basic functionality test passed!")

def test_edge_cases():
    """Test edge cases"""
    print("\nTesting edge cases...")

    # Empty data
    try:
        engine = ge.FastQuantEngine(
            np.array([], dtype=np.uint64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )
        results = engine.run_parameter_sweep((1, 5), (10, 15))
        assert len(results) == 0, "Empty data should yield no results"
        print("Empty data test passed")
    except Exception as e:
        print(f"Empty data test failed: {e}")

    # Single point
    try:
        timestamps = np.array([1000000000], dtype=np.uint64)
        prices = np.array([100.0], dtype=np.float64)
        volumes = np.array([100.0], dtype=np.float64)
        engine = ge.FastQuantEngine(timestamps, prices, volumes)
        results = engine.run_parameter_sweep((1, 2), (5, 10))
        print(f"Single point test: {len(results)} results")
        print("Single point test completed")
    except Exception as e:
        print(f"Single point test failed: {e}")

    # Mismatched array sizes should raise an error
    try:
        engine = ge.FastQuantEngine(
            np.array([1, 2, 3], dtype=np.uint64),
            np.array([100.0, 200.0], dtype=np.float64),
            np.array([10.0, 20.0, 30.0], dtype=np.float64),
        )
        print("Mismatched size test FAILED (should have raised error)")
    except Exception as e:
        print(f"Mismatched size test passed: caught '{e}'")

if __name__ == "__main__":
    print("Testing GPU-Accelerated Backtesting Engine")
    print("=" * 50)

    test_basic_functionality()
    test_edge_cases()

    print("\nAll tests completed!")