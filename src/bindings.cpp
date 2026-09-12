#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include "../include/engine.cuh"

namespace nb = nanobind;

NB_MODULE(gpu_engine, m) {
    nb::class_<FastQuantEngine>(m, "FastQuantEngine")
        .def("__init__",
             [](FastQuantEngine* engine,
                nb::ndarray<uint64_t, nb::ndim<1>> timestamps,
                nb::ndarray<double, nb::ndim<1>> prices,
                nb::ndarray<double, nb::ndim<1>> volumes) {
                 // Validate array sizes match (Issue #11)
                 size_t size = timestamps.shape(0);
                 if (prices.shape(0) != size || volumes.shape(0) != size) {
                     throw std::invalid_argument(
                         "All arrays must have the same length. Got timestamps=" +
                         std::to_string(size) + ", prices=" +
                         std::to_string(prices.shape(0)) + ", volumes=" +
                         std::to_string(volumes.shape(0)));
                 }
                 // nanobind holds references to the numpy arrays for the duration
                 // of this call, preventing use-after-free (Issue #15)
                 new (engine) FastQuantEngine(
                     timestamps.data(),
                     prices.data(),
                     volumes.data(),
                     size
                 );
             },
             nb::arg("timestamps"),
             nb::arg("prices"),
             nb::arg("volumes"))
        .def("run_parameter_sweep",
             &FastQuantEngine::run_parameter_sweep,
             nb::arg("fast_window_range"),
             nb::arg("slow_window_range"),
             nb::arg("slippage_bps") = 1.5,
             nb::arg("commission_bps") = 0.5);

    // Bind PerformanceMetrics struct
    nb::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def_prop_rw("total_pnl",
                     [](const PerformanceMetrics& self) { return self.total_pnl; },
                     [](PerformanceMetrics& self, double value) { self.total_pnl = value; })
        .def_prop_rw("sharpe_ratio",
                     [](const PerformanceMetrics& self) { return self.sharpe_ratio; },
                     [](PerformanceMetrics& self, double value) { self.sharpe_ratio = value; })
        .def_prop_rw("max_drawdown",
                     [](const PerformanceMetrics& self) { return self.max_drawdown; },
                     [](PerformanceMetrics& self, double value) { self.max_drawdown = value; })
        .def_prop_rw("total_trades",
                     [](const PerformanceMetrics& self) { return self.total_trades; },
                     [](PerformanceMetrics& self, uint64_t value) { self.total_trades = value; })
        .def_prop_rw("win_rate",
                     [](const PerformanceMetrics& self) { return self.win_rate; },
                     [](PerformanceMetrics& self, double value) { self.win_rate = value; });
}