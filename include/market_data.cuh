#ifndef TICK_DATA_SOA_CUH
#define TICK_DATA_SOA_CUH

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
// Define a dummy cudaStream_t for CPU fallback
typedef void* cudaStream_t;
#endif
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

/**
 * @brief Structure of Arrays (SoA) container for tick data using CUDA Unified Memory (when USE_CUDA is ON)
 *        or regular host memory (when USE_CUDA is OFF).
 *
 * This class manages tick data in separate contiguous arrays for optimal memory
 * access patterns on GPU. When USE_CUDA is ON, uses cudaMallocManaged for zero-copy CPU-to-GPU transfer.
 * When USE_CUDA is OFF, uses regular host memory (new[]/delete[]).
 */
class TickDataSoA {
public:
    /**
     * @brief Construct a new TickDataSoA object
     *
     * @param size Number of tick elements to allocate space for
     */
    explicit TickDataSoA(size_t size) : size_(size) {
        if (size == 0) {
            // No allocation needed for zero-size data
            timestamps_ = nullptr;
            prices_ = nullptr;
            volumes_ = nullptr;
            return;
        }

#ifdef USE_CUDA
        // Allocate unified memory for zero-copy access
        cudaError_t err;
        err = cudaMallocManaged(&timestamps_, size * sizeof(uint64_t));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate timestamps: " + std::string(cudaGetErrorString(err)));
        }

        err = cudaMallocManaged(&prices_, size * sizeof(double));
        if (err != cudaSuccess) {
            cudaFree(timestamps_);
            throw std::runtime_error("Failed to allocate prices: " + std::string(cudaGetErrorString(err)));
        }

        err = cudaMallocManaged(&volumes_, size * sizeof(double));
        if (err != cudaSuccess) {
            cudaFree(timestamps_);
            cudaFree(prices_);
            throw std::runtime_error("Failed to allocate volumes: " + std::string(cudaGetErrorString(err)));
        }

        // Initialize to zero
        cudaMemset(timestamps_, 0, size * sizeof(uint64_t));
        cudaMemset(prices_, 0, size * sizeof(double));
        cudaMemset(volumes_, 0, size * sizeof(double));
#else
        // Allocate regular host memory
        timestamps_ = new uint64_t[size];
        prices_ = new double[size];
        volumes_ = new double[size];

        // Initialize to zero
        std::memset(timestamps_, 0, size * sizeof(uint64_t));
        std::memset(prices_, 0, size * sizeof(double));
        std::memset(volumes_, 0, size * sizeof(double));
#endif
    }

    /**
     * @brief Destroy the TickDataSoA object and free memory
     */
    ~TickDataSoA() {
#ifdef USE_CUDA
        if (timestamps_) cudaFree(timestamps_);
        if (prices_) cudaFree(prices_);
        if (volumes_) cudaFree(volumes_);
#else
        delete[] timestamps_;
        delete[] prices_;
        delete[] volumes_;
#endif
    }

    // Delete copy constructor and assignment to prevent double-free
    TickDataSoA(const TickDataSoA&) = delete;
    TickDataSoA& operator=(const TickDataSoA&) = delete;

    // Allow move semantics
    TickDataSoA(TickDataSoA&& other) noexcept
        : timestamps_(other.timestamps_),
          prices_(other.prices_),
          volumes_(other.volumes_),
          size_(other.size_) {
        other.timestamps_ = nullptr;
        other.prices_ = nullptr;
        other.volumes_ = nullptr;
        other.size_ = 0;
    }

    TickDataSoA& operator=(TickDataSoA&& other) noexcept {
        if (this != &other) {
#ifdef USE_CUDA
            // Free existing resources
            if (timestamps_) cudaFree(timestamps_);
            if (prices_) cudaFree(prices_);
            if (volumes_) cudaFree(volumes_);
#else
            // Free existing resources
            delete[] timestamps_;
            delete[] prices_;
            delete[] volumes_;
#endif

            // Transfer ownership
            timestamps_ = other.timestamps_;
            prices_ = other.prices_;
            volumes_ = other.volumes_;
            size_ = other.size_;

            // Nullify source
            other.timestamps_ = nullptr;
            other.prices_ = nullptr;
            other.volumes_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get pointer to timestamps array (host-accessible)
     *
     * @return uint64_t* Pointer to timestamps array
     */
    uint64_t* timestamps() const { return timestamps_; }

    /**
     * @brief Get pointer to prices array (host-accessible)
     *
     * @return double* Pointer to prices array
     */
    double* prices() const { return prices_; }

    /**
     * @brief Get pointer to volumes array (host-accessible)
     *
     * @return double* Pointer to volumes array
     */
    double* volumes() const { return volumes_; }

    /**
     * @brief Get the number of elements
     *
     * @return size_t Number of tick elements
     */
    size_t size() const { return size_; }

    /**
     * @brief Asynchronously prefetch memory block to GPU
     *
     * @param start_index Starting index to prefetch
     * @param count Number of elements to prefetch (0 for all remaining)
     * @param stream CUDA stream for async operation (0 for default stream)
     */
    void prefetch_to_gpu(size_t start_index = 0, size_t count = 0, cudaStream_t stream = 0) const {
#ifdef USE_CUDA
        if (start_index >= size_) return;

        size_t prefetch_count = (count == 0 || start_index + count > size_) ?
                                (size_ - start_index) : count;

        if (prefetch_count == 0) return;

        // Get the current GPU device ID
        int device;
        cudaGetDevice(&device);

        cudaMemPrefetchAsync(timestamps_ + start_index,
                            prefetch_count * sizeof(uint64_t),
                            device, stream);

        cudaMemPrefetchAsync(prices_ + start_index,
                            prefetch_count * sizeof(double),
                            device, stream);

        cudaMemPrefetchAsync(volumes_ + start_index,
                            prefetch_count * sizeof(double),
                            device, stream);
#endif
    }

    /**
     * @brief Asynchronously prefetch memory block to CPU
     *
     * @param start_index Starting index to prefetch
     * @param count Number of elements to prefetch (0 for all remaining)
     * @param stream CUDA stream for async operation (0 for default stream)
     */
    void prefetch_to_cpu(size_t start_index = 0, size_t count = 0, cudaStream_t stream = 0) const {
#ifdef USE_CUDA
        if (start_index >= size_) return;

        size_t prefetch_count = (count == 0 || start_index + count > size_) ?
                                (size_ - start_index) : count;

        if (prefetch_count == 0) return;

        cudaMemPrefetchAsync(timestamps_ + start_index,
                            prefetch_count * sizeof(uint64_t),
                            cudaCpuDeviceId, stream);

        cudaMemPrefetchAsync(prices_ + start_index,
                            prefetch_count * sizeof(double),
                            cudaCpuDeviceId, stream);

        cudaMemPrefetchAsync(volumes_ + start_index,
                            prefetch_count * sizeof(double),
                            cudaCpuDeviceId, stream);
#endif
    }

private:
    uint64_t* timestamps_ = nullptr;  ///< Array of timestamps (nanoseconds since epoch)
    double* prices_ = nullptr;        ///< Array of prices
    double* volumes_ = nullptr;       ///< Array of volumes
    size_t size_ = 0;                 ///< Number of elements
};

#endif // TICK_DATA_SOA_CUH