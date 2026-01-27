// bench_hybrid_compare.cu                                                     // File name: benchmark comparing CPU vs GPU vs hybrid execution
// Compare CPU(OpenMP) vs GPU(CUDA total) vs HYBRID (CPU half + GPU half concurrently) // High-level goal of the benchmark program
// Uses a *compute-heavy* per-element workload to make the comparison meaningful. // Work per element is intentionally large
// Measures MEDIAN time over multiple repeats for stability.                    // Use median across repeats for robust timing
// Includes checksum to prevent “dead-code elimination”.                        // Simple correctness / optimization guard

#include <cuda_runtime.h>                                                      // CUDA runtime API (malloc, memcpy, streams, events)
#include <omp.h>                                                               // OpenMP header for CPU parallel loops

#include <algorithm>                                                           // std::sort, std::max_element
#include <chrono>                                                              // timing utilities
#include <cstdlib>                                                             // std::exit, std::atoi
#include <iomanip>                                                             // std::setw, std::setprecision, formatting
#include <iostream>                                                            // std::cout, std::cerr
#include <numeric>                                                             // numeric algorithms (not strictly used, but included)
#include <vector>                                                              // std::vector container

#define CUDA_CHECK(call) do {                                  \              // Macro wrapper for CUDA error-checking
    cudaError_t err = (call);                                  \              // Execute CUDA call and store returned error code
    if (err != cudaSuccess) {                                  \              // If error occurred
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \             // Print human-readable CUDA error string
                  << " at " << __FILE__ << ":" << __LINE__      \             // Print file and line number for debugging
                  << "\n";                                     \              // Newline after error message
        std::exit(1);                                          \              // Abort program with non-zero exit
    }                                                          \              // End error branch
} while(0)                                                                   // Ensure macro behaves like a single statement

// ---------------------- Workload (same logic CPU/GPU) ---------------------- // Section header: compute workload functions
// Simple compute-heavy loop (no trig), stable & fast.                          // Heavy computation without slow trig
// More ITERS => more compute => GPU/hybrid advantage becomes clearer.          // Increasing iterations increases GPU advantage
__device__ __forceinline__ float heavy_op(float x, int iters) {                // Device inline function: compute-heavy operation on GPU
    float a = x;                                                               // Initialize accumulator a
    float b = x * 1.000001f + 0.000001f;                                       // Initialize accumulator b with small offset
    #pragma unroll 4                                                           // Suggest compiler unroll the loop by factor 4
    for (int k = 0; k < iters; ++k) {                                          // Loop for compute-heavy iterations
        a = a * 1.000001f + b * 0.999999f + 0.000001f;                         // Update a with weighted combination
        b = b * 1.0000007f + a * 0.9999993f + 0.0000007f;                      // Update b using updated a
    }                                                                          // End loop
    return a + b;                                                              // Return final combined value
}                                                                              // End device function

__global__ void heavy_kernel(const float* __restrict__ in,                     // CUDA kernel: input array pointer (read-only, restricted)
                             float* __restrict__ out,                          // CUDA kernel: output array pointer (write-only, restricted)
                             int n,                                            // CUDA kernel: number of elements
                             int iters)                                        // CUDA kernel: iterations per element
{                                                                              // Begin kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                            // Compute global thread index
    if (idx < n) {                                                             // Bounds check to avoid out-of-range access
        float x = in[idx];                                                     // Load input value
        // Inline same math as device heavy_op                                  // Comment: replicate heavy_op logic directly in kernel
        float a = x;                                                           // Initialize a
        float b = x * 1.000001f + 0.000001f;                                   // Initialize b
        #pragma unroll 4                                                       // Request loop unrolling for performance
        for (int k = 0; k < iters; ++k) {                                      // Loop for compute-heavy iterations
            a = a * 1.000001f + b * 0.999999f + 0.000001f;                     // Update a
            b = b * 1.0000007f + a * 0.9999993f + 0.0000007f;                  // Update b
        }                                                                      // End loop
        out[idx] = a + b;                                                      // Store output result
    }                                                                          // End bounds check
}                                                                              // End kernel

// ---------------------- Helpers ----------------------                        // Section header: helper functions
static double median_ms(std::vector<double>& v) {                              // Compute median of a vector of timings (ms)
    std::sort(v.begin(), v.end());                                             // Sort values ascending
    size_t n = v.size();                                                       // Number of samples
    if (n == 0) return 0.0;                                                    // Edge case: empty vector
    if (n % 2 == 1) return v[n / 2];                                           // If odd, return middle element
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);                                    // If even, average two middle elements
}                                                                              // End median function

static float checksum(const float* arr, int n) {                               // Compute a light checksum from a float array
    // sample a few points to avoid O(n) overhead                               // Avoid summing all elements for performance
    if (n <= 0) return 0.0f;                                                   // Edge case: invalid/empty array
    int idxs[8] = {0, n/7, 2*n/7, 3*n/7, 4*n/7, 5*n/7, 6*n/7, n-1};             // Select 8 sample indices spread across the array
    float s = 0.0f;                                                            // Checksum accumulator
    for (int k = 0; k < 8; ++k) s += arr[idxs[k]];                             // Sum sampled elements
    return s;                                                                  // Return checksum
}                                                                              // End checksum function

// CPU full array (OpenMP), wall time                                          // Section header: CPU benchmark function
static double run_cpu_openmp(const float* in, float* out, int n, int iters) {  // Run full array workload on CPU with OpenMP
    auto t0 = std::chrono::high_resolution_clock::now();                       // Start wall-clock timer

    #pragma omp parallel for                                                   // Parallelize the loop across CPU threads
    for (int i = 0; i < n; ++i) {                                              // Iterate over all elements
        float x = in[i];                                                       // Load input value
        float a = x;                                                           // Initialize accumulator a
        float b = x * 1.000001f + 0.000001f;                                   // Initialize accumulator b
        for (int k = 0; k < iters; ++k) {                                      // Compute-heavy loop
            a = a * 1.000001f + b * 0.999999f + 0.000001f;                     // Update a
            b = b * 1.0000007f + a * 0.9999993f + 0.0000007f;                  // Update b
        }                                                                      // End loop
        out[i] = a + b;                                                        // Write output
    }                                                                          // End parallel for

    auto t1 = std::chrono::high_resolution_clock::now();                       // Stop timer
    return std::chrono::duration<double, std::milli>(t1 - t0).count();         // Return elapsed milliseconds
}                                                                              // End CPU run function

// GPU total: H2D + kernel + D2H (cudaEvents on same stream)                    // Section header: GPU total time function
static double run_gpu_total(cudaStream_t stream,                               // CUDA stream to use for async operations/timing
                            float* d_in, float* d_out,                         // Device buffers: input and output
                            const float* h_in, float* h_out,                   // Host buffers: input and output
                            int n, int iters)                                  // Size and compute iterations
{                                                                              // Begin GPU function
    size_t bytes = static_cast<size_t>(n) * sizeof(float);                     // Bytes to copy for n elements
    int block = 256;                                                           // Threads per block
    int grid  = (n + block - 1) / block;                                       // Number of blocks (ceil division)

    cudaEvent_t ev0, ev1;                                                      // CUDA events for timing
    CUDA_CHECK(cudaEventCreate(&ev0));                                         // Create start event
    CUDA_CHECK(cudaEventCreate(&ev1));                                         // Create stop event

    CUDA_CHECK(cudaEventRecord(ev0, stream));                                  // Record start event into stream
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream)); // Async host->device copy
    heavy_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n, iters);            // Launch compute kernel
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream)); // Async device->host copy
    CUDA_CHECK(cudaEventRecord(ev1, stream));                                  // Record stop event after queued operations
    CUDA_CHECK(cudaEventSynchronize(ev1));                                     // Wait until stop event completes
    CUDA_CHECK(cudaGetLastError());                                            // Check kernel launch errors (async error reporting)

    float ms = 0.0f;                                                           // Timing result in milliseconds
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));                            // Compute elapsed GPU time between events
    CUDA_CHECK(cudaEventDestroy(ev0));                                         // Destroy start event
    CUDA_CHECK(cudaEventDestroy(ev1));                                         // Destroy stop event
    return static_cast<double>(ms);                                            // Return milliseconds as double
}                                                                              // End GPU run function

// HYBRID: CPU half + GPU half concurrently (wall time)                         // Section header: hybrid function
static double run_hybrid(cudaStream_t stream,                                  // CUDA stream for async GPU portion
                         float* d_in, float* d_out,                            // Device buffers for GPU half
                         const float* h_in, float* h_out,                      // Host buffers for input/output
                         int n, int iters)                                     // Total size and iterations
{                                                                              // Begin hybrid function
    int half = n / 2;                                                          // First half to CPU, second half to GPU
    int n_gpu = n - half;                                                      // Number of elements processed by GPU
    size_t bytes_gpu = static_cast<size_t>(n_gpu) * sizeof(float);             // Bytes for GPU half

    int block = 256;                                                           // Threads per block
    int grid  = (n_gpu + block - 1) / block;                                   // Blocks needed for GPU half

    auto t0 = std::chrono::high_resolution_clock::now();                       // Start wall-clock timer

    // GPU second half async: H2D -> kernel -> D2H                              // Comment: GPU pipeline is asynchronous
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in + half, bytes_gpu, cudaMemcpyHostToDevice, stream)); // Copy second half to device
    heavy_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n_gpu, iters);        // Process second half on GPU
    CUDA_CHECK(cudaMemcpyAsync(h_out + half, d_out, bytes_gpu, cudaMemcpyDeviceToHost, stream)); // Copy GPU results back
    CUDA_CHECK(cudaGetLastError());                                            // Check kernel launch errors

    // CPU first half concurrently                                              // Comment: CPU computes while GPU runs
    #pragma omp parallel for                                                   // Parallelize CPU part
    for (int i = 0; i < half; ++i) {                                           // Process first half indices
        float x = h_in[i];                                                     // Load input
        float a = x;                                                           // Initialize a
        float b = x * 1.000001f + 0.000001f;                                   // Initialize b
        for (int k = 0; k < iters; ++k) {                                      // Compute-heavy loop
            a = a * 1.000001f + b * 0.999999f + 0.000001f;                     // Update a
            b = b * 1.0000007f + a * 0.9999993f + 0.0000007f;                  // Update b
        }                                                                      // End compute loop
        h_out[i] = a + b;                                                      // Store CPU result
    }                                                                          // End CPU loop

    CUDA_CHECK(cudaStreamSynchronize(stream));                                 // Wait for GPU stream to finish all queued ops

    auto t1 = std::chrono::high_resolution_clock::now();                       // Stop timer
    return std::chrono::duration<double, std::milli>(t1 - t0).count();         // Return hybrid wall time in ms
}                                                                              // End hybrid function

int main(int argc, char** argv) {                                              // Program entry point
    // You can pass custom ITERS and REPEATS:                                   // Explanation: optional CLI arguments
    //   bench_hybrid_compare.exe 256 25                                        // Example usage: iters=256 repeats=25
    int ITERS   = (argc >= 2) ? std::atoi(argv[1]) : 256;                      // Read ITERS from argv or use default 256
    int REPEATS = (argc >= 3) ? std::atoi(argv[2]) : 25;                       // Read REPEATS from argv or use default 25

    std::vector<int> sizes = {                                                 // Test sizes for N
        100'000, 200'000, 500'000,                                             // Small/medium input sizes
        1'000'000, 2'000'000, 5'000'000, 10'000'000                            // Larger input sizes
    };                                                                         // End sizes vector initialization

    int maxN = *std::max_element(sizes.begin(), sizes.end());                  // Find maximum N to allocate once
    size_t maxBytes = static_cast<size_t>(maxN) * sizeof(float);               // Max bytes for host/device arrays

    // Pinned host buffers (stable async + faster copies)                       // Use page-locked memory for faster H2D/D2H
    float* h_in  = nullptr;                                                    // Host input pointer
    float* h_out = nullptr;                                                    // Host output pointer
    CUDA_CHECK(cudaMallocHost(&h_in,  maxBytes));                              // Allocate pinned host input
    CUDA_CHECK(cudaMallocHost(&h_out, maxBytes));                              // Allocate pinned host output

    for (int i = 0; i < maxN; ++i) h_in[i] = static_cast<float>(i % 1024) / 1024.0f; // Initialize input with repeating pattern

    // Device buffers (full size to reuse for any N)                            // Allocate largest device arrays once
    float *d_in = nullptr, *d_out = nullptr;                                   // Device input/output pointers
    CUDA_CHECK(cudaMalloc(&d_in,  maxBytes));                                  // Allocate device input buffer
    CUDA_CHECK(cudaMalloc(&d_out, maxBytes));                                  // Allocate device output buffer

    cudaStream_t stream;                                                       // CUDA stream handle
    CUDA_CHECK(cudaStreamCreate(&stream));                                     // Create stream for async operations

    // Warm-up (remove CUDA context init from measurements)                     // Warm-up to avoid first-run overhead
    {                                                                          // Begin warm-up scope
        int n = 1'000'000;                                                     // Warm-up size
        size_t bytes = static_cast<size_t>(n) * sizeof(float);                 // Warm-up bytes
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream)); // Copy warm-up data
        heavy_kernel<<<(n+255)/256, 256, 0, stream>>>(d_in, d_out, n, ITERS);   // Launch warm-up kernel
        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream)); // Copy warm-up result back
        CUDA_CHECK(cudaStreamSynchronize(stream));                             // Wait for warm-up to complete
        CUDA_CHECK(cudaGetLastError());                                        // Check for warm-up kernel errors
    }                                                                          // End warm-up scope

    std::cout << "=== CPU vs GPU vs HYBRID benchmark (compute-heavy) ===\n";   // Header output
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";          // Print max OpenMP threads
    std::cout << "ITERS (work per element): " << ITERS << "\n";                // Print iters
    std::cout << "REPEATS (median over): " << REPEATS << "\n\n";               // Print repeats and spacing

    std::cout << std::left                                                      // Left-align output columns
              << std::setw(12) << "N"                                           // Column: array size
              << std::setw(14) << "CPU(ms)"                                     // Column: CPU time
              << std::setw(14) << "GPU(ms)"                                     // Column: GPU time
              << std::setw(14) << "HYBRID(ms)"                                  // Column: hybrid time
              << std::setw(14) << "GPU speedup"                                 // Column: CPU/GPU
              << std::setw(14) << "HYB speedup"                                 // Column: CPU/HYB
              << std::setw(12) << "CHK"                                         // Column: checksum
              << "\n";                                                          // End header line
    std::cout << std::string(94, '-') << "\n";                                  // Print separator line

    for (int n : sizes) {                                                       // Loop over each problem size
        std::vector<double> t_cpu, t_gpu, t_hyb;                                // Timing vectors for repeats
        t_cpu.reserve(REPEATS);                                                 // Reserve memory for CPU timings
        t_gpu.reserve(REPEATS);                                                 // Reserve memory for GPU timings
        t_hyb.reserve(REPEATS);                                                 // Reserve memory for hybrid timings

        // CPU                                                                     // Comment: measure CPU mode
        for (int r = 0; r < REPEATS; ++r) {                                      // Repeat CPU benchmark multiple times
            t_cpu.push_back(run_cpu_openmp(h_in, h_out, n, ITERS));              // Run and store elapsed time
        }                                                                        // End CPU repeats
        float chk_cpu = checksum(h_out, n);                                      // Compute checksum for CPU results

        // GPU total                                                               // Comment: measure GPU total time
        for (int r = 0; r < REPEATS; ++r) {                                      // Repeat GPU benchmark multiple times
            t_gpu.push_back(run_gpu_total(stream, d_in, d_out, h_in, h_out, n, ITERS)); // Run and store elapsed time
        }                                                                        // End GPU repeats
        float chk_gpu = checksum(h_out, n);                                      // Compute checksum for GPU output

        // Hybrid                                                                  // Comment: measure hybrid wall time
        for (int r = 0; r < REPEATS; ++r) {                                      // Repeat hybrid benchmark multiple times
            t_hyb.push_back(run_hybrid(stream, d_in, d_out, h_in, h_out, n, ITERS)); // Run and store elapsed time
        }                                                                        // End hybrid repeats
        float chk_hyb = checksum(h_out, n);                                      // Compute checksum for hybrid output

        // Use median for stable numbers                                           // Comment: median reduces effect of outliers
        double cpu_ms = median_ms(t_cpu);                                        // Median CPU time
        double gpu_ms = median_ms(t_gpu);                                        // Median GPU time
        double hyb_ms = median_ms(t_hyb);                                        // Median hybrid time

        double sp_gpu = cpu_ms / gpu_ms;                                         // Speedup of GPU over CPU
        double sp_hyb = cpu_ms / hyb_ms;                                         // Speedup of hybrid over CPU

        // Print one checksum (they should be close; slight float diffs are ok)    // Comment: GPU/CPU float results can differ slightly
        float chk = (chk_cpu + chk_gpu + chk_hyb) / 3.0f;                        // Average checksum to print one value

        std::cout << std::left                                                   // Left-align row
                  << std::setw(12) << n                                          // Print N
                  << std::setw(14) << std::fixed << std::setprecision(4) << cpu_ms // Print CPU time with 4 decimals
                  << std::setw(14) << gpu_ms                                     // Print GPU time
                  << std::setw(14) << hyb_ms                                     // Print hybrid time
                  << std::setw(14) << std::setprecision(3) << sp_gpu             // Print GPU speedup with 3 decimals
                  << std::setw(14) << sp_hyb                                     // Print hybrid speedup
                  << std::setw(12) << std::setprecision(3) << chk                // Print checksum
                  << "\n";                                                       // End row
    }                                                                            // End sizes loop

    CUDA_CHECK(cudaStreamDestroy(stream));                                       // Destroy CUDA stream
    CUDA_CHECK(cudaFree(d_in));                                                  // Free device input buffer
    CUDA_CHECK(cudaFree(d_out));                                                 // Free device output buffer
    CUDA_CHECK(cudaFreeHost(h_in));                                              // Free pinned host input buffer
    CUDA_CHECK(cudaFreeHost(h_out));                                             // Free pinned host output buffer

    return 0;                                                                    // Exit success
}                                                                                // End main
