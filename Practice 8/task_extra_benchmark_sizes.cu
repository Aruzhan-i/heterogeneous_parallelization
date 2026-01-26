// bench_hybrid_compare.cu
// Compare CPU(OpenMP) vs GPU(CUDA total) vs HYBRID (CPU half + GPU half concurrently)
// Uses a *compute-heavy* per-element workload to make the comparison meaningful.
// Measures MEDIAN time over multiple repeats for stability.
// Includes checksum to prevent “dead-code elimination”.

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << "\n";                                     \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// ---------------------- Workload (same logic CPU/GPU) ----------------------
// Simple compute-heavy loop (no trig), stable & fast.
// More ITERS => more compute => GPU/hybrid advantage becomes clearer.
__device__ __forceinline__ float heavy_op(float x, int iters) {
    float a = x;
    float b = x * 1.000001f + 0.000001f;
    #pragma unroll 4
    for (int k = 0; k < iters; ++k) {
        a = a * 1.000001f + b * 0.999999f + 0.000001f;
        b = b * 1.0000007f + a * 0.9999993f + 0.0000007f;
    }
    return a + b;
}

__global__ void heavy_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             int n,
                             int iters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = in[idx];
        // Inline same math as device heavy_op
        float a = x;
        float b = x * 1.000001f + 0.000001f;
        #pragma unroll 4
        for (int k = 0; k < iters; ++k) {
            a = a * 1.000001f + b * 0.999999f + 0.000001f;
            b = b * 1.0000007f + a * 0.9999993f + 0.0000007f;
        }
        out[idx] = a + b;
    }
}

// ---------------------- Helpers ----------------------
static double median_ms(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return 0.0;
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static float checksum(const float* arr, int n) {
    // sample a few points to avoid O(n) overhead
    if (n <= 0) return 0.0f;
    int idxs[8] = {0, n/7, 2*n/7, 3*n/7, 4*n/7, 5*n/7, 6*n/7, n-1};
    float s = 0.0f;
    for (int k = 0; k < 8; ++k) s += arr[idxs[k]];
    return s;
}

// CPU full array (OpenMP), wall time
static double run_cpu_openmp(const float* in, float* out, int n, int iters) {
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float x = in[i];
        float a = x;
        float b = x * 1.000001f + 0.000001f;
        for (int k = 0; k < iters; ++k) {
            a = a * 1.000001f + b * 0.999999f + 0.000001f;
            b = b * 1.0000007f + a * 0.9999993f + 0.0000007f;
        }
        out[i] = a + b;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// GPU total: H2D + kernel + D2H (cudaEvents on same stream)
static double run_gpu_total(cudaStream_t stream,
                            float* d_in, float* d_out,
                            const float* h_in, float* h_out,
                            int n, int iters)
{
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    int block = 256;
    int grid  = (n + block - 1) / block;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    heavy_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n, iters);
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return static_cast<double>(ms);
}

// HYBRID: CPU half + GPU half concurrently (wall time)
static double run_hybrid(cudaStream_t stream,
                         float* d_in, float* d_out,
                         const float* h_in, float* h_out,
                         int n, int iters)
{
    int half = n / 2;
    int n_gpu = n - half;
    size_t bytes_gpu = static_cast<size_t>(n_gpu) * sizeof(float);

    int block = 256;
    int grid  = (n_gpu + block - 1) / block;

    auto t0 = std::chrono::high_resolution_clock::now();

    // GPU second half async: H2D -> kernel -> D2H
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in + half, bytes_gpu, cudaMemcpyHostToDevice, stream));
    heavy_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n_gpu, iters);
    CUDA_CHECK(cudaMemcpyAsync(h_out + half, d_out, bytes_gpu, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaGetLastError());

    // CPU first half concurrently
    #pragma omp parallel for
    for (int i = 0; i < half; ++i) {
        float x = h_in[i];
        float a = x;
        float b = x * 1.000001f + 0.000001f;
        for (int k = 0; k < iters; ++k) {
            a = a * 1.000001f + b * 0.999999f + 0.000001f;
            b = b * 1.0000007f + a * 0.9999993f + 0.0000007f;
        }
        h_out[i] = a + b;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    // You can pass custom ITERS and REPEATS:
    //   bench_hybrid_compare.exe 256 25
    int ITERS   = (argc >= 2) ? std::atoi(argv[1]) : 256;
    int REPEATS = (argc >= 3) ? std::atoi(argv[2]) : 25;

    std::vector<int> sizes = {
        100'000, 200'000, 500'000,
        1'000'000, 2'000'000, 5'000'000, 10'000'000
    };

    int maxN = *std::max_element(sizes.begin(), sizes.end());
    size_t maxBytes = static_cast<size_t>(maxN) * sizeof(float);

    // Pinned host buffers (stable async + faster copies)
    float* h_in  = nullptr;
    float* h_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in,  maxBytes));
    CUDA_CHECK(cudaMallocHost(&h_out, maxBytes));

    for (int i = 0; i < maxN; ++i) h_in[i] = static_cast<float>(i % 1024) / 1024.0f;

    // Device buffers (full size to reuse for any N)
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  maxBytes));
    CUDA_CHECK(cudaMalloc(&d_out, maxBytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warm-up (remove CUDA context init from measurements)
    {
        int n = 1'000'000;
        size_t bytes = static_cast<size_t>(n) * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
        heavy_kernel<<<(n+255)/256, 256, 0, stream>>>(d_in, d_out, n, ITERS);
        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
    }

    std::cout << "=== CPU vs GPU vs HYBRID benchmark (compute-heavy) ===\n";
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "ITERS (work per element): " << ITERS << "\n";
    std::cout << "REPEATS (median over): " << REPEATS << "\n\n";

    std::cout << std::left
              << std::setw(12) << "N"
              << std::setw(14) << "CPU(ms)"
              << std::setw(14) << "GPU(ms)"
              << std::setw(14) << "HYBRID(ms)"
              << std::setw(14) << "GPU speedup"
              << std::setw(14) << "HYB speedup"
              << std::setw(12) << "CHK"
              << "\n";
    std::cout << std::string(94, '-') << "\n";

    for (int n : sizes) {
        std::vector<double> t_cpu, t_gpu, t_hyb;
        t_cpu.reserve(REPEATS);
        t_gpu.reserve(REPEATS);
        t_hyb.reserve(REPEATS);

        // CPU
        for (int r = 0; r < REPEATS; ++r) {
            t_cpu.push_back(run_cpu_openmp(h_in, h_out, n, ITERS));
        }
        float chk_cpu = checksum(h_out, n);

        // GPU total
        for (int r = 0; r < REPEATS; ++r) {
            t_gpu.push_back(run_gpu_total(stream, d_in, d_out, h_in, h_out, n, ITERS));
        }
        float chk_gpu = checksum(h_out, n);

        // Hybrid
        for (int r = 0; r < REPEATS; ++r) {
            t_hyb.push_back(run_hybrid(stream, d_in, d_out, h_in, h_out, n, ITERS));
        }
        float chk_hyb = checksum(h_out, n);

        // Use median for stable numbers
        double cpu_ms = median_ms(t_cpu);
        double gpu_ms = median_ms(t_gpu);
        double hyb_ms = median_ms(t_hyb);

        double sp_gpu = cpu_ms / gpu_ms;
        double sp_hyb = cpu_ms / hyb_ms;

        // Print one checksum (they should be close; slight float diffs are ok)
        float chk = (chk_cpu + chk_gpu + chk_hyb) / 3.0f;

        std::cout << std::left
                  << std::setw(12) << n
                  << std::setw(14) << std::fixed << std::setprecision(4) << cpu_ms
                  << std::setw(14) << gpu_ms
                  << std::setw(14) << hyb_ms
                  << std::setw(14) << std::setprecision(3) << sp_gpu
                  << std::setw(14) << sp_hyb
                  << std::setw(12) << std::setprecision(3) << chk
                  << "\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));

    return 0;
}
