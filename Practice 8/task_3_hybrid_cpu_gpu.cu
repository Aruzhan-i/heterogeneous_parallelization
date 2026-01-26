// task_3_hybrid_cpu_gpu.cu
// Задание 3: Гибридная обработка массива
// 1) Первая половина массива обрабатывается на CPU (OpenMP)
// 2) Вторая половина — на GPU (CUDA)
// 3) CPU и GPU работают одновременно (async memcpy + stream + pinned memory)
// 4) Замеряем общее время выполнения (wall time)

#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <chrono>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << "\n";                                     \
        std::exit(1);                                          \
    }                                                          \
} while(0)

__global__ void mul2_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f;
}

int main() {
    const int N = 1'000'000;
    const int half = N / 2;
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);
    const size_t bytes_half = static_cast<size_t>(half) * sizeof(float);

    // ===================== Host memory (PINNED) =====================
    // Pinned (page-locked) память нужна, чтобы cudaMemcpyAsync реально была async.
    float* h_in  = nullptr;
    float* h_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in,  bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));

    // init input
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    // ===================== Device memory for SECOND HALF =====================
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes_half));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_half));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int block = 256;
    const int grid  = (half + block - 1) / block;

    // Optional warm-up (чтобы исключить стоимость инициализации CUDA контекста из замера)
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in + half, bytes_half, cudaMemcpyHostToDevice, stream));
    mul2_kernel<<<grid, block, 0, stream>>>(d_in, d_out, half);
    CUDA_CHECK(cudaMemcpyAsync(h_out + half, d_out, bytes_half, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // ===================== HYBRID TIMING (wall time) =====================
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- GPU part (SECOND HALF) async: H2D -> kernel -> D2H ----
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in + half, bytes_half, cudaMemcpyHostToDevice, stream));
    mul2_kernel<<<grid, block, 0, stream>>>(d_in, d_out, half);
    CUDA_CHECK(cudaMemcpyAsync(h_out + half, d_out, bytes_half, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaGetLastError());

    // ---- CPU part (FIRST HALF) in parallel with GPU work ----
    #pragma omp parallel for
    for (int i = 0; i < half; ++i) {
        h_out[i] = h_in[i] * 2.0f;
    }

    // Wait GPU completion
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;

    // ===================== Verify =====================
    bool ok = true;
    // few samples
    for (int i = 0; i < 5; ++i) {
        if (h_out[i] != h_in[i] * 2.0f) ok = false;
    }
    for (int i = half; i < half + 5; ++i) {
        if (h_out[i] != h_in[i] * 2.0f) ok = false;
    }
    if (h_out[N-1] != h_in[N-1] * 2.0f) ok = false;

    // ===================== Output =====================
    int used_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        used_threads = omp_get_num_threads();
    }

    std::cout << "=== HYBRID CPU(OpenMP) + GPU(CUDA) array processing ===\n";
    std::cout << "N: " << N << " (CPU half: " << half << ", GPU half: " << (N - half) << ")\n";
    std::cout << "CPU threads used: " << used_threads << "\n";
    std::cout << "GPU grid: " << grid << ", block: " << block << "\n";
    std::cout << "Total hybrid wall time: " << std::fixed << std::setprecision(4)
              << ms.count() << " ms\n";
    std::cout << "Check: out[0]=" << std::fixed << std::setprecision(4) << h_out[0]
              << ", out[N-1]=" << h_out[N-1]
              << " -> " << (ok ? "OK" : "FAIL") << "\n";

    // ===================== Cleanup =====================
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));

    return 0;
}
