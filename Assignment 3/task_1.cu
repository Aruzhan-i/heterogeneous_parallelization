// vec_mul_compare.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

#ifdef _WIN32
  #include <windows.h>
#endif

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)    \
                  << " at " << __FILE__ << ":" << __LINE__        \
                  << std::endl;                                   \
        std::exit(1);                                             \
    }                                                             \
} while(0)

// =======================
// 1) Только глобальная память
// =======================
__global__ void mul_global(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * k;
}

// =======================
// 2) Shared memory
// =======================
__global__ void mul_shared(const float* __restrict__ in,
                           float* __restrict__ out,
                           float k, int n)
{
    extern __shared__ float sh[]; // shmem = blockDim.x * sizeof(float)
    int g = blockIdx.x * blockDim.x + threadIdx.x;

    if (g < n) sh[threadIdx.x] = in[g];
    __syncthreads();

    if (g < n) sh[threadIdx.x] *= k;
    __syncthreads();

    if (g < n) out[g] = sh[threadIdx.x];
}

// Универсальный таймер для kernel
template <typename Kernel, typename... Args>
float time_kernel_avg(Kernel kernel,
                      dim3 grid, dim3 block,
                      size_t shmem_bytes,
                      int warmup, int iters,
                      Args... args)
{
    // Прогрев (важно для честного замера)
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block, shmem_bytes>>>(args...);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block, shmem_bytes>>>(args...);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters; // среднее время одного запуска (мс)
}

int main()
{
#ifdef _WIN32
    // Делает консоль Windows UTF-8 (чтобы русский печатался нормально)
    SetConsoleOutputCP(CP_UTF8);
#endif

    const int N = 1'000'000;
    const float k = 2.5f;

    std::vector<float> h_in(N), h_out_g(N), h_out_s(N);
    for (int i = 0; i < N; ++i) h_in[i] = i * 0.001f;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    const int WARMUP = 50;
    const int ITERS  = 1000;

    // 1) global
    float t_global = time_kernel_avg(
        mul_global, grid, block,
        0, WARMUP, ITERS,
        d_in, d_out, k, N
    );
    CUDA_CHECK(cudaMemcpy(h_out_g.data(), d_out,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // 2) shared
    size_t shmem = BLOCK * sizeof(float);
    float t_shared = time_kernel_avg(
        mul_shared, grid, block,
        shmem, WARMUP, ITERS,
        d_in, d_out, k, N
    );
    CUDA_CHECK(cudaMemcpy(h_out_s.data(), d_out,
                          N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Проверка корректности (несколько элементов)
    bool ok = true;
    for (int i = 0; i < 20; ++i) {
        float ref = h_in[i] * k;
        if (std::fabs(h_out_g[i] - ref) > 1e-5f) ok = false;
        if (std::fabs(h_out_s[i] - ref) > 1e-5f) ok = false;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Размер массива: " << N << "\n";
    std::cout << "Потоков в блоке: " << BLOCK << ", блоков: " << grid.x << "\n";
    std::cout << "Замер: warmup=" << WARMUP << ", iters=" << ITERS << "\n\n";

    std::cout << "Время (глобальная память), среднее: " << t_global << " мс\n";
    std::cout << "Время (shared память),     среднее: " << t_shared << " мс\n";
    std::cout << "Отношение (global/shared): " << (t_global / t_shared) << "x\n";
    std::cout << "Проверка: " << (ok ? "OK" : "FAILED") << "\n";

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
