// task_3_perf_benchmark.cu
// Бенчмарк: GPU vs CPU для reduction + prefix scan, разные N,
// сравнение времени, и оптимизация через pinned memory (page-locked).

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// -------------------------- REDUCTION (shared) --------------------------
__global__ void reduce_sum_shared(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float sum = 0.0f;
    if (i < (unsigned)n) sum += in[i];
    if (i + blockDim.x < (unsigned)n) sum += in[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

float gpu_reduce_sum_device(const float* d_in, int n, int blockSize)
{
    float* curIn = const_cast<float*>(d_in);
    float* d_tmp = nullptr;
    int curN = n;

    // Важно: тут мы не хотим освобождать исходный d_in, поэтому
    // освобождаем только промежуточные буферы.
    bool firstPass = true;

    while (curN > 1) {
        int gridSize = (curN + (blockSize * 2 - 1)) / (blockSize * 2);
        CUDA_CHECK(cudaMalloc(&d_tmp, gridSize * sizeof(float)));

        size_t shmBytes = blockSize * sizeof(float);
        reduce_sum_shared<<<gridSize, blockSize, shmBytes>>>(curIn, d_tmp, curN);
        CUDA_CHECK(cudaGetLastError());

        if (!firstPass) CUDA_CHECK(cudaFree(curIn));
        firstPass = false;

        curIn = d_tmp;
        d_tmp = nullptr;
        curN = gridSize;
    }

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, curIn, sizeof(float), cudaMemcpyDeviceToHost));
    if (!firstPass) CUDA_CHECK(cudaFree(curIn));
    return h_sum;
}

// -------------------------- SCAN (block shared + offsets) --------------------------
// Scan внутри блока (Blelloch) -> inclusive, плюс сохраняем сумму блока.
__global__ void scan_block_blelloch_inclusive(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              float* __restrict__ block_sums,
                                              int n)
{
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int blockStart = 2 * blockDim.x * blockIdx.x;

    int i1 = blockStart + tid;
    int i2 = blockStart + tid + blockDim.x;

    float x1 = (i1 < n) ? in[i1] : 0.0f;
    float x2 = (i2 < n) ? in[i2] : 0.0f;

    s[tid] = x1;
    s[tid + blockDim.x] = x2;

    for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < 2 * blockDim.x) s[idx] += s[idx - stride];
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = s[2 * blockDim.x - 1];
        s[2 * blockDim.x - 1] = 0.0f;
    }

    for (int stride = blockDim.x; stride > 0; stride >>= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < 2 * blockDim.x) {
            float t = s[idx - stride];
            s[idx - stride] = s[idx];
            s[idx] += t;
        }
    }
    __syncthreads();

    if (i1 < n) out[i1] = s[tid] + x1;
    if (i2 < n) out[i2] = s[tid + blockDim.x] + x2;
}

__global__ void add_block_offsets(float* data,
                                  const float* __restrict__ scanned_block_sums,
                                  int n)
{
    int blockStart = 2 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    int i1 = blockStart + tid;
    int i2 = blockStart + tid + blockDim.x;

    float offset = 0.0f;
    if (blockIdx.x > 0) offset = scanned_block_sums[blockIdx.x - 1];

    if (i1 < n) data[i1] += offset;
    if (i2 < n) data[i2] += offset;
}

static std::vector<float> cpu_inclusive_scan(const std::vector<float>& a)
{
    std::vector<float> out(a.size());
    float run = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) { run += a[i]; out[i] = run; }
    return out;
}

static void gpu_inclusive_scan_device(const float* d_in, float* d_out, int n, int blockSize)
{
    int elemsPerBlock = 2 * blockSize;
    int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;

    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));

    size_t shmBytes = elemsPerBlock * sizeof(float);
    scan_block_blelloch_inclusive<<<numBlocks, blockSize, shmBytes>>>(d_in, d_out, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());

    if (numBlocks > 1) {
        // Для простоты: scan сумм блоков на CPU (в бенчмарке это учитываем как часть GPU pipeline)
        std::vector<float> h_block_sums(numBlocks);
        CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

        auto h_scanned = cpu_inclusive_scan(h_block_sums);

        float* d_scanned = nullptr;
        CUDA_CHECK(cudaMalloc(&d_scanned, numBlocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_scanned, h_scanned.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice));

        add_block_offsets<<<numBlocks, blockSize>>>(d_out, d_scanned, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_scanned));
    }

    CUDA_CHECK(cudaFree(d_block_sums));
}

// -------------------------- Timing helpers --------------------------
static float elapsed_ms(cudaEvent_t a, cudaEvent_t b)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

static double ms_cpu(std::chrono::high_resolution_clock::time_point a,
                     std::chrono::high_resolution_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// -------------------------- Main benchmark --------------------------
int main()
{
    std::cout << std::fixed << std::setprecision(3);

    // размеры для теста
    std::vector<int> sizes = {
        1 << 10,   // 1024
        1 << 14,   // 16384
        1 << 18,   // 262144
        1 << 20,   // 1,048,576
        1 << 22    // 4,194,304
    };

    const int blockSize = 256;
    const int warmup = 2;
    const int iters = 5;

    // CUDA events
    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));

    std::cout << "Columns:\n";
    std::cout << "N | CPU_reduce(ms) CPU_scan(ms) | GPU_reduce(ms) GPU_scan(ms) | "
                 "H2D(ms) D2H(ms) (normal) | H2D(ms) D2H(ms) (pinned)\n\n";

    for (int N : sizes) {
        // ---------------- Host data (normal) ----------------
        std::vector<float> h(N);
        for (int i = 0; i < N; ++i) h[i] = 1.0f;

        // CPU reference reduction
        auto t0 = std::chrono::high_resolution_clock::now();
        double cpu_red = 0.0;
        for (float x : h) cpu_red += x;
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_red_ms = ms_cpu(t0, t1);

        // CPU reference scan
        t0 = std::chrono::high_resolution_clock::now();
        auto cpu_scan = cpu_inclusive_scan(h);
        t1 = std::chrono::high_resolution_clock::now();
        double cpu_scan_ms = ms_cpu(t0, t1);

        // ---------------- Device buffers ----------------
        float *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

        // ---------------- Measure copies (normal) ----------------
        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float h2d_normal = elapsed_ms(e0, e1);

        // ---------------- GPU reduction timing ----------------
        // warmup
        for (int k = 0; k < warmup; ++k) {
            (void)gpu_reduce_sum_device(d_in, N, blockSize);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        float gpu_red_ms = 0.0f;
        for (int k = 0; k < iters; ++k) {
            CUDA_CHECK(cudaEventRecord(e0));
            float s = gpu_reduce_sum_device(d_in, N, blockSize);
            (void)s;
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            gpu_red_ms += elapsed_ms(e0, e1);
        }
        gpu_red_ms /= iters;

        // ---------------- GPU scan timing ----------------
        // warmup
        for (int k = 0; k < warmup; ++k) {
            gpu_inclusive_scan_device(d_in, d_out, N, blockSize);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        float gpu_scan_ms = 0.0f;
        for (int k = 0; k < iters; ++k) {
            CUDA_CHECK(cudaEventRecord(e0));
            gpu_inclusive_scan_device(d_in, d_out, N, blockSize);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            gpu_scan_ms += elapsed_ms(e0, e1);
        }
        gpu_scan_ms /= iters;

        // ---------------- D2H (normal) ----------------
        std::vector<float> h_scan_gpu(N);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaMemcpy(h_scan_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));
        float d2h_normal = elapsed_ms(e2, e3);

        // quick correctness (only last value)
        double diff_last = std::abs((double)cpu_scan.back() - (double)h_scan_gpu.back());

        // ---------------- Pinned memory copy timing ----------------
        float* h_pinned = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_pinned, N * sizeof(float))); // pinned host
        for (int i = 0; i < N; ++i) h_pinned[i] = h[i];

        CUDA_CHECK(cudaEventRecord(e0));
        CUDA_CHECK(cudaMemcpy(d_in, h_pinned, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float h2d_pinned = elapsed_ms(e0, e1);

        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaMemcpy(h_pinned, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(e3));
        CUDA_CHECK(cudaEventSynchronize(e3));
        float d2h_pinned = elapsed_ms(e2, e3);

        CUDA_CHECK(cudaFreeHost(h_pinned));

        // ---------------- Print row ----------------
        std::cout << N << " | "
                  << cpu_red_ms << " " << cpu_scan_ms << " | "
                  << gpu_red_ms << " " << gpu_scan_ms << " | "
                  << h2d_normal << " " << d2h_normal << " | "
                  << h2d_pinned << " " << d2h_pinned
                  << " | diff_last=" << diff_last
                  << "\n";

        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_in));
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
    CUDA_CHECK(cudaEventDestroy(e3));

    return 0;
}
