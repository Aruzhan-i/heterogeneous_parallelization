// task_2_prefix_sum.cu
// CUDA prefix sum (inclusive scan) с shared memory + много-блочная версия + проверка CPU

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// 1) Scan внутри каждого блока (Blelloch) для 2*BLOCK элементов.
//    Выход: out — просканированный массив
//    block_sums[blockIdx.x] = сумма элементов блока (нужно для склейки блоков)
__global__ void scan_block_blelloch_inclusive(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              float* __restrict__ block_sums,
                                              int n)
{
    extern __shared__ float s[]; // размер = 2*blockDim.x floats

    int tid = threadIdx.x;
    int blockStart = 2 * blockDim.x * blockIdx.x;

    int i1 = blockStart + tid;
    int i2 = blockStart + tid + blockDim.x;

    // Загружаем 2 элемента на поток в shared (если за границей — 0)
    float x1 = (i1 < n) ? in[i1] : 0.0f;
    float x2 = (i2 < n) ? in[i2] : 0.0f;

    s[tid] = x1;
    s[tid + blockDim.x] = x2;

    // --- Up-sweep (reduce) ---
    // stride: 1,2,4,... пока не дойдём до конца
    for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {
        __syncthreads();
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < 2 * blockDim.x) {
            s[idx] += s[idx - stride];
        }
    }

    // Сумма блока лежит в последнем элементе
    if (tid == 0) {
        block_sums[blockIdx.x] = s[2 * blockDim.x - 1];
        // Для exclusive scan по Blelloch нужно обнулить последний
        s[2 * blockDim.x - 1] = 0.0f;
    }

    // --- Down-sweep ---
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

    // Сейчас s[] содержит EXCLUSIVE scan.
    // Делаем INCLUSIVE: out[i] = exclusive[i] + in[i]
    if (i1 < n) out[i1] = s[tid] + x1;
    if (i2 < n) out[i2] = s[tid + blockDim.x] + x2;
}

// 2) Добавляем смещение блока (просканированные суммы блоков) ко всем элементам блока
__global__ void add_block_offsets(float* data,
                                  const float* __restrict__ scanned_block_sums,
                                  int n)
{
    int blockStart = 2 * blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    int i1 = blockStart + tid;
    int i2 = blockStart + tid + blockDim.x;

    // scanned_block_sums — INCLUSIVE scan сумм блоков
    // Для блока 0 offset = 0, для блока k offset = scanned_block_sums[k-1]
    float offset = 0.0f;
    if (blockIdx.x > 0) offset = scanned_block_sums[blockIdx.x - 1];

    if (i1 < n) data[i1] += offset;
    if (i2 < n) data[i2] += offset;
}

// CPU inclusive prefix sum
static std::vector<float> cpu_inclusive_scan(const std::vector<float>& a)
{
    std::vector<float> out(a.size());
    float run = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        run += a[i];
        out[i] = run;
    }
    return out;
}

// GPU multi-block inclusive scan
static void gpu_inclusive_scan(const std::vector<float>& h_in,
                               std::vector<float>& h_out,
                               int blockSize = 256)
{
    int n = (int)h_in.size();
    h_out.resize(n);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Каждый блок обрабатывает 2*blockSize элементов
    int elemsPerBlock = 2 * blockSize;
    int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;

    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));

    // 1) scan по блокам
    size_t shmBytes = elemsPerBlock * sizeof(float);
    scan_block_blelloch_inclusive<<<numBlocks, blockSize, shmBytes>>>(d_in, d_out, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2) если блок один — готово
    if (numBlocks > 1) {
        // Нужно просканировать d_block_sums (inclusive scan), чтобы получить offset-ы блоков
        // Для простоты (и корректности) сделаем это на CPU (для задания обычно ок).
        // Если нужно полностью на GPU — скажи, дам рекурсивный вариант.
        std::vector<float> h_block_sums(numBlocks);
        CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

        auto h_scanned = cpu_inclusive_scan(h_block_sums);

        float* d_scanned = nullptr;
        CUDA_CHECK(cudaMalloc(&d_scanned, numBlocks * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_scanned, h_scanned.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice));

        // 3) добавить offset-ы к каждому блоку
        add_block_offsets<<<numBlocks, blockSize>>>(d_out, d_scanned, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_scanned));
    }

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in));
}

int main()
{
    const int N = 100000; // тест
    std::vector<float> a(N);

    // тестовые данные: 1,1,1,... (тогда prefix[i] = i+1)
    for (int i = 0; i < N; ++i) a[i] = 1.0f;

    // CPU
    auto cpu = cpu_inclusive_scan(a);

    // GPU
    std::vector<float> gpu;
    gpu_inclusive_scan(a, gpu, 256);

    // Проверка
    double maxAbsDiff = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = std::abs((double)cpu[i] - (double)gpu[i]);
        if (d > maxAbsDiff) maxAbsDiff = d;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N = " << N << "\n";
    std::cout << "CPU last = " << cpu.back() << "\n";
    std::cout << "GPU last = " << gpu.back() << "\n";
    std::cout << "Max abs diff = " << maxAbsDiff << "\n";

    const double eps = 1e-2;
    if (maxAbsDiff < eps) std::cout << "✅ OK: префиксная сумма корректна\n";
    else                 std::cout << "❌ FAIL: есть расхождения\n";

    return 0;
}
