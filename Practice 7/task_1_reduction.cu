// task_1_reduction.cu
// CUDA редукция (сумма массива) с использованием shared memory + проверка на CPU

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
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

// Ядро: каждый блок суммирует свой кусок массива -> записывает 1 число в out[blockIdx.x]
__global__ void reduce_sum_shared(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n)
{
    extern __shared__ float sdata[]; // shared memory (размер задаём при запуске ядра)

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 1) Загружаем данные в shared: сразу по 2 элемента на поток (если есть)
    float sum = 0.0f;
    if (i < (unsigned)n) sum += in[i];
    if (i + blockDim.x < (unsigned)n) sum += in[i + blockDim.x];

    sdata[tid] = sum;
    __syncthreads();

    // 2) Редукция внутри блока в shared memory (дерево суммирования)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3) Поток 0 пишет результат блока
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

// Хост-функция: многопроходная редукция до одного числа
float gpu_reduce_sum(const std::vector<float>& h_in, int blockSize = 256)
{
    const int n = (int)h_in.size();

    float* d_in = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int curN = n;
    float* curIn = d_in;

    // будем запускать редукцию, пока не останется 1 элемент
    while (curN > 1) {
        int gridSize = (curN + (blockSize * 2 - 1)) / (blockSize * 2);
        CUDA_CHECK(cudaMalloc(&d_out, gridSize * sizeof(float)));

        size_t shmBytes = blockSize * sizeof(float);
        reduce_sum_shared<<<gridSize, blockSize, shmBytes>>>(curIn, d_out, curN);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // освобождаем старый вход (кроме самого первого d_in)
        if (curIn != d_in) CUDA_CHECK(cudaFree(curIn));

        // следующий проход
        curIn = d_out;
        curN = gridSize;
        d_out = nullptr;
    }

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, curIn, sizeof(float), cudaMemcpyDeviceToHost));

    if (curIn != d_in) CUDA_CHECK(cudaFree(curIn));
    CUDA_CHECK(cudaFree(d_in));

    return h_sum;
}

int main()
{
    // Тестовый массив (можешь менять размер)
    const int N = 100000; // попробуй 16, 1024, 1000000 и т.д.
    std::vector<float> h(N);

    // Заполняем предсказуемыми значениями
    for (int i = 0; i < N; ++i) h[i] = 1.0f; // сумма должна быть N

    // CPU-эталон
    double cpu_sum = 0.0;
    for (float x : h) cpu_sum += (double)x;

    // GPU
    float gpu_sum = gpu_reduce_sum(h, 256);

    // Проверка
    double diff = std::abs(cpu_sum - (double)gpu_sum);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N = " << N << "\n";
    std::cout << "CPU sum = " << cpu_sum << "\n";
    std::cout << "GPU sum = " << gpu_sum << "\n";
    std::cout << "Abs diff = " << diff << "\n";

    // Допуск (из-за float)
    const double eps = 1e-2;
    if (diff < eps) std::cout << "✅ OK: результат корректный\n";
    else           std::cout << "❌ FAIL: результат отличается\n";

    return 0;
}
