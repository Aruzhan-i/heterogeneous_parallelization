#include <iostream>
#include <cuda_runtime.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

// 1. Ядро с использованием только глобальной памяти
__global__ void multiply_global(float *data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * factor;
    }
}

// 2. Ядро с использованием разделяемой памяти
__global__ void multiply_shared(float *data, float factor, int n) {
    __shared__ float temp[THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) {
        // Копирование из глобальной в разделяемую
        temp[tid] = data[idx];
        __syncthreads();

        // Вычисление
        temp[tid] *= factor;
        __syncthreads();

        // Запись обратно в глобальную
        data[idx] = temp[tid];
    }
}

void checkError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(1);
    }
}

int main() {
    size_t size = N * sizeof(float);
    float *h_data = new float[N];
    float *d_data;
    float factor = 2.0f;

    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    checkError(cudaMalloc(&d_data, size), "Malloc failed");

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Создание событий CUDA для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- РАЗОГРЕВ (WARM-UP) ---
    // Первый запуск часто медленный из-за инициализации драйвера
    multiply_global<<<blocks, THREADS_PER_BLOCK>>>(d_data, factor, N);
    cudaDeviceSynchronize();

    // --- ТЕСТ 1: Глобальная память ---
    checkError(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice), "Memcpy failed");
    
    cudaEventRecord(start);
    multiply_global<<<blocks, THREADS_PER_BLOCK>>>(d_data, factor, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms_global = 0;
    cudaEventElapsedTime(&ms_global, start, stop);

    // --- ТЕСТ 2: Разделяемая память ---
    checkError(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice), "Memcpy failed");

    cudaEventRecord(start);
    multiply_shared<<<blocks, THREADS_PER_BLOCK>>>(d_data, factor, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms_shared = 0;
    cudaEventElapsedTime(&ms_shared, start, stop);

    // Вывод результатов
    std::cout << "Array size: " << N << std::endl;
    std::cout << "Global Memory Time: " << ms_global << " ms" << std::endl;
    std::cout << "Shared Memory Time: " << ms_shared << " ms" << std::endl;

    // Очистка ресурсов
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] h_data;
    cudaFree(d_data);

    return 0;
}