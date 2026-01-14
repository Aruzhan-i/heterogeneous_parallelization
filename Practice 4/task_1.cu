// task1_generate_data.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)     \
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << std::endl;                                    \
        std::exit(1);                                              \
    }                                                              \
} while(0)

static void generate_random(std::vector<float>& a, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &x : a) x = dist(rng);
}

int main() {
    const int N = 1'000'000;
    const size_t bytes = N * sizeof(float);

    // 1) Генерируем данные на CPU
    std::vector<float> h_a(N);
    generate_random(h_a);

    // 2) Выделяем память на GPU и копируем
    float *d_a = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    // 3) Небольшая проверка: распечатаем первые 5 элементов (CPU-данные)
    std::cout << "N = " << N << "\nFirst 5 values (host): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(6) << h_a[i] << " ";
    }
    std::cout << "\n";

    CUDA_CHECK(cudaFree(d_a));
    return 0;
}
