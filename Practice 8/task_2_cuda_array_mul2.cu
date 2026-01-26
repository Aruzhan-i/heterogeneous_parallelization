// task_2_cuda_array_mul2.cu
// Задание 2: Обработка массива на GPU (CUDA)
// Шаги: CPU->GPU копия, kernel A[i]*2, GPU->CPU копия, замер времени.
// Замер: (1) total GPU pipeline time: H2D + kernel + D2H (cudaEvent)
//        (2) kernel-only time (cudaEvent) — с warm-up и synchronize, чтобы было честно.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>

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
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> h_in(N), h_out(N);

    // -------- init on CPU --------
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    const int block = 256;
    const int grid  = (N + block - 1) / block;

    // -------- events for timing --------
    cudaEvent_t evStartTotal, evStopTotal, evStartKernel, evStopKernel;
    CUDA_CHECK(cudaEventCreate(&evStartTotal));
    CUDA_CHECK(cudaEventCreate(&evStopTotal));
    CUDA_CHECK(cudaEventCreate(&evStartKernel));
    CUDA_CHECK(cudaEventCreate(&evStopKernel));

    // ===================== WARM-UP (прогрев контекста) =====================
    // Первый запуск CUDA часто включает накладные расходы (инициализация контекста/частоты).
    // Прогреваем один раз и синхронизируемся, чтобы дальнейшие замеры были "честнее".
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    mul2_kernel<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ===================== TOTAL: H2D + kernel + D2H =====================
    CUDA_CHECK(cudaEventRecord(evStartTotal));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // ===================== KERNEL-ONLY (точный замер) =====================
    // Важно:
    // 1) evStartKernel записываем прямо перед запуском ядра
    // 2) evStopKernel записываем сразу после запуска ядра
    // 3) synchronize по evStopKernel гарантирует, что ядро завершилось,
    //    и elapsedTime измерит только интервал выполнения kernel.
    CUDA_CHECK(cudaEventRecord(evStartKernel));
    mul2_kernel<<<grid, block>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(evStopKernel));
    CUDA_CHECK(cudaEventSynchronize(evStopKernel)); // <-- ключевой фикс
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(evStopTotal));
    CUDA_CHECK(cudaEventSynchronize(evStopTotal));

    float msTotal = 0.0f, msKernel = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, evStartTotal, evStopTotal));
    CUDA_CHECK(cudaEventElapsedTime(&msKernel, evStartKernel, evStopKernel));

    // -------- verify --------
    bool ok = true;
    for (int i = 0; i < 5; ++i) {
        if (h_out[i] != h_in[i] * 2.0f) ok = false;
    }
    if (h_out[N - 1] != h_in[N - 1] * 2.0f) ok = false;

    std::cout << "=== CUDA array processing (fixed kernel timing) ===\n";
    std::cout << "N: " << N << "\n";
    std::cout << "Grid: " << grid << ", Block: " << block << "\n";
    std::cout << "Total GPU pipeline time (H2D + kernel + D2H): "
              << std::fixed << std::setprecision(4) << msTotal << " ms\n";
    std::cout << "Kernel-only time: " << msKernel << " ms\n";
    std::cout << "Check: out[0]=" << std::fixed << std::setprecision(4) << h_out[0]
              << ", out[N-1]=" << h_out[N - 1]
              << " -> " << (ok ? "OK" : "FAIL") << "\n";

    // -------- cleanup --------
    CUDA_CHECK(cudaEventDestroy(evStartTotal));
    CUDA_CHECK(cudaEventDestroy(evStopTotal));
    CUDA_CHECK(cudaEventDestroy(evStartKernel));
    CUDA_CHECK(cudaEventDestroy(evStopKernel));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
