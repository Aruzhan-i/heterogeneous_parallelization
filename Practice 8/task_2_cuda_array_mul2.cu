// task_2_cuda_array_mul2.cu                                                   // имя файла
// Задание 2: Обработка массива на GPU (CUDA)                                  // описание задания
// Шаги: CPU->GPU копия, kernel A[i]*2, GPU->CPU копия, замер времени.         // основные шаги пайплайна
// Замер: (1) total GPU pipeline time: H2D + kernel + D2H (cudaEvent)          // что измеряем (всё вместе)
//        (2) kernel-only time (cudaEvent) — с warm-up и synchronize, чтобы было честно. // что измеряем (только kernel)

#include <cuda_runtime.h>                                                     // CUDA runtime API (cudaMalloc, cudaMemcpy, events, etc.)
#include <iostream>                                                          // std::cout / std::cerr
#include <vector>                                                            // std::vector
#include <iomanip>                                                           // std::fixed, std::setprecision
#include <cstdlib>                                                           // std::exit

#define CUDA_CHECK(call) do {                                                /* макрос: обертка для проверки ошибок CUDA */ \
    cudaError_t err = (call);                                                /* выполняем вызов и сохраняем код ошибки */ \
    if (err != cudaSuccess) {                                                /* если ошибка */ \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)               /* печатаем текст ошибки */ \
                  << " at " << __FILE__ << ":" << __LINE__                   /* печатаем файл и строку */ \
                  << "\n";                                                   /* перевод строки */ \
        std::exit(1);                                                        /* аварийное завершение программы */ \
    }                                                                        /* конец if */ \
} while(0)                                                                   /* конструкция для корректного использования макроса как одной инструкции */

__global__ void mul2_kernel(const float* __restrict__ in,                    // CUDA kernel: входной массив (только чтение), __restrict__ для оптимизации
                            float* __restrict__ out,                         // CUDA kernel: выходной массив (запись результата)
                            int n)                                           // CUDA kernel: размер массива
{                                                                            // начало тела kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                         // глобальный индекс потока (элемент массива)
    if (idx < n) out[idx] = in[idx] * 2.0f;                                  // проверка границ + умножение на 2
}                                                                            // конец kernel

int main() {                                                                 // точка входа в программу
    const int N = 1'000'000;                                                 // размер массива
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);             // размер в байтах для выделения памяти/копий

    std::vector<float> h_in(N), h_out(N);                                    // хостовые (CPU) массивы: вход и выход

    // -------- init on CPU --------                                          // заголовок блока инициализации на CPU
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);             // заполняем входной массив значениями 0..N-1

    float *d_in = nullptr, *d_out = nullptr;                                 // указатели на память GPU (device)
    CUDA_CHECK(cudaMalloc(&d_in, bytes));                                    // выделяем память на GPU под вход
    CUDA_CHECK(cudaMalloc(&d_out, bytes));                                   // выделяем память на GPU под выход

    const int block = 256;                                                   // размер блока (threads per block)
    const int grid  = (N + block - 1) / block;                               // число блоков (округление вверх)

    // -------- events for timing --------                                    // заголовок: события CUDA для тайминга
    cudaEvent_t evStartTotal, evStopTotal, evStartKernel, evStopKernel;      // объявления CUDA events для total и kernel-only замеров
    CUDA_CHECK(cudaEventCreate(&evStartTotal));                              // создаем event начала total-замера
    CUDA_CHECK(cudaEventCreate(&evStopTotal));                               // создаем event конца total-замера
    CUDA_CHECK(cudaEventCreate(&evStartKernel));                             // создаем event начала kernel-замера
    CUDA_CHECK(cudaEventCreate(&evStopKernel));                              // создаем event конца kernel-замера

    // ===================== WARM-UP (прогрев контекста) ===================== // заголовок блока warm-up
    // Первый запуск CUDA часто включает накладные расходы (инициализация контекста/частоты). // пояснение про накладные расходы первого запуска
    // Прогреваем один раз и синхронизируемся, чтобы дальнейшие замеры были "честнее".        // зачем делаем прогрев
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice)); // копируем входные данные CPU->GPU (H2D) для прогрева
    mul2_kernel<<<grid, block>>>(d_in, d_out, N);                             // запускаем kernel один раз (прогрев)
    CUDA_CHECK(cudaGetLastError());                                           // проверяем, не было ли ошибки запуска kernel
    CUDA_CHECK(cudaDeviceSynchronize());                                      // ждем завершения kernel (и всех предыдущих операций)

    // ===================== TOTAL: H2D + kernel + D2H =====================   // заголовок total-замера
    CUDA_CHECK(cudaEventRecord(evStartTotal));                                // записываем event начала total (в текущий stream)

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice)); // копируем данные CPU->GPU (H2D) уже для реального замера

    // ===================== KERNEL-ONLY (точный замер) =====================  // заголовок kernel-only замера
    // Важно:                                                                   // пояснение: важные условия корректного замера
    // 1) evStartKernel записываем прямо перед запуском ядра                     // правило 1: start прямо перед kernel
    // 2) evStopKernel записываем сразу после запуска ядра                       // правило 2: stop сразу после kernel launch
    // 3) synchronize по evStopKernel гарантирует, что ядро завершилось,         // правило 3: синхронизация гарантирует завершение
    //    и elapsedTime измерит только интервал выполнения kernel.               // итог: измеряется только kernel
    CUDA_CHECK(cudaEventRecord(evStartKernel));                                // записываем event старта kernel
    mul2_kernel<<<grid, block>>>(d_in, d_out, N);                              // запускаем kernel: out[i] = in[i]*2
    CUDA_CHECK(cudaEventRecord(evStopKernel));                                 // записываем event конца kernel (после launch)
    CUDA_CHECK(cudaEventSynchronize(evStopKernel));                            // <-- ключевой фикс: ждем завершения kernel по событию
    CUDA_CHECK(cudaGetLastError());                                            // проверяем ошибки (launch/runtime error видим после sync)

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost)); // копируем результат GPU->CPU (D2H)

    CUDA_CHECK(cudaEventRecord(evStopTotal));                                  // записываем event конца total (после D2H)
    CUDA_CHECK(cudaEventSynchronize(evStopTotal));                             // ждем, пока все операции до evStopTotal завершатся

    float msTotal = 0.0f, msKernel = 0.0f;                                     // переменные для времени в миллисекундах
    CUDA_CHECK(cudaEventElapsedTime(&msTotal, evStartTotal, evStopTotal));     // считаем время total: H2D + kernel + D2H
    CUDA_CHECK(cudaEventElapsedTime(&msKernel, evStartKernel, evStopKernel));  // считаем время только kernel

    // -------- verify --------                                                 // заголовок блока проверки результата
    bool ok = true;                                                            // флаг корректности
    for (int i = 0; i < 5; ++i) {                                               // проверяем первые 5 элементов
        if (h_out[i] != h_in[i] * 2.0f) ok = false;                             // если не совпало — ставим FAIL
    }                                                                           // конец проверки первых элементов
    if (h_out[N - 1] != h_in[N - 1] * 2.0f) ok = false;                         // проверяем последний элемент массива

    std::cout << "=== CUDA array processing (fixed kernel timing) ===\n";       // вывод заголовка
    std::cout << "N: " << N << "\n";                                            // вывод N
    std::cout << "Grid: " << grid << ", Block: " << block << "\n";              // вывод параметров запуска kernel
    std::cout << "Total GPU pipeline time (H2D + kernel + D2H): "               // текст для total времени
              << std::fixed << std::setprecision(4) << msTotal << " ms\n";      // форматируем и выводим total время
    std::cout << "Kernel-only time: " << msKernel << " ms\n";                   // выводим время kernel-only
    std::cout << "Check: out[0]=" << std::fixed << std::setprecision(4) << h_out[0] // выводим out[0] с форматированием
              << ", out[N-1]=" << h_out[N - 1]                                  // выводим последний элемент
              << " -> " << (ok ? "OK" : "FAIL") << "\n";                        // выводим OK/FAIL в зависимости от проверки

    // -------- cleanup --------                                                // заголовок блока очистки ресурсов
    CUDA_CHECK(cudaEventDestroy(evStartTotal));                                 // удаляем event start total
    CUDA_CHECK(cudaEventDestroy(evStopTotal));                                  // удаляем event stop total
    CUDA_CHECK(cudaEventDestroy(evStartKernel));                                // удаляем event start kernel
    CUDA_CHECK(cudaEventDestroy(evStopKernel));                                 // удаляем event stop kernel
    CUDA_CHECK(cudaFree(d_in));                                                 // освобождаем память GPU для входа
    CUDA_CHECK(cudaFree(d_out));                                                // освобождаем память GPU для выхода

    return 0;                                                                   // успешное завершение программы
}                                                                               // конец main
