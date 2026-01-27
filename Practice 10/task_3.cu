// task_3_hybrid_cpu_gpu.cu                                                 // Имя файла / задача: гибрид CPU+GPU
// UPDATED VERSION (real overlap using cudaStreamWaitEvent instead of per-chunk synchronize) // Обновлённая версия: реальный overlap через cudaStreamWaitEvent вместо синхронизации по каждому чанку
//                                                                          // (пустая строка-комментарий не нужна; оставляем как есть)

// Hybrid algorithm:                                                        // Описание гибридного алгоритма:
//   - CPU processes first half: out[i] = in[i] + 5                          //   - CPU обрабатывает первую половину: прибавляет 5
//   - GPU processes second half: out[i] = in[i] * 2                         //   - GPU обрабатывает вторую половину: умножает на 2
//                                                                          // (пустая строка)

// Requirements covered:                                                    // Какие требования закрыты:
// 1) Hybrid CPU+GPU processing                                              // 1) Гибридная обработка CPU+GPU
// 2) Async transfers (cudaMemcpyAsync) + CUDA streams (2 streams, double buffering) // 2) Асинхронные копии + 2 stream + двойная буферизация
// 3) Profiling: measure H2D, kernel, D2H with cudaEvent + wall time          // 3) Профилирование: H2D/kernel/D2H через cudaEvent + общее wall time
// 4) Optimization reducing overhead: remove per-chunk cudaStreamSynchronize; // 4) Оптимизация: убрать cudaStreamSynchronize на каждый чанк
//    use cudaEvent + cudaStreamWaitEvent to allow overlap + keep pinned host memory. //    использовать события + WaitEvent для overlap и pinned host memory
//                                                                          // (пустая строка)

// Build (Windows):                                                         // Сборка под Windows:
//   nvcc -O3 -Xcompiler /openmp task_3_hybrid_cpu_gpu.cu -o task_3           // nvcc + флаг OpenMP для MSVC
//                                                                          // (пустая строка)

// Build (Linux/macOS):                                                     // Сборка под Linux/macOS:
//   nvcc -O3 -Xcompiler -fopenmp task_3_hybrid_cpu_gpu.cu -o task_3          // nvcc + флаг OpenMP для gcc/clang
//                                                                          // (пустая строка)

// Run:                                                                     // Запуск:
//   task_3 67108864                                                        // Пример: N = 2^26
//                                                                          // (пустая строка)

#include <cuda_runtime.h>                                                  // CUDA runtime API (cudaMallocHost, cudaMemcpyAsync, cudaEvent, streams, ...)
#include <iostream>                                                       // std::cout / std::cerr
#include <iomanip>                                                        // форматирование вывода (fixed, setprecision)
#include <cstdlib>                                                        // std::atoi, std::exit
#include <chrono>                                                         // замер wall time через std::chrono
#include <vector>                                                         // std::vector (в этом коде подключён, но почти не используется)
#include <algorithm>                                                      // std::min

#ifdef _OPENMP                                                            // Если компилируемся с поддержкой OpenMP
#include <omp.h>                                                          // Подключаем OpenMP (omp_get_max_threads, директивы)
#endif                                                                    // Конец условия _OPENMP


#define CUDA_CHECK(call) do {                                                /* макрос: обертка для проверки ошибок CUDA */ \
    cudaError_t err = (call);                                                /* выполняем вызов и сохраняем код ошибки */ \
    if (err != cudaSuccess) {                                                /* если ошибка */ \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)               /* печатаем текст ошибки */ \
                  << " at " << __FILE__ << ":" << __LINE__                   /* печатаем файл и строку */ \
                  << "\n";                                                   /* перевод строки */ \
        std::exit(1);                                                        /* аварийное завершение программы */ \
    }                                                                        /* конец if */ \
} while(0)                                                                   /* конструкция для корректного использования макроса как одной инструкции */

// GPU kernel: multiply by 2                                                // CUDA-ядро: умножение на 2
__global__ void mul2_kernel(const float* __restrict__ in,                  // in: входной массив на device (restrict для оптимизации)
                            float* __restrict__ out,                       // out: выходной массив на device
                            int n)                                         // n: количество элементов
{                                                                          // Начало mul2_kernel
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                       // Глобальный индекс потока
    if (tid < n) out[tid] = in[tid] * 2.0f;                                // Если в пределах: умножаем на 2 и записываем
}                                                                          // Конец mul2_kernel

// CPU part: add 5                                                          // CPU-функция: прибавление 5
void cpu_add5(const float* in, float* out, int n) {                        // in/out: указатели на host, n: длина участка
#ifdef _OPENMP                                                             // Если есть OpenMP
    #pragma omp parallel for                                               // Параллелим цикл по потокам CPU
#endif                                                                     // Конец условия OpenMP
    for (int i = 0; i < n; ++i) out[i] = in[i] + 5.0f;                     // Для каждого элемента: out = in + 5
}                                                                          // Конец cpu_add5

int main(int argc, char** argv) {                                          // Главная функция программы
    int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 26); // default ~67M    // N: из аргумента или 2^26 по умолчанию
    std::cout << "N=" << N << " floats (" << (N * sizeof(float) / (1024.0 * 1024.0)) << " MB)\n"; // Печать N и размера данных в MB

#ifdef _OPENMP                                                             // Если OpenMP включён
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";      // Печать максимального числа потоков OpenMP
#else                                                                      // Иначе
    std::cout << "OpenMP: not enabled\n";                                  // Сообщаем, что OpenMP не включён
#endif                                                                     // Конец условия OpenMP

    // Split: half CPU, half GPU                                            // Делим работу: половина CPU, половина GPU
    int N_cpu = N / 2;                                                     // Размер CPU части (первая половина)
    int N_gpu = N - N_cpu;                                                 // Размер GPU части (оставшиеся элементы)

    // ----------------------- OPT: pinned host memory ----------------------- // Оптимизация: pinned host memory
    // Pinned memory enables true async H2D/D2H and higher bandwidth.        // Pinned (page-locked) память даёт реальные async копии и выше bandwidth
    float *h_in  = nullptr;                                                // Указатель на pinned host input
    float *h_out = nullptr;                                                // Указатель на pinned host output
    CUDA_CHECK(cudaMallocHost(&h_in,  (size_t)N * sizeof(float)));         // Выделяем pinned память под вход
    CUDA_CHECK(cudaMallocHost(&h_out, (size_t)N * sizeof(float)));         // Выделяем pinned память под выход

    // Init input (CPU)                                                      // Инициализация входа на CPU
    for (int i = 0; i < N; ++i) h_in[i] = (float)(i % 1000) * 0.001f;      // Заполняем h_in детерминированными значениями

    // ----------------------- GPU chunking + double buffering ----------------------- // GPU пайплайн: чанки + double buffering
    // Chunk size (elements) for pipelining the GPU half.                    // Размер чанка (элементов) для конвейеризации GPU-половины
    // 4M floats ~ 16MB per chunk is a reasonable default.                   // 4M float ≈ 16MB — разумный дефолт
    const int CHUNK = 1 << 22;                                             // CHUNK = 2^22 элементов (≈4,194,304)
    int chunks = (N_gpu + CHUNK - 1) / CHUNK;                              // Кол-во чанков для GPU части (ceil)

    float *d_in[2]  = {nullptr, nullptr};                                  // Двойной буфер device input: 2 буфера
    float *d_out[2] = {nullptr, nullptr};                                  // Двойной буфер device output: 2 буфера

    CUDA_CHECK(cudaMalloc(&d_in[0],  (size_t)CHUNK * sizeof(float)));      // Выделяем d_in[0] под один чанк
    CUDA_CHECK(cudaMalloc(&d_out[0], (size_t)CHUNK * sizeof(float)));      // Выделяем d_out[0] под один чанк
    CUDA_CHECK(cudaMalloc(&d_in[1],  (size_t)CHUNK * sizeof(float)));      // Выделяем d_in[1] под один чанк
    CUDA_CHECK(cudaMalloc(&d_out[1], (size_t)CHUNK * sizeof(float)));      // Выделяем d_out[1] под один чанк

    cudaStream_t stream[2];                                                // Два CUDA stream для overlap (попеременная работа)
    CUDA_CHECK(cudaStreamCreate(&stream[0]));                              // Создаём stream 0
    CUDA_CHECK(cudaStreamCreate(&stream[1]));                              // Создаём stream 1

    // ----------------------- OPT: event-based dependency (NO per-chunk sync) ------- // Оптимизация: зависимости через event (без sync на каждый чанк)
    // done[s] marks when the previous chunk in stream s finished (after D2H). // done[s] показывает, что предыдущий чанк в stream s завершён (после D2H)
    cudaEvent_t done[2];                                                   // События готовности буферов/стримов
    CUDA_CHECK(cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming));// Создаём event done[0] без тайминга (дешевле)
    CUDA_CHECK(cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming));// Создаём event done[1] без тайминга
    CUDA_CHECK(cudaEventRecord(done[0], stream[0])); // initially "ready"   // Записываем done[0] сразу => stream[0] изначально "свободен"
    CUDA_CHECK(cudaEventRecord(done[1], stream[1]));                       // Аналогично для stream[1]

    // ----------------------- Profiling events (timed) ----------------------- // Тайминги по стадиям: H2D / kernel / D2H
    // We'll accumulate per-chunk times by syncing each stream AFTER recording, // Идея: мерить события на чанках, но синхронизировать только в конце
    // but only once per stream at the end of the pipeline.                  // чтобы не ломать overlap синхронизацией на каждом чанке
    cudaEvent_t h2d_start[2], h2d_stop[2], k_start[2], k_stop[2], d2h_start[2], d2h_stop[2]; // События старта/стопа для каждой стадии и каждого stream
    for (int s = 0; s < 2; ++s) {                                          // Цикл по двум stream
        CUDA_CHECK(cudaEventCreate(&h2d_start[s]));                        // Event: старт H2D
        CUDA_CHECK(cudaEventCreate(&h2d_stop[s]));                         // Event: стоп H2D
        CUDA_CHECK(cudaEventCreate(&k_start[s]));                          // Event: старт kernel
        CUDA_CHECK(cudaEventCreate(&k_stop[s]));                           // Event: стоп kernel
        CUDA_CHECK(cudaEventCreate(&d2h_start[s]));                        // Event: старт D2H
        CUDA_CHECK(cudaEventCreate(&d2h_stop[s]));                         // Event: стоп D2H
    }                                                                      // Конец цикла создания событий

    // We'll store the "last" events of each stream to measure totals at the end. // Комментарий: используем "последние" события каждого stream
    // Also measure total GPU pipeline wall time via events.                 // Плюс мерим общее время GPU пайплайна отдельными событиями
    cudaEvent_t gpu_all_start, gpu_all_stop;                               // События общего времени GPU пайплайна
    CUDA_CHECK(cudaEventCreate(&gpu_all_start));                           // Создаём gpu_all_start
    CUDA_CHECK(cudaEventCreate(&gpu_all_stop));                            // Создаём gpu_all_stop

    float ms_h2d_total = 0.0f;                                             // Оценка суммарного H2D времени (ms)
    float ms_kernel_total = 0.0f;                                          // Оценка суммарного kernel времени (ms)
    float ms_d2h_total = 0.0f;                                             // Оценка суммарного D2H времени (ms)

    // ----------------------- TOTAL WALL TIME -----------------------         // Общий wall time (CPU+GPU) через std::chrono
    auto wall0 = std::chrono::high_resolution_clock::now();               // Засекаем старт wall time

    // ----------------------- CPU compute (first half) ----------------------- // CPU вычисление (первая половина)
    auto cpu0 = std::chrono::high_resolution_clock::now();                // Засекаем начало CPU работы
    cpu_add5(h_in, h_out, N_cpu);                                         // CPU: out = in + 5 для первой половины
    auto cpu1 = std::chrono::high_resolution_clock::now();                // Засекаем конец CPU работы
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu1 - cpu0).count(); // Переводим длительность CPU в миллисекунды

    // ----------------------- GPU pipeline (second half) ----------------------- // GPU пайплайн (вторая половина)
    // Start global GPU timing                                               // Старт общего GPU тайминга
    CUDA_CHECK(cudaEventRecord(gpu_all_start, stream[0]));                 // Записываем gpu_all_start в stream[0] (начало измерения)

    int block = 256;                                                      // Размер блока потоков для kernel

    for (int c = 0; c < chunks; ++c) {                                     // Цикл по чанкам GPU части
        int s = c & 1; // alternate 0/1                                    // Выбираем stream/буфер: чередуем 0 и 1
        int offset = N_cpu + c * CHUNK;                                    // Смещение в общем массиве (начинаем с N_cpu)
        int n_this = std::min(CHUNK, N - offset);                          // Реальный размер чанка (последний может быть меньше)

        // Wait for this stream/buffer to be free (event-based, non-blocking) // Ждём, пока буфер/stream освободится (через event, без CPU-блокировки)
        CUDA_CHECK(cudaStreamWaitEvent(stream[s], done[s], 0));            // Stream s ждёт event done[s] (зависимость внутри GPU)

        // H2D (async)                                                       // Копирование Host->Device (асинхронно)
        CUDA_CHECK(cudaEventRecord(h2d_start[s], stream[s]));              // Ставим метку старта H2D в stream s
        CUDA_CHECK(cudaMemcpyAsync(d_in[s],                                // Асинхронная копия в d_in[s]
                                  h_in + offset,                           // Источник: кусок host input
                                  (size_t)n_this * sizeof(float),          // Размер копии в байтах
                                  cudaMemcpyHostToDevice,                  // Направление: host -> device
                                  stream[s]));                             // В каком stream выполняется операция
        CUDA_CHECK(cudaEventRecord(h2d_stop[s], stream[s]));               // Ставим метку окончания H2D в stream s

        // Kernel (async)                                                    // Запуск ядра (асинхронно)
        int grid = (n_this + block - 1) / block;                           // Кол-во блоков: ceil(n_this / block)
        CUDA_CHECK(cudaEventRecord(k_start[s], stream[s]));                // Метка старта kernel
        mul2_kernel<<<grid, block, 0, stream[s]>>>(d_in[s], d_out[s], n_this); // Запускаем kernel умножения на 2 в stream s
        CUDA_CHECK(cudaEventRecord(k_stop[s], stream[s]));                 // Метка окончания kernel

        // D2H (async)                                                       // Копирование Device->Host (асинхронно)
        CUDA_CHECK(cudaEventRecord(d2h_start[s], stream[s]));              // Метка старта D2H
        CUDA_CHECK(cudaMemcpyAsync(h_out + offset,                         // Асинхронная копия в host output (вторую половину)
                                  d_out[s],                                // Источник: device output буфер
                                  (size_t)n_this * sizeof(float),          // Размер копии в байтах
                                  cudaMemcpyDeviceToHost,                  // Направление: device -> host
                                  stream[s]));                             // В каком stream выполняется
        CUDA_CHECK(cudaEventRecord(d2h_stop[s], stream[s]));               // Метка окончания D2H

        // Mark completion of this chunk in stream s                          // Отмечаем завершение чанка в stream s
        CUDA_CHECK(cudaEventRecord(done[s], stream[s]));                   // Записываем done[s] после D2H => буфер/stream снова "свободен"

        CUDA_CHECK(cudaGetLastError());                                    // Проверяем ошибки запуска kernel (и прочих async операций)
    }                                                                      // Конец цикла по чанкам

    // Stop global GPU timing after both streams are done:                   // Остановка общего GPU тайминга после завершения обоих stream
    // Record stop event in stream 0, but make it wait for stream 1 completion as well. // Хотим учесть завершение stream1 тоже
    // Easiest: synchronize device, then record+sync is fine for total time. // Самый простой способ: cudaDeviceSynchronize и потом измерить стоп
    CUDA_CHECK(cudaDeviceSynchronize());                                   // Ждём завершения всех операций на GPU (оба stream)
    CUDA_CHECK(cudaEventRecord(gpu_all_stop, 0));                          // Записываем stop в default stream (после synchronize)
    CUDA_CHECK(cudaEventSynchronize(gpu_all_stop));                        // Ждём завершения события stop (для корректного чтения времени)

    // Now we can safely read the last per-stream events (they refer to last chunk launched on that stream) // Теперь можно безопасно читать последние события на каждом stream
    // BUT we also want totals across all chunks. We'll approximate totals by summing the per-chunk times // Но хотим totals по всем чанкам — тут используем приближение
    // by replaying the same events per chunk would require arrays; instead we measure: // Чтобы честно суммировать по каждому чанку нужны массивы событий на каждый чанк
    // - total GPU time (gpu_all_start..gpu_all_stop) for end-to-end          // - общее GPU время end-to-end (gpu_all_start..gpu_all_stop)
    // - and estimate composition by timing a representative chunk (optional). // - и оценка долей стадий по репрезентативному чанку (опционально)
    //                                                                          // (пустая строка)
    // To keep it simple and robust, we will compute composition from the *last* chunk per stream. // Для простоты берём breakdown по последнему чанку каждого stream
    // For coursework, it's usually enough to show global GPU time + H2D/D2H overhead dominance. // Для курсовой обычно хватает: total GPU time + доминирование копий
    //                                                                          // (пустая строка)

    // We'll still compute per-stream last-chunk breakdown and scale by chunks/2 (rough estimate). // Посчитаем breakdown последнего чанка и масштабируем (грубая оценка)
    // Better: just report end-to-end GPU time and separately measure pure copy time by a dedicated copy benchmark. // Лучше: отдельно бенчмаркнуть чистые копии
    float last_h2d_ms_s0 = 0.0f, last_k_ms_s0 = 0.0f, last_d2h_ms_s0 = 0.0f; // Времена последнего чанка (stream0): H2D/kernel/D2H
    float last_h2d_ms_s1 = 0.0f, last_k_ms_s1 = 0.0f, last_d2h_ms_s1 = 0.0f; // Времена последнего чанка (stream1): H2D/kernel/D2H

    CUDA_CHECK(cudaEventElapsedTime(&last_h2d_ms_s0, h2d_start[0], h2d_stop[0])); // Время H2D последнего чанка в stream0
    CUDA_CHECK(cudaEventElapsedTime(&last_k_ms_s0,   k_start[0],   k_stop[0]));   // Время kernel последнего чанка в stream0
    CUDA_CHECK(cudaEventElapsedTime(&last_d2h_ms_s0, d2h_start[0], d2h_stop[0])); // Время D2H последнего чанка в stream0

    CUDA_CHECK(cudaEventElapsedTime(&last_h2d_ms_s1, h2d_start[1], h2d_stop[1])); // Время H2D последнего чанка в stream1
    CUDA_CHECK(cudaEventElapsedTime(&last_k_ms_s1,   k_start[1],   k_stop[1]));   // Время kernel последнего чанка в stream1
    CUDA_CHECK(cudaEventElapsedTime(&last_d2h_ms_s1, d2h_start[1], d2h_stop[1])); // Время D2H последнего чанка в stream1

    // Rough estimate: average of last chunk times * number of chunks         // Грубая оценка: среднее по двум stream * число чанков
    float avg_h2d = 0.5f * (last_h2d_ms_s0 + last_h2d_ms_s1);               // Среднее H2D по двум stream
    float avg_k   = 0.5f * (last_k_ms_s0   + last_k_ms_s1);                 // Среднее kernel по двум stream
    float avg_d2h = 0.5f * (last_d2h_ms_s0 + last_d2h_ms_s1);               // Среднее D2H по двум stream

    ms_h2d_total    = avg_h2d * chunks;                                     // Оценка total H2D по всем чанкам
    ms_kernel_total = avg_k   * chunks;                                     // Оценка total kernel по всем чанкам
    ms_d2h_total    = avg_d2h * chunks;                                     // Оценка total D2H по всем чанкам

    float gpu_total_ms = 0.0f;                                              // Переменная для общего GPU времени
    CUDA_CHECK(cudaEventElapsedTime(&gpu_total_ms, gpu_all_start, gpu_all_stop)); // Измеряем общее время GPU пайплайна

    auto wall1 = std::chrono::high_resolution_clock::now();                // Засекаем конец wall time
    double wall_ms = std::chrono::duration<double, std::milli>(wall1 - wall0).count(); // Общее wall время в миллисекундах

    // ----------------------- Report -----------------------                  // Блок вывода результатов
    std::cout << std::fixed << std::setprecision(3);                       // Формат вывода: 3 знака после запятой
    std::cout << "\n=== Profiling results (UPDATED overlap version) ===\n"; // Заголовок отчёта
    std::cout << "CPU compute (first half) : " << cpu_ms << " ms\n";       // Печать времени CPU части
    std::cout << "GPU pipeline total (H2D+K+D2H, overlapped): " << gpu_total_ms << " ms\n"; // Печать общего времени GPU пайплайна (overlapped)
    std::cout << "\nEstimated breakdown (rough, based on avg last-chunk):\n"; // Заголовок breakdown (грубая оценка)
    std::cout << "GPU H2D total (est.)     : " << ms_h2d_total << " ms\n"; // Печать оценки H2D total
    std::cout << "GPU kernel total (est.)  : " << ms_kernel_total << " ms\n"; // Печать оценки kernel total
    std::cout << "GPU D2H total (est.)     : " << ms_d2h_total << " ms\n"; // Печать оценки D2H total
    std::cout << "TOTAL wall time          : " << wall_ms << " ms\n";      // Печать общего wall time

    float est_gpu_total = ms_h2d_total + ms_kernel_total + ms_d2h_total;   // Суммарная оценка GPU времени по стадиям (оценочная)
    if (est_gpu_total > 0.0f) {                                            // Если сумма > 0 (защита от деления на ноль)
        std::cout << "\nOverhead ratios (estimated):\n";                   // Заголовок: доли overhead
        std::cout << "  Transfer overhead (H2D+D2H)/GPU_total = "          // Печать доли трансферов в GPU_total
                  << ( (ms_h2d_total + ms_d2h_total) / est_gpu_total ) << "\n"; // (H2D + D2H) / total
        std::cout << "  Compute share (kernel)/GPU_total     = "           // Печать доли вычислений kernel в GPU_total
                  << ( ms_kernel_total / est_gpu_total ) << "\n";          // kernel / total
    }                                                                      // Конец if est_gpu_total

    // Spot-check correctness                                                 // Быстрая проверка корректности (несколько элементов)
    std::cout << "\nSpot-check:\n";                                        // Заголовок spot-check
    std::cout << "  out[0] (CPU)          = " << h_out[0] << " (expected " << (h_in[0] + 5.0f) << ")\n"; // Проверка: первый элемент (CPU часть)
    std::cout << "  out[N_cpu] (GPU)      = " << h_out[N_cpu] << " (expected " << (h_in[N_cpu] * 2.0f) << ")\n"; // Проверка: первый элемент GPU части
    std::cout << "  out[N-1] (GPU)        = " << h_out[N-1] << " (expected " << (h_in[N-1] * 2.0f) << ")\n"; // Проверка: последний элемент (GPU часть)

    // Cleanup events                                                         // Освобождение событий
    CUDA_CHECK(cudaEventDestroy(gpu_all_start));                           // Уничтожаем gpu_all_start
    CUDA_CHECK(cudaEventDestroy(gpu_all_stop));                            // Уничтожаем gpu_all_stop
    for (int s = 0; s < 2; ++s) {                                          // Цикл по двум stream
        CUDA_CHECK(cudaEventDestroy(done[s]));                             // Уничтожаем done[s]
        CUDA_CHECK(cudaEventDestroy(h2d_start[s]));                        // Уничтожаем h2d_start[s]
        CUDA_CHECK(cudaEventDestroy(h2d_stop[s]));                         // Уничтожаем h2d_stop[s]
        CUDA_CHECK(cudaEventDestroy(k_start[s]));                          // Уничтожаем k_start[s]
        CUDA_CHECK(cudaEventDestroy(k_stop[s]));                           // Уничтожаем k_stop[s]
        CUDA_CHECK(cudaEventDestroy(d2h_start[s]));                        // Уничтожаем d2h_start[s]
        CUDA_CHECK(cudaEventDestroy(d2h_stop[s]));                         // Уничтожаем d2h_stop[s]
    }                                                                      // Конец цикла уничтожения событий

    // Cleanup streams                                                        // Освобождение stream
    CUDA_CHECK(cudaStreamDestroy(stream[0]));                              // Уничтожаем stream 0
    CUDA_CHECK(cudaStreamDestroy(stream[1]));                              // Уничтожаем stream 1

    // Cleanup device buffers                                                 // Освобождение device буферов
    CUDA_CHECK(cudaFree(d_in[0]));                                         // Освобождаем d_in[0]
    CUDA_CHECK(cudaFree(d_out[0]));                                        // Освобождаем d_out[0]
    CUDA_CHECK(cudaFree(d_in[1]));                                         // Освобождаем d_in[1]
    CUDA_CHECK(cudaFree(d_out[1]));                                        // Освобождаем d_out[1]

    // Cleanup pinned host memory                                             // Освобождение pinned host памяти
    CUDA_CHECK(cudaFreeHost(h_in));                                        // Освобождаем pinned h_in
    CUDA_CHECK(cudaFreeHost(h_out));                                       // Освобождаем pinned h_out

    return 0;                                                              // Успешное завершение
}                                                                          // Конец main
