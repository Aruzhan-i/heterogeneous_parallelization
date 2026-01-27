// task_2.cu                                                               // Имя файла / задача 2
// Practice 10 - Task 2: GPU memory access patterns (CUDA)                  // Практика 10: паттерны доступа к памяти на GPU (CUDA)
// Requirements:                                                           // Требования задания:
//  1) Two kernels:                                                         //  1) Два ядра (kernel):
//     a) efficient (coalesced) global memory access                         //     a) эффективный доступ (coalesced) к глобальной памяти
//     b) inefficient (non-coalesced) global memory access (GATHER pattern)  //     b) неэффективный доступ (non-coalesced) (паттерн GATHER)
//  2) Timing with cudaEvent                                                //  2) Замер времени через cudaEvent
//  3) Optimizations:                                                       //  3) Оптимизации:
//     a) shared memory (tile load + permute in shared)                      //     a) shared memory (загрузка тайла + перестановка в shared)
//     b) change thread organization (grid-stride loop)                      //     b) изменение организации потоков (grid-stride loop)
//  4) Compare results, conclude impact of memory access pattern.           //  4) Сравнить результаты и сделать вывод о влиянии доступа к памяти.
//
// Build:                                                                   // Сборка:
//   nvcc -O3 task_2.cu -o task_2                                            // Команда компиляции nvcc с оптимизацией -O3
//
// Run:                                                                     // Запуск:
//   task_2 67108864                                                        // Пример запуска с N=2^26
//
// Notes:                                                                   // Примечания:
// - For the non-coalesced "gather" kernel we require N to be a power of two // - Для gather-ядра N должно быть степенью двойки
//   to use fast wraparound: src = (tid * STRIDE) & (N-1).                   //   чтобы использовать быстрое "зацикливание" по маске (N-1)

#include <cuda_runtime.h>                                                  // CUDA runtime API (cudaMalloc, cudaMemcpy, cudaEvent и т.д.)
#include <iostream>                                                       // std::cout / std::cerr
#include <vector>                                                         // std::vector для массива на host
#include <iomanip>                                                        // форматированный вывод
#include <cstdlib>                                                        // std::atoi, std::exit
#include <cstdint>                                                        // фиксированные целые типы (необязательно, но подключено)

#define CUDA_CHECK(call) do {                                                /* макрос: обертка для проверки ошибок CUDA */ \
    cudaError_t err = (call);                                                /* выполняем вызов и сохраняем код ошибки */ \
    if (err != cudaSuccess) {                                                /* если ошибка */ \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)               /* печатаем текст ошибки */ \
                  << " at " << __FILE__ << ":" << __LINE__                   /* печатаем файл и строку */ \
                  << "\n";                                                   /* перевод строки */ \
        std::exit(1);                                                        /* аварийное завершение программы */ \
    }                                                                        /* конец if */ \
} while(0)                                                                   /* конструкция для корректного использования макроса как одной инструкции */

static bool is_power_of_two(int x) {                                       // Функция: проверка, является ли число степенью двойки
    return x > 0 && ( (x & (x - 1)) == 0 );                                // Классическая проверка степени двойки через битовую магию
}                                                                          // Конец функции is_power_of_two

// -------------------- 1) COALESCED (good global access) -------------------- // 1) Coalesced доступ (хороший, последовательный)
__global__ void kernel_coalesced(const float* __restrict__ in,             // Ядро: читает входной массив (только чтение, restrict для оптимизации)
                                 float* __restrict__ out,                 // Ядро: пишет в выходной массив
                                 int n)                                   // Ядро: размер массива
{                                                                          // Начало тела kernel_coalesced
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                       // Глобальный индекс потока
    if (tid < n) {                                                         // Проверка выхода за границы
        float x = in[tid];                 // contiguous reads (coalesced)  // Чтение подряд: потоки warp читают соседние элементы => coalescing
        out[tid] = x * 1.000001f + 2.0f;   // contiguous writes (coalesced) // Запись подряд: тоже coalesced запись
    }                                                                      // Конец if
}                                                                          // Конец kernel_coalesced

// -------------------- 2) NON-COALESCED (bad global access) ----------------- // 2) Non-coalesced доступ (плохой, нерегулярный)
// "Gather": each thread writes out[tid] contiguously, but reads in[src] where // Gather: запись out[tid] подряд, но чтение in[src] разреженное
// src jumps by STRIDE across threads in a warp => poor coalescing for reads.   // src прыгает с шагом STRIDE => плохой coalescing чтений
// N must be power-of-two for the bitmask wraparound.                           // N должен быть степенью 2 для wrap-around через маску
template<int STRIDE>                                                      // Шаблон ядра: STRIDE задаётся на этапе компиляции
__global__ void kernel_gather_noncoalesced(const float* __restrict__ in,   // Ядро gather: входной массив
                                           float* __restrict__ out,       // Ядро gather: выходной массив
                                           int n)                         // Ядро gather: размер массива
{                                                                          // Начало kernel_gather_noncoalesced
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                       // Глобальный индекс потока
    if (tid < n) {                                                         // Проверка границ
        int src = (tid * STRIDE) & (n - 1); // wrap to [0..n-1]             // Индекс чтения с wrap-around (быстро через & при n=2^k)
        float x = in[src];                  // scattered reads (non-coalesced) // Разреженное чтение: плохой coalescing
        out[tid] = x * 1.000001f + 2.0f;    // contiguous writes            // Запись подряд (coalesced), но чтение — нет
    }                                                                      // Конец if
}                                                                          // Конец kernel_gather_noncoalesced

// -------------------- 3) OPT A: SHARED MEMORY ----------------------------- // 3) Оптимизация A: shared memory
// Idea: make global memory access coalesced (tile load/store),              // Идея: сделать global доступ coalesced (загрузка/выгрузка тайла)
// then do "bad" access pattern in shared memory (fast).                    // а "плохую" перестановку выполнить в shared памяти (быстро)
// This demonstrates how shared memory can mitigate irregular access         // Демонстрация: shared может смягчать нерегулярный доступ
// *within a tile* (not true random global gather).                         // только внутри тайла (это не произвольный global gather)
template<int STRIDE>                                                      // STRIDE также compile-time константа
__global__ void kernel_shared_permute(const float* __restrict__ in,        // Ядро: вход
                                      float* __restrict__ out,            // Ядро: выход
                                      int n)                              // Ядро: размер
{                                                                          // Начало kernel_shared_permute
    extern __shared__ float sh[]; // blockDim.x floats                      // Shared массив размера blockDim.x (задаётся при запуске)

    int g = blockIdx.x * blockDim.x + threadIdx.x;                         // Глобальный индекс элемента
    int t = threadIdx.x;                                                   // Локальный индекс внутри блока

    // Coalesced global load:                                               // Coalesced загрузка из global
    if (g < n) sh[t] = in[g];                                              // Каждый поток грузит свой элемент подряд => coalescing
    __syncthreads();                                                       // Барьер: ждём пока все загрузят в shared

    // "Bad" pattern inside shared:                                         // "Плохой" паттерн внутри shared (но shared быстрый)
    // blockDim must be power-of-two for &: we will launch with 256 by default. // blockDim должен быть степенью 2 из-за маски (&), BLOCK=256
    int src = (t * STRIDE) & (blockDim.x - 1);                             // Индекс чтения внутри shared с wrap-around по размеру блока
    float x = sh[src];                                                     // Читаем из shared по разреженному индексу

    // Coalesced global store:                                              // Coalesced запись в global
    if (g < n) out[g] = x * 1.000001f + 2.0f;                              // Запись подряд в out[g]
}                                                                          // Конец kernel_shared_permute

// -------------------- 4) OPT B: THREAD ORGANIZATION (grid-stride) ---------- // 4) Оптимизация B: grid-stride loop
// Same coalesced access, but grid-stride loop can improve occupancy/latency hiding. // Доступ coalesced, но grid-stride улучшает скрытие латентности
__global__ void kernel_coalesced_gridstride(const float* __restrict__ in,  // Ядро: вход
                                            float* __restrict__ out,      // Ядро: выход
                                            int n)                        // Ядро: размер
{                                                                          // Начало kernel_coalesced_gridstride
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;                      // Глобальный индекс первого элемента
    int step = gridDim.x * blockDim.x;                                     // Шаг по сетке (stride): сколько потоков всего в гриде

    for (int i = tid; i < n; i += step) {                                  // Grid-stride цикл: поток обрабатывает несколько элементов
        float x = in[i];                                                   // Coalesced чтение (внутри каждой итерации)
        out[i] = x * 1.000001f + 2.0f;                                     // Coalesced запись
    }                                                                      // Конец for
}                                                                          // Конец kernel_coalesced_gridstride

// -------------------- TIMING HARNESS (cudaEvent) -------------------------- // Обвязка для замера времени через cudaEvent
using launch_fn = void (*)(const float*, float*, int, cudaStream_t);       // Тип: указатель на функцию-лаунчер kernel (с stream)

float run_and_time(launch_fn launch,                                       // Функция: запускает kernel много раз и измеряет среднее время
                   const float* d_in, float* d_out, int n,                 // Указатели на device вход/выход + размер n
                   int warmup = 5, int iters = 50,                         // Кол-во прогревочных запусков и измерительных итераций
                   bool debug_sync = false)                                // debug_sync: если true — синхронизация после каждого запуска (для отладки)
{                                                                          // Начало функции run_and_time
    cudaEvent_t start, stop;                                               // CUDA события для тайминга
    CUDA_CHECK(cudaEventCreate(&start));                                   // Создаём event start
    CUDA_CHECK(cudaEventCreate(&stop));                                    // Создаём event stop

    cudaStream_t stream;                                                   // CUDA stream для асинхронных запусков
    CUDA_CHECK(cudaStreamCreate(&stream));                                 // Создаём stream

    // Warmup                                                              // Прогрев: чтобы исключить влияние первого запуска/кэшей/частот
    for (int i = 0; i < warmup; ++i) {                                     // Цикл прогрева
        launch(d_in, d_out, n, stream);                                    // Запускаем kernel через функцию-лаунчер
        if (debug_sync) {                                                  // Если включён режим отладки
            CUDA_CHECK(cudaStreamSynchronize(stream));                     // Синхронизация (ждём завершения kernel)
            CUDA_CHECK(cudaGetLastError());                                // Проверяем последнюю CUDA ошибку
        }                                                                  // Конец if debug_sync
    }                                                                      // Конец warmup цикла
    CUDA_CHECK(cudaStreamSynchronize(stream));                             // Финальная синхронизация после прогрева (важно перед таймингом)

    CUDA_CHECK(cudaEventRecord(start, stream));                            // Записываем событие start в stream (старт тайминга)
    for (int i = 0; i < iters; ++i) {                                      // Цикл замеров: запускаем kernel iters раз
        launch(d_in, d_out, n, stream);                                    // Запускаем kernel
        if (debug_sync) {                                                  // Если debug_sync
            CUDA_CHECK(cudaStreamSynchronize(stream));                     // Синхронизируем stream
            CUDA_CHECK(cudaGetLastError());                                // Проверяем ошибки
        }                                                                  // Конец if debug_sync
    }                                                                      // Конец цикла iters
    CUDA_CHECK(cudaEventRecord(stop, stream));                             // Записываем событие stop в stream (конец тайминга)
    CUDA_CHECK(cudaEventSynchronize(stop));                                // Ждём выполнения stop event => все kernel завершились

    float ms_total = 0.0f;                                                 // Переменная для общего времени в миллисекундах
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));              // Вычисляем время между start и stop (ms)

    CUDA_CHECK(cudaStreamDestroy(stream));                                 // Уничтожаем stream (освобождаем ресурсы)
    CUDA_CHECK(cudaEventDestroy(start));                                   // Уничтожаем event start
    CUDA_CHECK(cudaEventDestroy(stop));                                    // Уничтожаем event stop

    return ms_total / iters; // ms per launch                               // Возвращаем среднее время одного запуска kernel
}                                                                          // Конец run_and_time

// -------------------- LAUNCHERS ------------------------------------------- // Лаунчеры: функции запуска ядра с <<<grid, block, shmem, stream>>>
static constexpr int BLOCK = 256; // power-of-two (needed for shared_permute) // Размер блока 256 (степень 2 нужна для shared_permute)

void launch_coalesced(const float* in, float* out, int n, cudaStream_t s) { // Лаунчер kernel_coalesced
    int grid = (n + BLOCK - 1) / BLOCK;                                    // Кол-во блоков: ceil(n / BLOCK)
    kernel_coalesced<<<grid, BLOCK, 0, s>>>(in, out, n);                   // Запуск kernel_coalesced в stream s
}                                                                          // Конец launch_coalesced

void launch_gather32(const float* in, float* out, int n, cudaStream_t s) { // Лаунчер gather с STRIDE=32
    int grid = (n + BLOCK - 1) / BLOCK;                                    // Вычисляем grid
    kernel_gather_noncoalesced<32><<<grid, BLOCK, 0, s>>>(in, out, n);     // Запуск шаблонного gather kernel (stride=32)
}                                                                          // Конец launch_gather32

void launch_gather128(const float* in, float* out, int n, cudaStream_t s) { // Лаунчер gather с STRIDE=128
    int grid = (n + BLOCK - 1) / BLOCK;                                    // Вычисляем grid
    kernel_gather_noncoalesced<128><<<grid, BLOCK, 0, s>>>(in, out, n);    // Запуск gather kernel (stride=128)
}                                                                          // Конец launch_gather128

void launch_shared32(const float* in, float* out, int n, cudaStream_t s) { // Лаунчер shared_permute с STRIDE=32
    int grid = (n + BLOCK - 1) / BLOCK;                                    // Вычисляем grid
    size_t shmem = (size_t)BLOCK * sizeof(float);                          // Shared память: BLOCK float'ов
    kernel_shared_permute<32><<<grid, BLOCK, shmem, s>>>(in, out, n);      // Запуск shared kernel с динамической shared памятью
}                                                                          // Конец launch_shared32

void launch_gridstride(const float* in, float* out, int n, cudaStream_t s) { // Лаунчер grid-stride ядра
    // Choose a grid size using occupancy hint                              // Комментарий: выбираем grid по occupancy (подсказка CUDA runtime)
    int blocksPerSM = 0;                                                   // Переменная: активные блоки на SM
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(              // Вычисляем максимальные активные блоки на SM
        &blocksPerSM, kernel_coalesced_gridstride, BLOCK, 0));             // Для kernel_coalesced_gridstride, block=BLOCK, shared=0

    cudaDeviceProp prop{};                                                 // Структура характеристик устройства
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));                         // Получаем свойства GPU устройства 0
    int grid = blocksPerSM * prop.multiProcessorCount;                     // Итоговый grid = blocksPerSM * число SM

    kernel_coalesced_gridstride<<<grid, BLOCK, 0, s>>>(in, out, n);        // Запуск grid-stride ядра в stream s
}                                                                          // Конец launch_gridstride

// -------------------- MAIN ------------------------------------------------- // Главная функция программы
int main(int argc, char** argv) {                                          // Точка входа main
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 26); // default 67,108,864 // Размер массива: argv[1] или дефолт 2^26
    if (!is_power_of_two(n)) {                                             // Проверяем, что n степень двойки
        std::cerr << "ERROR: N must be a power of two for gather kernels.\n"; // Сообщаем ошибку
        std::cerr << "Try N=67108864 (2^26)\n";                            // Подсказываем корректное значение
        return 1;                                                          // Выход с ошибкой
    }                                                                      // Конец проверки n

    std::cout << "N=" << n << " floats (" << (n * sizeof(float) / (1024.0 * 1024.0)) << " MB)\n"; // Печатаем N и размер массива в MB

    // Host init                                                            // Инициализация данных на host (CPU)
    std::vector<float> h(n);                                               // Host массив h из n элементов
    for (int i = 0; i < n; ++i) h[i] = (float)(i % 1000) * 0.001f;         // Заполняем h детерминированными числами (для проверки)

    // Device alloc                                                         // Выделение памяти на GPU
    float *d_in = nullptr, *d_out = nullptr;                               // Device указатели: вход и выход
    CUDA_CHECK(cudaMalloc(&d_in,  (size_t)n * sizeof(float)));             // cudaMalloc для входного массива на device
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)n * sizeof(float)));             // cudaMalloc для выходного массива на device
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice)); // Копируем host->device

    // Set true only if you see illegal access and need exact location      // Включай true только при отладке illegal access для точного места
    bool debug_sync = false;                                               // Флаг синхронизации для отладки (false = быстрее)

    // Timings                                                              // Блок замеров времени разных вариантов
    float t_coal  = run_and_time(launch_coalesced, d_in, d_out, n, 5, 50, debug_sync); // Время baseline coalesced
    CUDA_CHECK(cudaGetLastError());                                        // Проверка на ошибки запуска kernel

    float t_g32   = run_and_time(launch_gather32,  d_in, d_out, n, 5, 50, debug_sync); // Время gather stride=32
    CUDA_CHECK(cudaGetLastError());                                        // Проверка ошибок

    float t_g128  = run_and_time(launch_gather128, d_in, d_out, n, 5, 50, debug_sync); // Время gather stride=128
    CUDA_CHECK(cudaGetLastError());                                        // Проверка ошибок

    float t_sh32  = run_and_time(launch_shared32,  d_in, d_out, n, 5, 50, debug_sync); // Время shared optimization stride=32
    CUDA_CHECK(cudaGetLastError());                                        // Проверка ошибок

    float t_gs    = run_and_time(launch_gridstride,d_in, d_out, n, 5, 50, debug_sync); // Время coalesced grid-stride
    CUDA_CHECK(cudaGetLastError());                                        // Проверка ошибок

    std::cout << std::fixed << std::setprecision(4);                       // Формат вывода: 4 знака после запятой
    std::cout << "\nAverage kernel time (ms per launch):\n";               // Заголовок вывода средних времён
    std::cout << "  coalesced (baseline)           : " << t_coal << " ms\n"; // Печать t_coal
    std::cout << "  gather non-coalesced (stride32): " << t_g32  << " ms\n"; // Печать t_g32
    std::cout << "  gather non-coalesced (stride128): " << t_g128 << " ms\n"; // Печать t_g128
    std::cout << "  shared tile+permute (stride32) : " << t_sh32 << " ms\n"; // Печать t_sh32
    std::cout << "  coalesced grid-stride          : " << t_gs   << " ms\n"; // Печать t_gs

    // Cleanup                                                              // Очистка памяти и ресурсов
    CUDA_CHECK(cudaFree(d_in));                                            // Освобождаем d_in
    CUDA_CHECK(cudaFree(d_out));                                           // Освобождаем d_out
    return 0;                                                              // Завершаем программу успешно
}                                                                          // Конец main
