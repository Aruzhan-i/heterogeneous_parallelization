// task_3_perf_benchmark.cu                                                // Имя файла: бенчмарк
// Бенчмарк: GPU vs CPU для reduction + prefix scan, разные N,            // Описание: сравнение CPU/GPU
// сравнение времени, и оптимизация через pinned memory (page-locked).     // Описание: плюс pinned memory

#include <cuda_runtime.h>                                                 // CUDA runtime API
#include <iostream>                                                       // std::cout / std::cerr
#include <vector>                                                         // std::vector
#include <numeric>                                                        // (обычно для accumulate, здесь почти не используется)
#include <iomanip>                                                        // std::fixed, std::setprecision
#include <chrono>                                                         // Таймер CPU (std::chrono)
#include <cmath>                                                          // std::abs, математика
#include <cstdlib>                                                        // std::exit

#define CUDA_CHECK(call)                                                  /* Макрос для проверки CUDA ошибок */ \
    do {                                                                  /* Начало блока */ \
        cudaError_t err = (call);                                         /* Выполнить call и сохранить код ошибки */ \
        if (err != cudaSuccess) {                                         /* Если ошибка */ \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)        /* Вывести текст ошибки */ \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";  /* Вывести файл и строку */ \
            std::exit(1);                                                 /* Аварийный выход */ \
        }                                                                 /* Конец if */ \
    } while (0)                                                           /* Конец макроса */

// -------------------------- REDUCTION (shared) -------------------------- // Раздел: редукция
__global__ void reduce_sum_shared(const float* __restrict__ in,            // Ядро: входной массив
                                  float* __restrict__ out,                // Ядро: выход (сумма блока)
                                  int n)                                  // Ядро: размер массива
{                                                                          // Начало ядра
    extern __shared__ float sdata[];                                       // Shared memory для редукции
    unsigned int tid = threadIdx.x;                                        // Индекс потока в блоке
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;        // Глобальный индекс (2 элемента на поток)

    float sum = 0.0f;                                                     // Локальная сумма
    if (i < (unsigned)n) sum += in[i];                                     // Первый элемент, если в пределах
    if (i + blockDim.x < (unsigned)n) sum += in[i + blockDim.x];           // Второй элемент, если в пределах

    sdata[tid] = sum;                                                     // Записать в shared
    __syncthreads();                                                      // Синхронизация

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {                // Редукция в shared (дерево)
        if (tid < s) sdata[tid] += sdata[tid + s];                         // Суммирование пар
        __syncthreads();                                                  // Синхронизация на каждом шаге
    }                                                                      // Конец цикла редукции
    if (tid == 0) out[blockIdx.x] = sdata[0];                              // Поток 0 пишет сумму блока
}                                                                          // Конец ядра reduce_sum_shared

float gpu_reduce_sum_device(const float* d_in, int n, int blockSize)       // Хост-функция: редукция, вход уже на GPU
{                                                                          // Начало функции
    float* curIn = const_cast<float*>(d_in);                               // Текущий вход (нельзя free d_in)
    float* d_tmp = nullptr;                                                // Временный буфер на GPU
    int curN = n;                                                         // Текущий размер массива

    // Важно: тут мы не хотим освобождать исходный d_in, поэтому            // Комментарий: про освобождение
    // освобождаем только промежуточные буферы.                             // Комментарий: только временные
    bool firstPass = true;                                                 // Флаг: первый проход (curIn == d_in)

    while (curN > 1) {                                                     // Пока не сведём к одному числу
        int gridSize = (curN + (blockSize * 2 - 1)) / (blockSize * 2);     // Сколько блоков нужно
        CUDA_CHECK(cudaMalloc(&d_tmp, gridSize * sizeof(float)));          // Выделить буфер под частичные суммы

        size_t shmBytes = blockSize * sizeof(float);                       // Shared bytes для блока
        reduce_sum_shared<<<gridSize, blockSize, shmBytes>>>(curIn, d_tmp, curN); // Запуск ядра редукции
        CUDA_CHECK(cudaGetLastError());                                    // Проверка ошибок запуска

        if (!firstPass) CUDA_CHECK(cudaFree(curIn));                       // Освободить предыдущий временный вход
        firstPass = false;                                                 // После первого прохода curIn уже временный

        curIn = d_tmp;                                                     // Новый вход = результат
        d_tmp = nullptr;                                                   // Обнуляем указатель (защита от повторного free)
        curN = gridSize;                                                   // Новый размер = число блоков
    }                                                                      // Конец while

    float h_sum = 0.0f;                                                    // Результат на CPU
    CUDA_CHECK(cudaMemcpy(&h_sum, curIn, sizeof(float), cudaMemcpyDeviceToHost)); // Копируем итог GPU->CPU
    if (!firstPass) CUDA_CHECK(cudaFree(curIn));                           // Освободить последний временный буфер (если был)
    return h_sum;                                                          // Вернуть сумму
}                                                                          // Конец gpu_reduce_sum_device

// -------------------------- SCAN (block shared + offsets) -------------------------- // Раздел: scan
// Scan внутри блока (Blelloch) -> inclusive, плюс сохраняем сумму блока.   // Пояснение: ядро scan по блокам
__global__ void scan_block_blelloch_inclusive(const float* __restrict__ in, // Входной массив
                                              float* __restrict__ out,     // Выходной массив (scan)
                                              float* __restrict__ block_sums, // Суммы блоков
                                              int n)                       // Размер массива
{                                                                          // Начало ядра scan
    extern __shared__ float s[];                                           // Shared memory (2*blockDim элементов)
    int tid = threadIdx.x;                                                 // Индекс потока
    int blockStart = 2 * blockDim.x * blockIdx.x;                          // Старт индекса блока (2*blockDim на блок)

    int i1 = blockStart + tid;                                             // Первый индекс для потока
    int i2 = blockStart + tid + blockDim.x;                                // Второй индекс для потока

    float x1 = (i1 < n) ? in[i1] : 0.0f;                                   // Загружаем x1 или 0
    float x2 = (i2 < n) ? in[i2] : 0.0f;                                   // Загружаем x2 или 0

    s[tid] = x1;                                                           // Пишем x1 в shared
    s[tid + blockDim.x] = x2;                                              // Пишем x2 во вторую половину shared

    for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {          // Up-sweep: stride 1,2,4,...
        __syncthreads();                                                   // Синхронизация
        int idx = (tid + 1) * stride * 2 - 1;                               // Индекс узла
        if (idx < 2 * blockDim.x) s[idx] += s[idx - stride];               // Суммирование в дереве
    }                                                                      // Конец up-sweep

    if (tid == 0) {                                                        // Поток 0 блока
        block_sums[blockIdx.x] = s[2 * blockDim.x - 1];                     // Сохраняем сумму блока
        s[2 * blockDim.x - 1] = 0.0f;                                      // Обнуляем последний элемент (для exclusive)
    }                                                                      // Конец if tid==0

    for (int stride = blockDim.x; stride > 0; stride >>= 1) {              // Down-sweep: stride вниз
        __syncthreads();                                                   // Синхронизация
        int idx = (tid + 1) * stride * 2 - 1;                               // Индекс узла
        if (idx < 2 * blockDim.x) {                                        // Проверка границы
            float t = s[idx - stride];                                     // Временная переменная
            s[idx - stride] = s[idx];                                      // Перестановка
            s[idx] += t;                                                   // Накопление
        }                                                                  // Конец if
    }                                                                      // Конец down-sweep
    __syncthreads();                                                       // Финальная синхронизация

    if (i1 < n) out[i1] = s[tid] + x1;                                     // Inclusive: exclusive + x1
    if (i2 < n) out[i2] = s[tid + blockDim.x] + x2;                         // Inclusive: exclusive + x2
}                                                                          // Конец scan_block_blelloch_inclusive

__global__ void add_block_offsets(float* data,                              // Ядро: добавление offset блоков
                                  const float* __restrict__ scanned_block_sums, // Просканированные суммы блоков
                                  int n)                                   // Размер массива
{                                                                          // Начало ядра add_block_offsets
    int blockStart = 2 * blockDim.x * blockIdx.x;                          // Старт индекса блока
    int tid = threadIdx.x;                                                 // Индекс потока

    int i1 = blockStart + tid;                                             // Первый индекс потока
    int i2 = blockStart + tid + blockDim.x;                                // Второй индекс потока

    float offset = 0.0f;                                                   // Offset по умолчанию
    if (blockIdx.x > 0) offset = scanned_block_sums[blockIdx.x - 1];        // Offset = сумма всех предыдущих блоков

    if (i1 < n) data[i1] += offset;                                        // Добавить offset к i1
    if (i2 < n) data[i2] += offset;                                        // Добавить offset к i2
}                                                                          // Конец add_block_offsets

static std::vector<float> cpu_inclusive_scan(const std::vector<float>& a)  // CPU inclusive scan (эталон/вспомогательное)
{                                                                          // Начало функции
    std::vector<float> out(a.size());                                      // Выходной массив
    float run = 0.0f;                                                      // Бегущая сумма
    for (size_t i = 0; i < a.size(); ++i) { run += a[i]; out[i] = run; }   // Один цикл: накапливаем и пишем
    return out;                                                            // Возвращаем результат
}                                                                          // Конец cpu_inclusive_scan

static void gpu_inclusive_scan_device(const float* d_in, float* d_out, int n, int blockSize) // GPU scan, данные на устройстве
{                                                                          // Начало функции
    int elemsPerBlock = 2 * blockSize;                                     // Элементов на блок
    int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;               // Количество блоков

    float* d_block_sums = nullptr;                                         // Буфер сумм блоков на GPU
    CUDA_CHECK(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));      // Выделить d_block_sums

    size_t shmBytes = elemsPerBlock * sizeof(float);                       // Shared bytes для scan ядра
    scan_block_blelloch_inclusive<<<numBlocks, blockSize, shmBytes>>>(d_in, d_out, d_block_sums, n); // Запуск scan ядра
    CUDA_CHECK(cudaGetLastError());                                        // Проверка ошибок запуска

    if (numBlocks > 1) {                                                   // Если блоков больше 1 — нужны offsets
        // Для простоты: scan сумм блоков на CPU (в бенчмарке это учитываем как часть GPU pipeline) // Комментарий
        std::vector<float> h_block_sums(numBlocks);                         // CPU буфер сумм блоков
        CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost)); // Копия GPU->CPU

        auto h_scanned = cpu_inclusive_scan(h_block_sums);                  // CPU scan сумм блоков

        float* d_scanned = nullptr;                                        // GPU буфер просканированных сумм блоков
        CUDA_CHECK(cudaMalloc(&d_scanned, numBlocks * sizeof(float)));     // Выделить d_scanned
        CUDA_CHECK(cudaMemcpy(d_scanned, h_scanned.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice)); // Копия CPU->GPU

        add_block_offsets<<<numBlocks, blockSize>>>(d_out, d_scanned, n);  // Прибавить offsets к блокам
        CUDA_CHECK(cudaGetLastError());                                    // Проверка ошибок запуска

        CUDA_CHECK(cudaFree(d_scanned));                                   // Освободить d_scanned
    }                                                                      // Конец if numBlocks > 1

    CUDA_CHECK(cudaFree(d_block_sums));                                    // Освободить d_block_sums
}                                                                          // Конец gpu_inclusive_scan_device

// -------------------------- Timing helpers --------------------------      // Раздел: измерение времени
static float elapsed_ms(cudaEvent_t a, cudaEvent_t b)                      // Время между CUDA events (мс)
{                                                                          // Начало функции
    float ms = 0.0f;                                                       // Переменная для миллисекунд
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));                            // Вычислить elapsed time
    return ms;                                                             // Вернуть миллисекунды
}                                                                          // Конец elapsed_ms

static double ms_cpu(std::chrono::high_resolution_clock::time_point a,      // Время CPU между time_point
                     std::chrono::high_resolution_clock::time_point b)     // Второй time_point
{                                                                          // Начало функции
    return std::chrono::duration<double, std::milli>(b - a).count();       // Возвращаем разницу в миллисекундах
}                                                                          // Конец ms_cpu

// -------------------------- Main benchmark --------------------------      // Раздел: main benchmark
int main()                                                                 // Точка входа
{                                                                          // Начало main
    std::cout << std::fixed << std::setprecision(3);                       // Формат вывода: 3 знака после запятой

    // размеры для теста                                                    // Набор размеров N
    std::vector<int> sizes = {                                             // Вектор размеров
        1 << 10,   // 1024                                                 // N = 1024
        1 << 14,   // 16384                                                // N = 16384
        1 << 18,   // 262144                                               // N = 262144
        1 << 20,   // 1,048,576                                            // N = 1,048,576
        1 << 22    // 4,194,304                                            // N = 4,194,304
    };                                                                     // Конец инициализации sizes

    const int blockSize = 256;                                             // Размер блока CUDA
    const int warmup = 2;                                                  // Кол-во прогонов разогрева
    const int iters = 5;                                                   // Кол-во измеряемых итераций

    // CUDA events                                                         // CUDA события для тайминга
    cudaEvent_t e0, e1, e2, e3;                                            // Объявляем события
    CUDA_CHECK(cudaEventCreate(&e0));                                      // Создаём e0
    CUDA_CHECK(cudaEventCreate(&e1));                                      // Создаём e1
    CUDA_CHECK(cudaEventCreate(&e2));                                      // Создаём e2
    CUDA_CHECK(cudaEventCreate(&e3));                                      // Создаём e3

    std::cout << "Columns:\n";                                             // Заголовок колонок
    std::cout << "N | CPU_reduce(ms) CPU_scan(ms) | GPU_reduce(ms) GPU_scan(ms) | " // Первая часть описания колонок
                 "H2D(ms) D2H(ms) (normal) | H2D(ms) D2H(ms) (pinned)\n\n"; // Вторая часть описания + пустая строка

    for (int N : sizes) {                                                  // Проходим по каждому N
        // ---------------- Host data (normal) ----------------              // Раздел: host данные обычные
        std::vector<float> h(N);                                           // Host вектор размера N
        for (int i = 0; i < N; ++i) h[i] = 1.0f;                           // Заполняем единицами

        // CPU reference reduction                                           // CPU редукция (эталон)
        auto t0 = std::chrono::high_resolution_clock::now();               // Старт таймера CPU
        double cpu_red = 0.0;                                              // CPU сумма
        for (float x : h) cpu_red += x;                                    // Суммируем на CPU
        auto t1 = std::chrono::high_resolution_clock::now();               // Стоп таймера CPU
        double cpu_red_ms = ms_cpu(t0, t1);                                // CPU время редукции в мс

        // CPU reference scan                                                // CPU scan (эталон)
        t0 = std::chrono::high_resolution_clock::now();                    // Старт таймера CPU
        auto cpu_scan = cpu_inclusive_scan(h);                             // CPU inclusive scan
        t1 = std::chrono::high_resolution_clock::now();                    // Стоп таймера CPU
        double cpu_scan_ms = ms_cpu(t0, t1);                               // CPU время scan в мс

        // ---------------- Device buffers ----------------                   // Раздел: буферы на GPU
        float *d_in = nullptr, *d_out = nullptr;                           // Указатели на device память
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));                  // Выделить d_in
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));                 // Выделить d_out

        // ---------------- Measure copies (normal) ----------------         // Раздел: замер копий (обычная память)
        CUDA_CHECK(cudaEventRecord(e0));                                   // Ставим событие начала H2D
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice)); // Копия Host->Device
        CUDA_CHECK(cudaEventRecord(e1));                                   // Ставим событие конца H2D
        CUDA_CHECK(cudaEventSynchronize(e1));                              // Ждём завершения e1
        float h2d_normal = elapsed_ms(e0, e1);                             // Время H2D (normal)

        // ---------------- GPU reduction timing ----------------            // Раздел: замер GPU reduction
        // warmup                                                           // Разогрев
        for (int k = 0; k < warmup; ++k) {                                 // warmup loop
            (void)gpu_reduce_sum_device(d_in, N, blockSize);               // Прогон редукции (результат не нужен)
        }                                                                  // Конец warmup
        CUDA_CHECK(cudaDeviceSynchronize());                               // Ждём завершения всех работ GPU

        float gpu_red_ms = 0.0f;                                           // Накопитель времени GPU reduction
        for (int k = 0; k < iters; ++k) {                                  // Цикл измерений
            CUDA_CHECK(cudaEventRecord(e0));                               // Старт измерения
            float s = gpu_reduce_sum_device(d_in, N, blockSize);           // Выполнить редукцию
            (void)s;                                                       // Подавить предупреждение unused
            CUDA_CHECK(cudaEventRecord(e1));                               // Конец измерения
            CUDA_CHECK(cudaEventSynchronize(e1));                          // Ждём e1
            gpu_red_ms += elapsed_ms(e0, e1);                              // Добавляем время итерации
        }                                                                  // Конец измерений
        gpu_red_ms /= iters;                                               // Среднее время GPU reduction

        // ---------------- GPU scan timing ----------------                  // Раздел: замер GPU scan
        // warmup                                                           // Разогрев
        for (int k = 0; k < warmup; ++k) {                                 // warmup loop
            gpu_inclusive_scan_device(d_in, d_out, N, blockSize);          // Прогон scan
        }                                                                  // Конец warmup
        CUDA_CHECK(cudaDeviceSynchronize());                               // Синхронизация GPU

        float gpu_scan_ms = 0.0f;                                          // Накопитель времени GPU scan
        for (int k = 0; k < iters; ++k) {                                  // Цикл измерений scan
            CUDA_CHECK(cudaEventRecord(e0));                               // Старт измерения
            gpu_inclusive_scan_device(d_in, d_out, N, blockSize);          // Выполнить scan
            CUDA_CHECK(cudaEventRecord(e1));                               // Конец измерения
            CUDA_CHECK(cudaEventSynchronize(e1));                          // Ждём e1
            gpu_scan_ms += elapsed_ms(e0, e1);                             // Добавляем время итерации
        }                                                                  // Конец измерений
        gpu_scan_ms /= iters;                                              // Среднее время GPU scan

        // ---------------- D2H (normal) ----------------                     // Раздел: Device->Host копирование (normal)
        std::vector<float> h_scan_gpu(N);                                  // Host буфер для результата scan
        CUDA_CHECK(cudaEventRecord(e2));                                   // Старт D2H
        CUDA_CHECK(cudaMemcpy(h_scan_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost)); // Копия Device->Host
        CUDA_CHECK(cudaEventRecord(e3));                                   // Конец D2H
        CUDA_CHECK(cudaEventSynchronize(e3));                              // Ждём e3
        float d2h_normal = elapsed_ms(e2, e3);                             // Время D2H (normal)

        // quick correctness (only last value)                               // Быстрая проверка корректности (последний элемент)
        double diff_last = std::abs((double)cpu_scan.back() - (double)h_scan_gpu.back()); // Разница last CPU vs GPU

        // ---------------- Pinned memory copy timing ----------------       // Раздел: pinned memory копии
        float* h_pinned = nullptr;                                         // Указатель на pinned host память
        CUDA_CHECK(cudaMallocHost(&h_pinned, N * sizeof(float)));          // Выделить pinned host память
        for (int i = 0; i < N; ++i) h_pinned[i] = h[i];                    // Скопировать данные в pinned буфер

        CUDA_CHECK(cudaEventRecord(e0));                                   // Старт H2D pinned
        CUDA_CHECK(cudaMemcpy(d_in, h_pinned, N * sizeof(float), cudaMemcpyHostToDevice)); // Копия pinned Host->Device
        CUDA_CHECK(cudaEventRecord(e1));                                   // Конец H2D pinned
        CUDA_CHECK(cudaEventSynchronize(e1));                              // Ждём e1
        float h2d_pinned = elapsed_ms(e0, e1);                             // Время H2D pinned

        CUDA_CHECK(cudaEventRecord(e2));                                   // Старт D2H pinned
        CUDA_CHECK(cudaMemcpy(h_pinned, d_out, N * sizeof(float), cudaMemcpyDeviceToHost)); // Копия Device->pinned Host
        CUDA_CHECK(cudaEventRecord(e3));                                   // Конец D2H pinned
        CUDA_CHECK(cudaEventSynchronize(e3));                              // Ждём e3
        float d2h_pinned = elapsed_ms(e2, e3);                             // Время D2H pinned

        CUDA_CHECK(cudaFreeHost(h_pinned));                                // Освободить pinned host память

        // ---------------- Print row ----------------                        // Раздел: печать строки результата
        std::cout << N << " | "                                             // Печать N и разделителя
                  << cpu_red_ms << " " << cpu_scan_ms << " | "             // Печать CPU времен
                  << gpu_red_ms << " " << gpu_scan_ms << " | "             // Печать GPU времен
                  << h2d_normal << " " << d2h_normal << " | "              // Печать H2D/D2H normal
                  << h2d_pinned << " " << d2h_pinned                       // Печать H2D/D2H pinned
                  << " | diff_last=" << diff_last                          // Печать ошибки по последнему элементу
                  << "\n";                                                 // Конец строки

        CUDA_CHECK(cudaFree(d_out));                                       // Освободить d_out
        CUDA_CHECK(cudaFree(d_in));                                        // Освободить d_in
    }                                                                      // Конец цикла по N

    CUDA_CHECK(cudaEventDestroy(e0));                                      // Уничтожить e0
    CUDA_CHECK(cudaEventDestroy(e1));                                      // Уничтожить e1
    CUDA_CHECK(cudaEventDestroy(e2));                                      // Уничтожить e2
    CUDA_CHECK(cudaEventDestroy(e3));                                      // Уничтожить e3

    return 0;                                                              // Выход из программы
}                                                                          // Конец main
