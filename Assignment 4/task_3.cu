// task_3_hybrid_scan.cu                                                // Имя файла (Task 3)
 // Assignment 4 - Task 3 (25 pts.)                                     // Описание задания (баллы)
 // Hybrid program: first part of array processed on CPU, second part on GPU. // Гибрид: первая часть CPU, вторая GPU
 // Operation: Prefix Sum (Inclusive Scan).                             // Операция: префиксная сумма (inclusive scan)
 // Compares CPU-only, GPU-only, Hybrid times + correctness.            // Сравнивает времена и корректность (CPU/GPU/Hybrid)
 //
 // Build:                                                              // Как собрать
 //   nvcc -O2 task_3_hybrid_scan.cu -o task_3                           // Команда компиляции

#include <cstdio>                                                       // std::printf
#include <cstdlib>                                                      // std::rand, std::srand, std::exit
#include <cmath>                                                        // std::fabs
#include <vector>                                                       // std::vector
#include <thread>                                                       // std::thread
#include <chrono>                                                       // std::chrono таймер

#include <cuda_runtime.h>                                               // CUDA runtime API

#define N 1000000                                                       // Размер массива
#define BLOCK_SIZE 256                                                  // Потоков в блоке
#define SECTION_SIZE (BLOCK_SIZE * 2) // 512                             // Элементов на блок (2 на поток) = 512

// ------------------------- CUDA error check -------------------------  // Заголовок секции: проверка ошибок CUDA
static void checkCuda(cudaError_t err, const char* where) {             // Функция: проверка err + место где случилось
    if (err != cudaSuccess) {                                           // Если ошибка не успешная
        std::printf("CUDA error at %s: %s\n", where, cudaGetErrorString(err)); // Печать места и текста ошибки
        std::exit(1);                                                   // Завершаем программу с кодом 1
    }                                                                   // Конец if
}                                                                       // Конец checkCuda

// ------------------------- CPU inclusive scan ------------------------- // Заголовок: CPU inclusive scan
static void scan_cpu(const float* in, float* out, int n) {              // CPU scan: in->out, размер n
    if (n <= 0) return;                                                 // Если n<=0 — ничего не делаем
    out[0] = in[0];                                                     // Первый элемент inclusive scan
    for (int i = 1; i < n; i++) out[i] = out[i - 1] + in[i];            // Префиксная сумма: out[i]=out[i-1]+in[i]
}                                                                       // Конец scan_cpu

// ------------------------- Verification with locate ------------------------- // Заголовок: верификация с поиском индекса ошибки
static bool verify_and_locate(const float* cpu, const float* gpu, int n, float eps = 1e-3f) { // Сравнение cpu/gpu с порогом eps
    int first_bad = -1;                                                 // Индекс первого несовпадения (если есть)
    float max_diff = 0.0f;                                              // Максимальная разница
    int max_i = -1;                                                     // Индекс максимальной разницы

    for (int i = 0; i < n; i++) {                                       // Проходим по всем элементам
        float d = std::fabs(cpu[i] - gpu[i]);                           // Абсолютная разница
        if (d > max_diff) { max_diff = d; max_i = i; }                  // Обновляем максимум и индекс
        if (first_bad == -1 && d > eps) first_bad = i;                  // Запоминаем первый индекс, где diff>eps
    }                                                                   // Конец цикла

    if (first_bad != -1) {                                              // Если нашли несовпадение
        std::printf("FIRST MISMATCH at i=%d: CPU=%.2f GPU=%.2f diff=%.6f\n", // Печатаем первый mismatch
                    first_bad, cpu[first_bad], gpu[first_bad], std::fabs(cpu[first_bad] - gpu[first_bad])); // Данные mismatch
        std::printf("MAX DIFF at i=%d: diff=%.6f\n", max_i, max_diff);   // Печатаем где максимум diff
        return false;                                                   // Возвращаем false (ошибка)
    }                                                                   // Конец if
    return true;                                                        // Иначе всё ок
}                                                                       // Конец verify_and_locate

// ------------------------- GPU kernels -------------------------       // Заголовок: GPU kernels
// Correct Blelloch scan (work-efficient) in shared memory, for 512 elements per block. // Blelloch scan в shared, 512 элементов/блок
// IMPORTANT: indices inside a block are contiguous pairs: (0,1), (2,3), ..., (510,511). // Важно: каждая нить обрабатывает пару подряд
__global__ void scan_blocks_inclusive_blelloch(const float* in, float* out, float* blockSums, int n) { // Kernel: scan секции + сумма секции
    __shared__ float temp[SECTION_SIZE];                                // Shared массив на 512 float

    int tid  = threadIdx.x;                                             // Индекс потока 0..255
    int base = blockIdx.x * SECTION_SIZE;                                // База секции для блока

    // contiguous indices in the section:                               // Комментарий: индексы пары подряд
    int i1 = base + (2 * tid);                                          // Первый индекс пары
    int i2 = i1 + 1;                                                    // Второй индекс пары

    float a = (i1 < n) ? in[i1] : 0.0f;                                  // Берём a или 0 если вне массива
    float b = (i2 < n) ? in[i2] : 0.0f;                                  // Берём b или 0 если вне массива

    // store contiguously                                               // Запись пары в temp подряд
    temp[2 * tid]     = a;                                              // temp[0], temp[2], ... = a
    temp[2 * tid + 1] = b;                                              // temp[1], temp[3], ... = b

    // ---- upsweep ----                                                // Фаза upsweep (reduce)
    int offset = 1;                                                     // Начальный шаг offset
    for (int d = SECTION_SIZE >> 1; d > 0; d >>= 1) {                   // d=256,128,...,1
        __syncthreads();                                                // Синхронизация перед операцией
        if (tid < d) {                                                  // Работает только часть потоков
            int ai = offset * (2 * tid + 1) - 1;                        // Индекс ai
            int bi = offset * (2 * tid + 2) - 1;                        // Индекс bi
            temp[bi] += temp[ai];                                       // Суммируем вверх по дереву
        }                                                               // Конец if
        offset <<= 1;                                                   // Удваиваем offset
    }                                                                   // Конец upsweep цикла

    __syncthreads();                                                    // Синхронизация перед downsweep подготовкой
    float total = temp[SECTION_SIZE - 1];                               // Общая сумма секции (последний элемент)
    if (tid == 0) temp[SECTION_SIZE - 1] = 0.0f;                        // Ставим 0 для exclusive базы

    // ---- downsweep ----                                               // Фаза downsweep (расстановка префиксов)
    for (int d = 1; d < SECTION_SIZE; d <<= 1) {                        // d=1,2,4,...,256
        offset >>= 1;                                                   // Делим offset обратно
        __syncthreads();                                                // Синхронизация
        if (tid < d) {                                                  // Активная часть потоков
            int ai = offset * (2 * tid + 1) - 1;                        // Индекс ai
            int bi = offset * (2 * tid + 2) - 1;                        // Индекс bi
            float t = temp[ai];                                         // Сохраняем temp[ai]
            temp[ai] = temp[bi];                                        // Перестановка
            temp[bi] += t;                                              // Накопление
        }                                                               // Конец if
    }                                                                   // Конец downsweep цикла
    __syncthreads();                                                    // Финальная синхронизация

    // temp is exclusive -> make inclusive                               // temp сейчас exclusive, делаем inclusive
    if (i1 < n) out[i1] = temp[2 * tid] + a;                            // inclusive = exclusive + исходный a
    if (i2 < n) out[i2] = temp[2 * tid + 1] + b;                        // inclusive = exclusive + исходный b

    if (tid == 0) blockSums[blockIdx.x] = total;                        // Записываем сумму секции в blockSums
}                                                                       // Конец scan_blocks_inclusive_blelloch

// Add scanned block offsets to each block (pair mapping consistent with scan kernel) // Добавление offsets по секциям (пары как в scan)
__global__ void add_block_offsets_pairs(float* data, const float* scannedBlockSums, int n) { // Kernel: data += offset для блока
    int tid  = threadIdx.x;                                             // Индекс потока
    int base = blockIdx.x * SECTION_SIZE;                               // Начало секции

    if (blockIdx.x == 0) return;                                        // Первому блоку смещение не нужно

    float offset = scannedBlockSums[blockIdx.x - 1];                    // offset = сумма всех предыдущих блоков

    int i1 = base + (2 * tid);                                          // Первый индекс пары
    int i2 = i1 + 1;                                                    // Второй индекс пары

    if (i1 < n) data[i1] += offset;                                     // Прибавляем offset к i1
    if (i2 < n) data[i2] += offset;                                     // Прибавляем offset к i2
}                                                                       // Конец add_block_offsets_pairs

// Add constant offset to whole array (used for hybrid right half)       // Kernel: прибавить константу ко всем элементам (для правой половины)
__global__ void add_constant_offset(float* data, int n, float offset) {  // Kernel: data[idx]+=offset
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                    // Глобальный индекс
    if (idx < n) data[idx] += offset;                                   // Если в пределах массива — прибавляем offset
}                                                                       // Конец add_constant_offset

// ------------------------- GPU scan wrapper (inclusive) ------------------------- // Обёртка: полный GPU scan inclusive
static void gpu_scan_inclusive(const float* d_in, float* d_out, int n) { // Вход/выход device pointers, размер n
    int numBlocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;              // Количество блоков (секций), округление вверх

    float *d_blockSums = nullptr, *d_scannedSums = nullptr;             // Device массивы: суммы секций и их scan
    checkCuda(cudaMalloc(&d_blockSums,  numBlocks * sizeof(float)), "cudaMalloc d_blockSums"); // Выделяем d_blockSums
    checkCuda(cudaMalloc(&d_scannedSums, numBlocks * sizeof(float)), "cudaMalloc d_scannedSums"); // Выделяем d_scannedSums

    scan_blocks_inclusive_blelloch<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n); // Kernel 1: scan секций + суммы
    checkCuda(cudaGetLastError(), "scan_blocks_inclusive_blelloch launch"); // Проверка ошибки запуска

    // block sums -> CPU scan -> back (fast: ~1954 elements)             // Перенос block sums на CPU, scan на CPU, обратно на GPU
    std::vector<float> h_sums(numBlocks), h_scanned(numBlocks);         // Host массивы для сумм и их скана
    checkCuda(cudaMemcpy(h_sums.data(), d_blockSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost),
              "D2H block sums");                                        // Копируем суммы блоков D->H

    if (numBlocks > 0) {                                                // Если есть блоки
        scan_cpu(h_sums.data(), h_scanned.data(), numBlocks);           // CPU scan сумм блоков
        checkCuda(cudaMemcpy(d_scannedSums, h_scanned.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice),
                  "H2D scanned block sums");                            // Копируем scanned sums H->D
    }                                                                   // Конец if

    if (numBlocks > 1) {                                                // Если блоков больше 1, offsets нужны
        add_block_offsets_pairs<<<numBlocks, BLOCK_SIZE>>>(d_out, d_scannedSums, n); // Kernel 2: add offsets
        checkCuda(cudaGetLastError(), "add_block_offsets_pairs launch"); // Проверка ошибки запуска offsets
    }                                                                   // Конец if

    checkCuda(cudaDeviceSynchronize(), "sync after gpu_scan_inclusive"); // Дожидаемся завершения GPU работы

    cudaFree(d_blockSums);                                              // Освобождаем d_blockSums
    cudaFree(d_scannedSums);                                            // Освобождаем d_scannedSums
}                                                                       // Конец gpu_scan_inclusive

// ------------------------- Utility: print first/last K ------------------------- // Утилита: печать первых/последних K элементов
static void print_head_tail(const float* cpu, const float* other, int n, int k, const char* title) { // Сравнение в таблице
    std::printf("\n%s (first/last %d):\n", title, k);                    // Заголовок блока вывода
    std::printf("%-8s %-15s %-15s %-15s\n", "Index", "CPU", "Other", "Diff"); // Шапка таблицы
    std::printf("------------------------------------------------------------\n"); // Разделитель
    for (int i = 0; i < k; i++) {                                       // Первые k
        float d = std::fabs(cpu[i] - other[i]);                         // Разница
        std::printf("%-8d %-15.2f %-15.2f %-15.6f %s\n", i, cpu[i], other[i], d, (d < 1e-3f ? "OK" : "X")); // Печать строки
    }                                                                   // Конец цикла первых k
    std::printf("...\n");                                               // Многоточие
    for (int i = n - k; i < n; i++) {                                   // Последние k
        float d = std::fabs(cpu[i] - other[i]);                         // Разница
        std::printf("%-8d %-15.2f %-15.2f %-15.6f %s\n", i, cpu[i], other[i], d, (d < 1e-3f ? "OK" : "X")); // Печать строки
    }                                                                   // Конец цикла последних k
}                                                                       // Конец print_head_tail

int main() {                                                            // Точка входа
    const int n = N;                                                    // Размер массива
    const size_t bytes = (size_t)n * sizeof(float);                     // Размер в байтах

    // Host buffers                                                     // Раздел: host буферы
    std::vector<float> h_in(n), h_cpu(n), h_gpu(n), h_hybrid(n);        // Вход + результаты (CPU/GPU/Hybrid)

    std::srand(42);                                                     // Seed для воспроизводимости
    for (int i = 0; i < n; i++) h_in[i] = float((std::rand() % 10) + 1); // Заполнение значениями 1..10

    std::printf("=================================================================\n"); // Разделитель
    std::printf("  Task 3: Hybrid CPU + GPU processing (Prefix Sum / Scan)\n"); // Заголовок
    std::printf("  Array size: %d\n", n);                                // Размер массива
    std::printf("=================================================================\n\n"); // Разделитель + пустая строка

    // ---------------- CPU-only (wall-clock) ----------------           // CPU-only измерение wall-clock
    auto t0 = std::chrono::high_resolution_clock::now();                // Время старта CPU
    scan_cpu(h_in.data(), h_cpu.data(), n);                             // CPU scan всего массива
    auto t1 = std::chrono::high_resolution_clock::now();                // Время конца CPU
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count(); // Время CPU в мс
    std::printf("CPU-only time: %.4f ms\n", cpu_ms);                     // Печатаем CPU время

    // ---------------- GPU-only (wall-clock incl. memcpy) ---------------- // GPU-only: время включая memcpy
    float *d_in = nullptr, *d_out = nullptr;                            // Device pointers для полного массива
    checkCuda(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");             // Выделяем d_in
    checkCuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out");           // Выделяем d_out

    auto g0 = std::chrono::high_resolution_clock::now();                // Старт GPU таймера
    checkCuda(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice), "H2D full input"); // Копируем весь вход H->D
    gpu_scan_inclusive(d_in, d_out, n);                                 // Выполняем GPU scan
    checkCuda(cudaMemcpy(h_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost), "D2H full output"); // Копируем результат D->H
    auto g1 = std::chrono::high_resolution_clock::now();                // Конец GPU таймера
    double gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count(); // Время GPU-only в мс
    std::printf("GPU-only time: %.4f ms\n", gpu_ms);                     // Печатаем GPU время

    // ---------------- Hybrid (CPU first half + GPU second half) ---------------- // Гибрид: CPU левая половина, GPU правая половина
    int n1 = n / 2;                                                     // Размер левой половины
    int n2 = n - n1;                                                    // Размер правой половины

    float *d_in2 = nullptr, *d_out2 = nullptr;                          // Device pointers для правой половины
    checkCuda(cudaMalloc(&d_in2, (size_t)n2 * sizeof(float)), "cudaMalloc d_in2"); // Выделяем d_in2
    checkCuda(cudaMalloc(&d_out2, (size_t)n2 * sizeof(float)), "cudaMalloc d_out2"); // Выделяем d_out2

    float sum_left = 0.0f;                                              // Сумма левой половины (для offset правой)

    auto h0 = std::chrono::high_resolution_clock::now();                // Старт таймера hybrid

    // Copy second half to GPU                                           // Копируем правую половину на GPU
    checkCuda(cudaMemcpy(d_in2, h_in.data() + n1, (size_t)n2 * sizeof(float), cudaMemcpyHostToDevice), "H2D half2"); // H->D для half2

    // CPU thread does left scan                                         // Поток CPU делает scan левой половины
    std::thread cpu_thread([&]() {                                      // Запускаем отдельный поток CPU
        scan_cpu(h_in.data(), h_hybrid.data(), n1);                     // Сканируем левую половину в h_hybrid[0..n1-1]
        sum_left = h_hybrid[n1 - 1];                                    // Берём сумму левой половины = последний элемент левого scan
    });                                                                 // Конец лямбды (поток ещё работает)

    // GPU does right scan concurrently                                  // GPU параллельно делает scan правой половины
    gpu_scan_inclusive(d_in2, d_out2, n2);                               // GPU scan для правой половины (без offset пока)

    cpu_thread.join();                                                  // Ждём завершения CPU потока

    // Add offset=sum_left to right half on GPU                           // Прибавляем offset=sum_left к правой половине на GPU
    int threads = 256;                                                  // Количество потоков для add_constant_offset
    int blocks = (n2 + threads - 1) / threads;                          // Количество блоков для add_constant_offset
    add_constant_offset<<<blocks, threads>>>(d_out2, n2, sum_left);      // Kernel: прибавить sum_left ко всем элементам правой половины
    checkCuda(cudaGetLastError(), "add_constant_offset launch");         // Проверка ошибки запуска kernel
    checkCuda(cudaDeviceSynchronize(), "sync after add_constant_offset"); // Синхронизация после прибавления offset

    // Copy right half back                                              // Копируем правую половину обратно на host
    checkCuda(cudaMemcpy(h_hybrid.data() + n1, d_out2, (size_t)n2 * sizeof(float), cudaMemcpyDeviceToHost),
              "D2H half2");                                             // D->H для half2

    auto h1 = std::chrono::high_resolution_clock::now();                // Конец таймера hybrid
    double hybrid_ms = std::chrono::duration<double, std::milli>(h1 - h0).count(); // Время hybrid в мс
    std::printf("Hybrid time:   %.4f ms\n", hybrid_ms);                  // Печатаем hybrid время

    // ---------------- Correctness ----------------                       // Раздел: проверка корректности
    bool ok_gpu = verify_and_locate(h_cpu.data(), h_gpu.data(), n);      // Проверяем GPU-only против CPU
    bool ok_hy  = verify_and_locate(h_cpu.data(), h_hybrid.data(), n);   // Проверяем Hybrid против CPU

    std::printf("\nCorrectness:\n");                                     // Заголовок корректности
    std::printf("  GPU-only   : %s\n", ok_gpu ? "OK" : "ERROR");         // Статус GPU-only
    std::printf("  Hybrid     : %s\n", ok_hy  ? "OK" : "ERROR");         // Статус Hybrid

    std::printf("\nLast element:\n");                                    // Заголовок: последний элемент
    std::printf("  CPU:    %.2f\n", h_cpu[n - 1]);                       // Последний элемент CPU результата
    std::printf("  GPU:    %.2f\n", h_gpu[n - 1]);                       // Последний элемент GPU результата
    std::printf("  Hybrid: %.2f\n", h_hybrid[n - 1]);                    // Последний элемент Hybrid результата

    if (!ok_gpu) print_head_tail(h_cpu.data(), h_gpu.data(), n, 5, "GPU-only check"); // Если GPU ошибка — печатаем head/tail
    if (!ok_hy)  print_head_tail(h_cpu.data(), h_hybrid.data(), n, 5, "Hybrid check"); // Если Hybrid ошибка — печатаем head/tail

    // Speedups                                                          // Раздел: ускорения
    std::printf("\n=================================================================\n"); // Разделитель
    std::printf("  PERFORMANCE SUMMARY (wall-clock)\n");                 // Заголовок summary
    std::printf("=================================================================\n"); // Разделитель
    std::printf("  CPU-only  : %.4f ms\n", cpu_ms);                       // Время CPU-only
    std::printf("  GPU-only  : %.4f ms\n", gpu_ms);                       // Время GPU-only
    std::printf("  Hybrid    : %.4f ms\n", hybrid_ms);                    // Время Hybrid
    std::printf("-----------------------------------------------------------------\n"); // Разделитель
    std::printf("  Speedup GPU vs CPU   : %.2fx\n", cpu_ms / gpu_ms);     // Ускорение GPU относительно CPU
    std::printf("  Speedup Hybrid vs CPU: %.2fx\n", cpu_ms / hybrid_ms);  // Ускорение Hybrid относительно CPU
    std::printf("=================================================================\n"); // Конец summary

    // Cleanup                                                           // Освобождение ресурсов
    cudaFree(d_in);                                                     // Освобождаем d_in (full)
    cudaFree(d_out);                                                    // Освобождаем d_out (full)
    cudaFree(d_in2);                                                    // Освобождаем d_in2 (half2)
    cudaFree(d_out2);                                                   // Освобождаем d_out2 (half2)

    return 0;                                                           // Успешный выход
}                                                                       // Конец main
