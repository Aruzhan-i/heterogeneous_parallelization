// task_2.cu                                                  // Имя файла / задача 2 (CUDA)
#include <cstdio>                                             // printf
#include <cstdlib>                                            // rand, srand, exit
#include <cmath>                                              // fabs
#include <vector>                                             // std::vector
#include <ctime>                                              // clock, time
#include <cuda_runtime.h>                                     // CUDA runtime API

#define N 1000000                                             // Количество элементов входного массива
#define BLOCK_SIZE 256                                        // Количество потоков в блоке
#define SECTION_SIZE (BLOCK_SIZE * 2)                         // Количество элементов, обрабатываемых одним блоком (2 элемента на поток)

static void checkCuda(cudaError_t err, const char* where) {   // Функция проверки ошибок CUDA + вывод места ошибки
    if (err != cudaSuccess) {                                 // Если ошибка не равна cudaSuccess
        printf("CUDA error at %s: %s\n", where, cudaGetErrorString(err)); // Печатаем место и текст ошибки
        exit(1);                                              // Прерываем программу
    }                                                         // Конец if
}                                                             // Конец checkCuda

// ---------------- CPU scan (inclusive) ----------------      // CPU реализация inclusive prefix sum
static void scan_cpu(const float* in, float* out, int n) {     // Вход: in, выход: out, размер: n
    out[0] = in[0];                                           // Первый элемент совпадает (inclusive scan)
    for (int i = 1; i < n; i++) out[i] = out[i - 1] + in[i];  // Каждый следующий элемент = предыдущая сумма + текущий вход
}                                                             // Конец scan_cpu

// ---------------- GPU block scan (inclusive) ---------------- // GPU scan по блокам
// Делает inclusive scan в пределах одной секции SECTION_SIZE.  // Каждая секция SECTION_SIZE обрабатывается одним блоком
// Также пишет сумму секции в blockSums[blockIdx.x].            // Итог суммы секции записывается в blockSums
__global__ void scan_blocks_inclusive(const float* in, float* out, float* blockSums, int n) { // CUDA kernel: скан секций
    __shared__ float temp[SECTION_SIZE];                      // Shared memory буфер секции (SECTION_SIZE элементов)

    int tid = threadIdx.x;                                    // Локальный индекс потока внутри блока
    int base = blockIdx.x * SECTION_SIZE;                     // Начальный индекс секции для данного блока

    int i1 = base + tid;                                      // Первый индекс элемента для потока
    int i2 = base + tid + BLOCK_SIZE;                         // Второй индекс элемента для потока (вторая половина секции)

    temp[tid] = (i1 < n) ? in[i1] : 0.0f;                     // Грузим 1-й элемент в shared memory или 0 если вне границы
    temp[tid + BLOCK_SIZE] = (i2 < n) ? in[i2] : 0.0f;        // Грузим 2-й элемент в shared memory или 0 если вне границы
    __syncthreads();                                          // Синхронизация: все должны загрузить temp

    // Hillis–Steele inclusive scan (просто, корректно)         // Алгоритм Hillis–Steele inclusive scan
    // (для учебки норм; Blelloch быстрее, но сложнее)          // Примечание: Blelloch быстрее, но сложнее
    for (int stride = 1; stride < SECTION_SIZE; stride <<= 1) { // stride = 1,2,4,8... до SECTION_SIZE
        float val = 0.0f;                                     // Временная переменная для значения соседа
        if (tid >= stride) val = temp[tid - stride];          // Если индекс >= stride, берем элемент слева на stride
        __syncthreads();                                      // Синхронизация перед модификацией
        temp[tid] += val;                                     // Добавляем val к текущему элементу (первая половина)

        val = 0.0f;                                           // Снова обнуляем временную переменную
        int t2 = tid + BLOCK_SIZE;                            // Индекс потока во второй половине секции
        if (t2 >= stride) val = temp[t2 - stride];            // Аналогично: берем элемент слева для второй половины
        __syncthreads();                                      // Синхронизация
        temp[t2] += val;                                      // Добавляем val для второй половины

        __syncthreads();                                      // Синхронизация на конец итерации stride
    }                                                         // Конец цикла stride

    if (i1 < n) out[i1] = temp[tid];                          // Записываем 1-й элемент результата в global memory
    if (i2 < n) out[i2] = temp[tid + BLOCK_SIZE];             // Записываем 2-й элемент результата в global memory

    // сумма секции = последний элемент после inclusive scan     // Последний элемент секции после scan = сумма секции
    if (tid == 0) {                                           // Один поток (tid=0) сохраняет сумму секции
        blockSums[blockIdx.x] = temp[SECTION_SIZE - 1];        // Записываем сумму секции (последний элемент)
    }                                                         // Конец if
}                                                             // Конец kernel scan_blocks_inclusive

// Добавляем смещение (сумму всех предыдущих блоков) к каждому элементу блока // Kernel добавления offsets к каждому блоку
__global__ void add_block_offsets(float* data, const float* scannedBlockSums, int n) { // CUDA kernel: прибавление offset
    int tid = threadIdx.x;                                    // Локальный индекс потока в блоке
    int base = blockIdx.x * SECTION_SIZE;                     // Начало секции в массиве data

    if (blockIdx.x == 0) return;                              // Для первого блока смещение = 0, выходим

    float offset = scannedBlockSums[blockIdx.x - 1];          // Offset = сумма всех предыдущих секций (inclusive sums)

    int i1 = base + tid;                                      // Первый индекс элемента потока
    int i2 = base + tid + BLOCK_SIZE;                         // Второй индекс элемента потока

    if (i1 < n) data[i1] += offset;                           // Прибавляем offset к первому элементу секции
    if (i2 < n) data[i2] += offset;                           // Прибавляем offset ко второму элементу секции
}                                                             // Конец kernel add_block_offsets

// Проверка нескольких элементов                               // Проверяем только часть элементов CPU vs GPU
static bool verify(const float* cpu, const float* gpu, int n) { // Функция сравнения массивов результатов
    bool ok = true;                                           // Флаг корректности
    int k = 5;                                                // Количество проверяемых элементов в начале и конце

    printf("\nProverka rezultatov (pervye i poslednie %d elementov):\n", k); // Заголовок проверки
    printf("%-8s %-15s %-15s %-15s\n", "Index", "CPU", "GPU", "Diff");      // Шапка таблицы
    printf("------------------------------------------------------------\n"); // Разделитель

    for (int i = 0; i < k; i++) {                             // Проверяем первые k элементов
        float d = fabs(cpu[i] - gpu[i]);                      // Абсолютная разница CPU/GPU
        printf("%-8d %-15.2f %-15.2f %-15.6f %s\n", i, cpu[i], gpu[i], d, (d < 1e-3f ? "OK" : "X")); // Печатаем сравнение
        if (d >= 1e-3f) ok = false;                           // Если разница большая — ошибка
    }                                                         // Конец цикла по первым k

    printf("...\n");                                          // Печатаем многоточие для середины массива

    for (int i = n - k; i < n; i++) {                         // Проверяем последние k элементов
        float d = fabs(cpu[i] - gpu[i]);                      // Абсолютная разница CPU/GPU
        printf("%-8d %-15.2f %-15.2f %-15.6f %s\n", i, cpu[i], gpu[i], d, (d < 1e-3f ? "OK" : "X")); // Печать результата
        if (d >= 1e-3f) ok = false;                           // Если разница выше порога — ставим ошибку
    }                                                         // Конец цикла по последним k

    return ok;                                                // Возвращаем итоговый статус проверки
}                                                             // Конец verify

int main() {                                                  // Точка входа
    int n = N;                                                // Локальная переменная размера данных
    size_t bytes = n * sizeof(float);                         // Размер памяти под входной/выходной массив в байтах

    // Host                                                    // Раздел: память CPU
    std::vector<float> h_in(n), h_cpu(n), h_gpu(n);           // Вход, результат CPU, результат GPU

    srand(42);                                                // Фиксируем seed для воспроизводимости
    for (int i = 0; i < n; i++) h_in[i] = float((rand() % 10) + 1); // Заполняем массив значениями 1..10

    printf("=================================================================\n"); // Линия-разделитель
    printf("  CUDA Prefix Sum (Scan): %d elements\n", n);      // Заголовок программы
    printf("=================================================================\n\n"); // Вторая линия + пустая строка

    // ---------------- CPU timing ----------------              // Раздел: время CPU
    printf("Executing on CPU...\n");                          // Сообщение о запуске CPU
    clock_t c0 = clock();                                     // Начало измерения времени CPU
    scan_cpu(h_in.data(), h_cpu.data(), n);                   // Запускаем CPU inclusive scan
    clock_t c1 = clock();                                     // Конец измерения времени CPU
    double cpu_ms = (double)(c1 - c0) / CLOCKS_PER_SEC * 1000.0; // Перевод в миллисекунды

    printf("--- CPU (sequential) ---\n");                      // Заголовок секции CPU
    printf("  Time: %.4f ms\n", cpu_ms);                      // Время CPU выполнения
    printf("  Last element (total sum): %.2f\n\n", h_cpu[n - 1]); // Последний элемент = сумма всех чисел

    // ---------------- GPU ----------------                      // Раздел: GPU выполнение
    printf("Executing on GPU...\n");                          // Сообщение о запуске GPU

    int numBlocks = (n + SECTION_SIZE - 1) / SECTION_SIZE;    // Число блоков (секций), округляем вверх

    float *d_in = nullptr, *d_out = nullptr, *d_blockSums = nullptr, *d_scannedBlockSums = nullptr; // Device pointers

    checkCuda(cudaMalloc(&d_in, bytes), "cudaMalloc d_in");   // Выделяем память на GPU для входа
    checkCuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out"); // Выделяем память на GPU для результата scan
    checkCuda(cudaMalloc(&d_blockSums, numBlocks * sizeof(float)), "cudaMalloc d_blockSums"); // Память под суммы секций
    checkCuda(cudaMalloc(&d_scannedBlockSums, numBlocks * sizeof(float)), "cudaMalloc d_scannedBlockSums"); // Память под просканенные суммы секций

    checkCuda(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice), "memcpy H2D"); // Копируем вход H->D

    cudaEvent_t start, stop;                                  // CUDA события для измерения GPU времени
    cudaEventCreate(&start);                                  // Создаём событие start
    cudaEventCreate(&stop);                                   // Создаём событие stop

    cudaEventRecord(start);                                   // Старт измерения (в очередь GPU)

    // 1) scan блоков + суммы блоков                             // Этап 1: scan секций + сохранение сумм секций
    scan_blocks_inclusive<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_blockSums, n); // Запуск kernel scan
    checkCuda(cudaGetLastError(), "scan_blocks_inclusive launch"); // Проверка ошибок запуска kernel

    // 2) blockSums -> host, scan на CPU (очень быстро, их ~1954) // Этап 2: перенос blockSums на host и скан на CPU
    std::vector<float> h_blockSums(numBlocks), h_scanned(numBlocks); // Host массивы: суммы блоков и их scan
    checkCuda(cudaMemcpy(h_blockSums.data(), d_blockSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost),
              "memcpy blockSums D2H");                        // Копируем blockSums D->H

    if (numBlocks > 0) {                                      // Если блоков вообще больше 0
        scan_cpu(h_blockSums.data(), h_scanned.data(), numBlocks); // CPU scan для block sums
        checkCuda(cudaMemcpy(d_scannedBlockSums, h_scanned.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice),
                  "memcpy scannedBlockSums H2D");             // Копируем scan sums H->D
    }                                                         // Конец if

    // 3) добавить offsets                                       // Этап 3: прибавляем offsets ко всем секциям кроме первой
    if (numBlocks > 1) {                                      // Если больше одного блока, offsets нужны
        add_block_offsets<<<numBlocks, BLOCK_SIZE>>>(d_out, d_scannedBlockSums, n); // Запуск kernel offsets
        checkCuda(cudaGetLastError(), "add_block_offsets launch"); // Проверка ошибок запуска offsets kernel
    }                                                         // Конец if

    cudaEventRecord(stop);                                    // Фиксируем событие stop
    cudaEventSynchronize(stop);                               // Ждём пока GPU завершит всё до stop

    float gpu_ms = 0.0f;                                      // Переменная времени GPU
    cudaEventElapsedTime(&gpu_ms, start, stop);               // Считаем elapsed time между start и stop

    checkCuda(cudaMemcpy(h_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost), "memcpy result D2H"); // Копируем результат D->H

    printf("--- GPU (parallel) ---\n");                        // Заголовок секции GPU
    printf("  Time: %.4f ms\n", gpu_ms);                       // Время GPU выполнения
    printf("  Last element (total sum): %.2f\n", h_gpu[n - 1]); // Последний элемент GPU результата
    printf("  Configuration:\n");                              // Заголовок конфигурации
    printf("    - Blocks (sections): %d\n", numBlocks);        // Количество блоков/секций
    printf("    - Threads per block: %d\n", BLOCK_SIZE);       // Потоков на блок
    printf("    - Elements per block: %d\n", SECTION_SIZE);    // Сколько элементов на блок

    bool ok = verify(h_cpu.data(), h_gpu.data(), n);           // Проверяем корректность первых/последних элементов

    float diff = fabs(h_cpu[n - 1] - h_gpu[n - 1]);            // Разница итоговых сумм
    float rel = diff / h_cpu[n - 1] * 100.0f;                  // Относительная ошибка в %

    printf("\n=================================================================\n"); // Разделитель
    printf("  ANALYSIS\n");                                    // Заголовок анализа
    printf("=================================================================\n"); // Разделитель
    printf("  Difference in final sums: %.6f\n", diff);         // Печать абсолютной разницы
    printf("  Relative error: %.6f%%\n", rel);                  // Печать относительной ошибки
    printf("  Speedup: %.2fx\n", cpu_ms / gpu_ms);              // Ускорение CPU/GPU
    printf("-----------------------------------------------------------------\n"); // Разделитель
    printf("  Status: %s\n", (ok && rel < 1e-3f) ? "OK - Results are CORRECT" : "ERROR - Computation errors detected"); // Статус корректности
    printf("=================================================================\n\n"); // Конец отчёта

    // Cleanup                                                  // Освобождение ресурсов
    cudaFree(d_in);                                            // Освобождаем d_in на GPU
    cudaFree(d_out);                                           // Освобождаем d_out на GPU
    cudaFree(d_blockSums);                                     // Освобождаем d_blockSums
    cudaFree(d_scannedBlockSums);                              // Освобождаем d_scannedBlockSums
    cudaEventDestroy(start);                                   // Удаляем CUDA event start
    cudaEventDestroy(stop);                                    // Удаляем CUDA event stop

    return 0;                                                  // Успешное завершение
}                                                             // Конец main
