// task_extra_cpu_add_gpu_mul.cu                                                // Имя файла: гибридная обработка массива CPU+GPU
// Доп. задание 1:                                                             // Описание дополнительного задания №1
//  - 1-я половина массива: CPU (OpenMP) -> A[i] = A[i] + 5                     // CPU обрабатывает первую половину: прибавляет 5
//  - 2-я половина массива: GPU (CUDA)   -> A[i] = A[i] * 2                     // GPU обрабатывает вторую половину: умножает на 2
//  - Выполняется одновременно (async memcpy + stream + pinned memory)          // CPU и GPU работают параллельно, используя async копии + stream + pinned memory
//  - Замер общего времени (wall time)                                          // Измеряется общее время выполнения (стеночное время)

#include <cuda_runtime.h>                                                      // Подключение CUDA runtime API (cudaMalloc, cudaMemcpyAsync, streams и т.д.)
#include <omp.h>                                                               // Подключение OpenMP (параллельные циклы на CPU)

#include <iostream>                                                            // Для вывода в консоль через std::cout/std::cerr
#include <iomanip>                                                             // Для форматирования вывода (setw, setprecision, fixed)
#include <cstdlib>                                                             // Для std::exit
#include <chrono>                                                              // Для измерения времени выполнения


#define CUDA_CHECK(call) do {       /* Макрос: обёртка для проверки ошибок CUDA вызовов*/                           \
    cudaError_t err = (call);        /* Выполнить CUDA вызов и получить код ошибки*/                          \
    if (err != cudaSuccess) {           /*  Если вызов завершился с ошибкой*/                       \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  /*  Напечатать текст ошибки CUDA*/ \
                  << " at " << __FILE__ << ":" << __LINE__    /* Указать файл и строку, где произошла ошибка*/  \
                  << "\n";                                    /* Перевод строки */ \
        std::exit(1);                                         /* Завершить программу с кодом 1 */  \
    }                                                        /* Конец if*/  \
} while(0)

__global__ void mul2_kernel(const float* __restrict__ in,                      // CUDA kernel: входной массив (read-only), restrict для оптимизаций
                            float* __restrict__ out,                           // CUDA kernel: выходной массив (write), restrict для оптимизаций
                            int n)                                             // CUDA kernel: количество элементов для обработки
{                                                                              // Начало тела ядра
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                            // Глобальный индекс потока
    if (idx < n) out[idx] = in[idx] * 2.0f;                                     // Если индекс в пределах: записать out=2*in
}                                                                              // Конец ядра

int main() {                                                                   // Точка входа программы
    const int N = 1'000'000;                                                   // Общее количество элементов массива
    const int half = N / 2;                                                    // Размер первой половины (CPU) и второй половины (GPU)

    const size_t bytes      = static_cast<size_t>(N) * sizeof(float);          // Количество байт для полного массива
    const size_t bytes_half = static_cast<size_t>(half) * sizeof(float);       // Количество байт для половины массива

    // ---------- Host pinned memory (для реального async) ----------           // Заголовок секции: pinned host memory для реального async memcpy
    float* h_in  = nullptr;                                                    // Указатель на pinned host input массив
    float* h_out = nullptr;                                                    // Указатель на pinned host output массив
    CUDA_CHECK(cudaMallocHost(&h_in,  bytes));                                  // Выделить pinned host память под входной массив
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));                                  // Выделить pinned host память под выходной массив

    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);               // Инициализация входного массива значениями 0..N-1 (float)

    // ---------- Device memory только для 2-й половины ----------              // Секция: device memory выделяется только под вторую половину
    float *d_in = nullptr, *d_out = nullptr;                                   // Указатели на device input/output для GPU половины
    CUDA_CHECK(cudaMalloc(&d_in,  bytes_half));                                 // Выделить device память под вход GPU половины
    CUDA_CHECK(cudaMalloc(&d_out, bytes_half));                                 // Выделить device память под выход GPU половины

    cudaStream_t stream;                                                       // CUDA stream для асинхронных операций
    CUDA_CHECK(cudaStreamCreate(&stream));                                     // Создать CUDA stream

    const int block = 256;                                                     // Размер блока (threads per block)
    const int grid  = (half + block - 1) / block;                               // Размер grid (число блоков) для half элементов (ceil)

    // ---------- Warm-up (чтобы убрать инициализацию CUDA из замера) ---------- // Warm-up секция: исключить cold-start overhead из измерения
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in + half, bytes_half, cudaMemcpyHostToDevice, stream)); // Async копирование 2-й половины host->device
    mul2_kernel<<<grid, block, 0, stream>>>(d_in, d_out, half);                 // Запуск CUDA kernel на half элементов (вторая половина)
    CUDA_CHECK(cudaMemcpyAsync(h_out + half, d_out, bytes_half, cudaMemcpyDeviceToHost, stream)); // Async копирование результата device->host
    CUDA_CHECK(cudaStreamSynchronize(stream));                                 // Дождаться завершения всех операций в stream
    CUDA_CHECK(cudaGetLastError());                                            // Проверить наличие ошибок CUDA (например kernel launch)

    // =================== HYBRID RUN + TIMING ===================              // Секция: гибридный запуск + измерение времени
    auto t0 = std::chrono::high_resolution_clock::now();                       // Засечь стартовое время (wall time)

    // GPU: 2-я половина -> умножение на 2 (async)                              // Комментарий: GPU обрабатывает вторую половину асинхронно
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in + half, bytes_half, cudaMemcpyHostToDevice, stream)); // Копия host->device для 2-й половины
    mul2_kernel<<<grid, block, 0, stream>>>(d_in, d_out, half);                 // Запуск kernel умножения на 2
    CUDA_CHECK(cudaMemcpyAsync(h_out + half, d_out, bytes_half, cudaMemcpyDeviceToHost, stream)); // Копия результата обратно на host
    CUDA_CHECK(cudaGetLastError());                                            // Проверить ошибки запуска kernel

    // CPU: 1-я половина -> прибавить 5 (параллельно с GPU)                     // Комментарий: CPU обрабатывает первую половину параллельно с GPU
    #pragma omp parallel for                                                   // OpenMP: распараллеливание цикла по потокам CPU
    for (int i = 0; i < half; ++i) {                                           // Цикл по первой половине массива
        h_out[i] = h_in[i] + 5.0f;                                             // Выход = вход + 5
    }                                                                          // Конец CPU цикла

    // ждём GPU часть                                                           // Комментарий: теперь надо дождаться завершения GPU операций
    CUDA_CHECK(cudaStreamSynchronize(stream));                                 // Синхронизация: дождаться выполнения GPU memcpy+kernel+memcpy

    auto t1 = std::chrono::high_resolution_clock::now();                       // Засечь конечное время (wall time)
    std::chrono::duration<double, std::milli> ms = t1 - t0;                    // Посчитать длительность в миллисекундах

    // =================== VERIFY ===================                            // Секция: проверка корректности результата
    bool ok = true;                                                            // Флаг корректности вычислений

    // CPU half check: out[i] == in[i] + 5                                      // Комментарий: проверяем CPU половину
    for (int i = 0; i < 5; ++i) {                                              // Проверяем первые 5 элементов первой половины
        if (h_out[i] != h_in[i] + 5.0f) ok = false;                            // Если есть несовпадение — ошибка
    }                                                                          // Конец проверки CPU части

    // GPU half check: out[i] == in[i] * 2                                      // Комментарий: проверяем GPU половину
    for (int i = half; i < half + 5; ++i) {                                    // Проверяем первые 5 элементов второй половины
        if (h_out[i] != h_in[i] * 2.0f) ok = false;                            // Если несовпадение — ошибка
    }                                                                          // Конец проверки GPU части

    if (h_out[N - 1] != h_in[N - 1] * 2.0f) ok = false;                        // Доп. проверка последнего элемента (GPU часть)

    int used_threads = 0;                                                      // Переменная: сколько OpenMP потоков реально использовано
    #pragma omp parallel                                                       // Создать параллельную область OpenMP
    {                                                                          // Начало параллельной области
        #pragma omp single                                                     // Выполнить блок только одним потоком
        used_threads = omp_get_num_threads();                                  // Получить общее число потоков в данной parallel области
    }                                                                          // Конец parallel области

    std::cout << "=== EXTRA: CPU(+5) on first half, GPU(*2) on second half ===\n"; // Вывести заголовок эксперимента
    std::cout << "N: " << N << " (CPU half: " << half << ", GPU half: " << (N - half) << ")\n"; // Вывести N и разбиение half/half
    std::cout << "CPU threads used: " << used_threads << "\n";                 // Показать число CPU потоков OpenMP
    std::cout << "GPU grid: " << grid << ", block: " << block << "\n";         // Показать конфигурацию CUDA kernel (grid/block)
    std::cout << "Total hybrid wall time: " << std::fixed << std::setprecision(4) // Вывести время с форматированием fixed и 4 знака
              << ms.count() << " ms\n";                                        // Напечатать измеренное wall time в миллисекундах
    std::cout << "Check: out[0]=" << std::fixed << std::setprecision(4) << h_out[0] // Напечатать out[0] с 4 знаками
              << ", out[half]=" << h_out[half]                                 // Напечатать элемент на границе половин
              << ", out[N-1]=" << h_out[N - 1]                                 // Напечатать последний элемент
              << " -> " << (ok ? "OK" : "FAIL") << "\n";                       // Напечатать итог проверки OK/FAIL

    // =================== CLEANUP ===================                           // Секция: освобождение ресурсов
    CUDA_CHECK(cudaStreamDestroy(stream));                                     // Удалить CUDA stream
    CUDA_CHECK(cudaFree(d_in));                                                // Освободить device input buffer
    CUDA_CHECK(cudaFree(d_out));                                               // Освободить device output buffer
    CUDA_CHECK(cudaFreeHost(h_in));                                            // Освободить pinned host input buffer
    CUDA_CHECK(cudaFreeHost(h_out));                                           // Освободить pinned host output buffer

    return 0;                                                                  // Возврат 0: успешное завершение программы
}                                                                              // Конец main
