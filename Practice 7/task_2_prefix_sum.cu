// task_2_prefix_sum.cu                                                     // Имя файла
// CUDA prefix sum (inclusive scan) с shared memory + много-блочная версия + проверка CPU // Описание программы

#include <cuda_runtime.h>                                                  // CUDA runtime API
#include <iostream>                                                        // std::cout / std::cerr
#include <vector>                                                          // std::vector
#include <numeric>                                                         // (может быть полезно, но здесь почти не используется)
#include <iomanip>                                                         // std::fixed, std::setprecision
#include <cstdlib>                                                         // std::exit
#include <cmath>                                                           // std::abs для float/double

#define CUDA_CHECK(call)                                                   /* Макрос: проверка CUDA вызовов */ \
    do {                                                                   /* Начало блока */ \
        cudaError_t err = (call);                                          /* Выполняем call, сохраняем код ошибки */ \
        if (err != cudaSuccess) {                                          /* Если возникла ошибка */ \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         /* Вывод текста ошибки */ \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";   /* Вывод файла и строки */ \
            std::exit(1);                                                  /* Выход из программы */ \
        }                                                                  /* Конец if */ \
    } while (0)                                                            /* Конец макроса */

// 1) Scan внутри каждого блока (Blelloch) для 2*BLOCK элементов.            // Пояснение: 1-й шаг
//    Выход: out — просканированный массив                                   // Пояснение: что выдаёт ядро
//    block_sums[blockIdx.x] = сумма элементов блока (нужно для склейки блоков) // Пояснение: частичные суммы блоков
__global__ void scan_block_blelloch_inclusive(const float* __restrict__ in, // Входной массив на GPU
                                              float* __restrict__ out,     // Выходной массив (scan результат) на GPU
                                              float* __restrict__ block_sums, // Массив сумм блоков
                                              int n)                       // Размер входного массива
{                                                                           // Начало ядра
    extern __shared__ float s[];                                            // Shared memory: размер = 2*blockDim.x floats

    int tid = threadIdx.x;                                                 // Номер потока в блоке
    int blockStart = 2 * blockDim.x * blockIdx.x;                           // Стартовый индекс для блока (каждый блок берёт 2*blockDim элементов)

    int i1 = blockStart + tid;                                             // Индекс 1-го элемента для потока
    int i2 = blockStart + tid + blockDim.x;                                 // Индекс 2-го элемента для потока (вторая половина)

    // Загружаем 2 элемента на поток в shared (если за границей — 0)         // Комментарий к этапу загрузки
    float x1 = (i1 < n) ? in[i1] : 0.0f;                                    // Берём элемент 1 или 0 если вне границ
    float x2 = (i2 < n) ? in[i2] : 0.0f;                                    // Берём элемент 2 или 0 если вне границ

    s[tid] = x1;                                                           // Записываем x1 в shared
    s[tid + blockDim.x] = x2;                                               // Записываем x2 во вторую половину shared

    // --- Up-sweep (reduce) ---                                             // Фаза up-sweep (reduce)
    // stride: 1,2,4,... пока не дойдём до конца                              // Пояснение к stride
    for (int stride = 1; stride < 2 * blockDim.x; stride <<= 1) {           // Увеличиваем stride в 2 раза
        __syncthreads();                                                    // Синхронизация перед шагом
        int idx = (tid + 1) * stride * 2 - 1;                                // Индекс узла дерева редукции
        if (idx < 2 * blockDim.x) {                                         // Проверка на выход за границы shared
            s[idx] += s[idx - stride];                                      // Суммируем с "левым" элементом
        }                                                                   // Конец if
    }                                                                       // Конец for up-sweep

    // Сумма блока лежит в последнем элементе                                 // Комментарий: итог reduce
    if (tid == 0) {                                                         // Только поток 0
        block_sums[blockIdx.x] = s[2 * blockDim.x - 1];                      // Записываем сумму блока
        // Для exclusive scan по Blelloch нужно обнулить последний            // Пояснение: для down-sweep
        s[2 * blockDim.x - 1] = 0.0f;                                       // Обнуляем последний элемент
    }                                                                       // Конец if tid==0

    // --- Down-sweep ---                                                     // Фаза down-sweep
    for (int stride = blockDim.x; stride > 0; stride >>= 1) {               // Уменьшаем stride в 2 раза
        __syncthreads();                                                    // Синхронизация перед шагом
        int idx = (tid + 1) * stride * 2 - 1;                                // Индекс узла down-sweep
        if (idx < 2 * blockDim.x) {                                         // Проверка границы
            float t = s[idx - stride];                                      // Сохраняем левый элемент
            s[idx - stride] = s[idx];                                       // Переписываем левый
            s[idx] += t;                                                    // Добавляем t к правому
        }                                                                   // Конец if
    }                                                                       // Конец for down-sweep
    __syncthreads();                                                        // Финальная синхронизация

    // Сейчас s[] содержит EXCLUSIVE scan.                                     // Комментарий: что сейчас в shared
    // Делаем INCLUSIVE: out[i] = exclusive[i] + in[i]                         // Комментарий: перевод exclusive->inclusive
    if (i1 < n) out[i1] = s[tid] + x1;                                      // Inclusive значение для i1
    if (i2 < n) out[i2] = s[tid + blockDim.x] + x2;                          // Inclusive значение для i2
}                                                                           // Конец ядра scan_block_blelloch_inclusive

// 2) Добавляем смещение блока (просканированные суммы блоков) ко всем элементам блока // Описание второго шага
__global__ void add_block_offsets(float* data,                               // Массив данных (уже scanned)
                                  const float* __restrict__ scanned_block_sums, // Просканированные суммы блоков
                                  int n)                                    // Размер массива
{                                                                           // Начало ядра
    int blockStart = 2 * blockDim.x * blockIdx.x;                            // Старт индекса блока
    int tid = threadIdx.x;                                                  // ID потока в блоке

    int i1 = blockStart + tid;                                              // Индекс 1-го элемента
    int i2 = blockStart + tid + blockDim.x;                                 // Индекс 2-го элемента

    // scanned_block_sums — INCLUSIVE scan сумм блоков                         // Пояснение структуры scanned_block_sums
    // Для блока 0 offset = 0, для блока k offset = scanned_block_sums[k-1]   // Пояснение вычисления offset
    float offset = 0.0f;                                                    // Смещение для блока
    if (blockIdx.x > 0) offset = scanned_block_sums[blockIdx.x - 1];         // Берём сумму предыдущих блоков

    if (i1 < n) data[i1] += offset;                                         // Добавляем offset к i1
    if (i2 < n) data[i2] += offset;                                         // Добавляем offset к i2
}                                                                           // Конец ядра add_block_offsets

// CPU inclusive prefix sum                                                  // Комментарий: CPU реализация scan
static std::vector<float> cpu_inclusive_scan(const std::vector<float>& a)   // Функция CPU scan
{                                                                           // Начало функции
    std::vector<float> out(a.size());                                       // Выходной вектор
    float run = 0.0f;                                                       // Бегущая сумма
    for (size_t i = 0; i < a.size(); ++i) {                                 // Проход по элементам
        run += a[i];                                                        // Обновляем сумму
        out[i] = run;                                                       // Записываем prefix sum
    }                                                                       // Конец цикла
    return out;                                                             // Возвращаем результат
}                                                                           // Конец cpu_inclusive_scan

// GPU multi-block inclusive scan                                             // Комментарий: GPU scan много-блочный
static void gpu_inclusive_scan(const std::vector<float>& h_in,              // Входной массив CPU
                               std::vector<float>& h_out,                  // Выходной массив CPU (результат GPU)
                               int blockSize = 256)                         // Размер блока
{                                                                           // Начало функции
    int n = (int)h_in.size();                                               // Размер массива
    h_out.resize(n);                                                        // Выделяем место под результат

    float *d_in = nullptr, *d_out = nullptr;                                // Указатели на GPU память
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));                       // Выделяем d_in
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));                      // Выделяем d_out
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice)); // Копируем вход CPU->GPU

    // Каждый блок обрабатывает 2*blockSize элементов                          // Пояснение: сколько обрабатывает блок
    int elemsPerBlock = 2 * blockSize;                                      // Кол-во элементов на блок
    int numBlocks = (n + elemsPerBlock - 1) / elemsPerBlock;                // Кол-во блоков (ceil)

    float* d_block_sums = nullptr;                                          // Указатель на суммы блоков (GPU)
    CUDA_CHECK(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));       // Выделяем память под суммы блоков

    // 1) scan по блокам                                                     // Шаг 1: локальный scan
    size_t shmBytes = elemsPerBlock * sizeof(float);                        // Shared bytes = 2*blockSize*sizeof(float)
    scan_block_blelloch_inclusive<<<numBlocks, blockSize, shmBytes>>>(d_in, d_out, d_block_sums, n); // Запуск scan по блокам
    CUDA_CHECK(cudaGetLastError());                                         // Проверка ошибок запуска
    CUDA_CHECK(cudaDeviceSynchronize());                                    // Ожидаем завершения

    // 2) если блок один — готово                                            // Если всё влезло в один блок
    if (numBlocks > 1) {                                                    // Если блоков больше 1
        // Нужно просканировать d_block_sums (inclusive scan), чтобы получить offset-ы блоков // Зачем нужно
        // Для простоты (и корректности) сделаем это на CPU (для задания обычно ок).          // Выбор CPU подхода
        // Если нужно полностью на GPU — скажи, дам рекурсивный вариант.                     // Примечание/опция
        std::vector<float> h_block_sums(numBlocks);                          // Буфер на CPU под суммы блоков
        CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost)); // GPU->CPU копирование

        auto h_scanned = cpu_inclusive_scan(h_block_sums);                   // CPU inclusive scan сумм блоков

        float* d_scanned = nullptr;                                         // GPU массив для scanned block sums
        CUDA_CHECK(cudaMalloc(&d_scanned, numBlocks * sizeof(float)));      // Выделение на GPU
        CUDA_CHECK(cudaMemcpy(d_scanned, h_scanned.data(), numBlocks * sizeof(float), cudaMemcpyHostToDevice)); // CPU->GPU копирование

        // 3) добавить offset-ы к каждому блоку                               // Шаг 3: прибавляем offsets
        add_block_offsets<<<numBlocks, blockSize>>>(d_out, d_scanned, n);   // Запуск ядра добавления offset
        CUDA_CHECK(cudaGetLastError());                                     // Проверка ошибок ядра
        CUDA_CHECK(cudaDeviceSynchronize());                                // Синхронизация

        CUDA_CHECK(cudaFree(d_scanned));                                    // Освобождаем d_scanned
    }                                                                       // Конец if numBlocks > 1

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // Копируем результат GPU->CPU

    CUDA_CHECK(cudaFree(d_block_sums));                                     // Освобождаем d_block_sums
    CUDA_CHECK(cudaFree(d_out));                                            // Освобождаем d_out
    CUDA_CHECK(cudaFree(d_in));                                             // Освобождаем d_in
}                                                                           // Конец gpu_inclusive_scan

int main()                                                                  // Точка входа
{                                                                           // Начало main
    const int N = 100000; // тест                                            // Размер тестового массива
    std::vector<float> a(N);                                                // Вектор входных данных

    // тестовые данные: 1,1,1,... (тогда prefix[i] = i+1)                    // Пояснение теста
    for (int i = 0; i < N; ++i) a[i] = 1.0f;                                // Заполняем единицами

    // CPU                                                                   // CPU scan
    auto cpu = cpu_inclusive_scan(a);                                       // CPU reference

    // GPU                                                                   // GPU scan
    std::vector<float> gpu;                                                 // Вектор для результата GPU
    gpu_inclusive_scan(a, gpu, 256);                                        // Запуск scan на GPU

    // Проверка                                                              // Проверка результатов
    double maxAbsDiff = 0.0;                                                // Максимальная абсолютная разница
    for (int i = 0; i < N; ++i) {                                           // Проходим по всем элементам
        double d = std::abs((double)cpu[i] - (double)gpu[i]);               // Разница CPU vs GPU
        if (d > maxAbsDiff) maxAbsDiff = d;                                 // Обновляем максимум
    }                                                                       // Конец цикла

    std::cout << std::fixed << std::setprecision(6);                        // Формат вывода
    std::cout << "N = " << N << "\n";                                       // Вывод N
    std::cout << "CPU last = " << cpu.back() << "\n";                       // Последний элемент CPU scan
    std::cout << "GPU last = " << gpu.back() << "\n";                       // Последний элемент GPU scan
    std::cout << "Max abs diff = " << maxAbsDiff << "\n";                   // Максимальная ошибка

    const double eps = 1e-2;                                                // Допуск
    if (maxAbsDiff < eps) std::cout << "✅ OK: префиксная сумма корректна\n"; // Успех
    else                 std::cout << "❌ FAIL: есть расхождения\n";         // Ошибка

    return 0;                                                               // Выход
}                                                                           // Конец main
