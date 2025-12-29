// Подключение основной библиотеки CUDA Runtime
// Содержит функции управления памятью, ядрами и устройством
#include <cuda_runtime.h>

// Подключение параметров запуска CUDA-ядер
// Используется для threadIdx, blockIdx, blockDim и т.д.
#include <device_launch_parameters.h>

// Подключение стандартной библиотеки ввода-вывода
#include <iostream>

// Подключение контейнера vector для хранения данных на CPU
#include <vector>

// Подключение библиотеки генерации случайных чисел
#include <random>

// Подключение библиотеки для измерения времени
#include <chrono>

// Подключение библиотеки с граничными значениями типов (INT_MAX)
#include <climits>

// Макрос для проверки ошибок CUDA-вызовов
#define CUDA_CHECK(call) \
    do { \
        /* Выполнение CUDA-функции */ \
        cudaError_t err = call; \
        /* Проверка результата выполнения */ \
        if (err != cudaSuccess) { \
            /* Вывод сообщения об ошибке */ \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            /* Аварийное завершение программы */ \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =====================================================
// 1. Сортировка блока (Bitonic Sort в shared memory)
// =====================================================

// CUDA-ядро для сортировки одного блока данных
__global__ void blockSort(int* data, int n) {

    // Объявление динамической shared memory
    extern __shared__ int s[];

    // Локальный индекс потока внутри блока
    int tid = threadIdx.x;

    // Глобальный индекс элемента массива
    int gid = blockIdx.x * blockDim.x + tid;

    // Загрузка элемента в shared memory или INT_MAX при выходе за границы
    s[tid] = (gid < n) ? data[gid] : INT_MAX;

    // Синхронизация всех потоков блока
    __syncthreads();

    // Внешний цикл битонической сортировки
    for (int k = 2; k <= blockDim.x; k <<= 1) {

        // Внутренний цикл сравнения элементов
        for (int j = k >> 1; j > 0; j >>= 1) {

            // Вычисление индекса партнёра для сравнения
            int ixj = tid ^ j;

            // Проверка, чтобы каждый обмен выполнялся один раз
            if (ixj > tid) {

                // Проверка направления сортировки (возрастание)
                if ((tid & k) == 0) {

                    // Обмен элементов при необходимости
                    if (s[tid] > s[ixj]) {
                        int tmp = s[tid];
                        s[tid] = s[ixj];
                        s[ixj] = tmp;
                    }

                } else {

                    // Проверка направления сортировки (убывание)
                    if (s[tid] < s[ixj]) {
                        int tmp = s[tid];
                        s[tid] = s[ixj];
                        s[ixj] = tmp;
                    }
                }
            }

            // Синхронизация после каждого шага
            __syncthreads();
        }
    }

    // Запись отсортированных данных обратно в глобальную память
    if (gid < n)
        data[gid] = s[tid];
}

// =====================================================
// 2. Слияние отсортированных сегментов
// =====================================================

// CUDA-ядро для слияния двух отсортированных сегментов
__global__ void mergeKernel(int* input, int* output, int width, int n) {

    // Глобальный индекс потока
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Начальный индекс сегмента
    int start = tid * width * 2;

    // Проверка выхода за границы массива
    if (start >= n) return;

    // Граница первого сегмента
    int mid = min(start + width, n);

    // Граница второго сегмента
    int end = min(start + 2 * width, n);

    // Индексы для слияния
    int i = start, j = mid, k = start;

    // Основной цикл слияния
    while (i < mid && j < end) {
        output[k++] = (input[i] < input[j]) ? input[i++] : input[j++];
    }

    // Копирование оставшихся элементов первого сегмента
    while (i < mid) output[k++] = input[i++];

    // Копирование оставшихся элементов второго сегмента
    while (j < end) output[k++] = input[j++];
}

// =====================================================
// 3. Основная программа
// =====================================================

// Функция запуска теста сортировки
void runTest(int n) {

    // Вектор данных на CPU
    std::vector<int> h_data(n);

    // Генератор псевдослучайных чисел
    std::mt19937 gen(42);

    // Диапазон случайных чисел
    std::uniform_int_distribution<int> dist(1, 100000);

    // Заполнение массива случайными числами
    for (int i = 0; i < n; i++)
        h_data[i] = dist(gen);

    // Указатели на память GPU
    int* d_a;
    int* d_b;

    // Выделение памяти на GPU
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(int)));

    // Копирование данных с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_data.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Размер блока CUDA
    int blockSize = 256;

    // Количество блоков
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Фиксация времени начала
    auto start = std::chrono::high_resolution_clock::now();

    // Сортировка каждого блока отдельно
    blockSort<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_a, n);

    // Ожидание завершения ядра
    CUDA_CHECK(cudaDeviceSynchronize());

    // Итеративное слияние отсортированных сегментов
    for (int width = blockSize; width < n; width *= 2) {

        // Количество блоков для текущего шага
        int blocks = (n + 2 * width - 1) / (2 * width);

        // Запуск ядра слияния
        mergeKernel<<<blocks, 1>>>(d_a, d_b, width, n);

        // Синхронизация GPU
        CUDA_CHECK(cudaDeviceSynchronize());

        // Обмен указателей
        std::swap(d_a, d_b);
    }

    // Фиксация времени окончания
    auto end = std::chrono::high_resolution_clock::now();

    // Вывод размера массива
    std::cout << "Размер массива: " << n << std::endl;

    // Вывод времени выполнения сортировки на GPU
    std::cout << "Время GPU-сортировки: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " мс\n\n";

    // Освобождение памяти GPU
    cudaFree(d_a);
    cudaFree(d_b);
}

// Точка входа в программу
int main() {

    // Заголовок программы
    std::cout << "CUDA Merge Sort\n\n";

    // Тест для массива из 10 000 элементов
    runTest(10000);

    // Тест для массива из 100 000 элементов
    runTest(100000);

    // Завершение программы
    return 0;
}
