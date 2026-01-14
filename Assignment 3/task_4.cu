#include <cuda_runtime.h>                                 // Подключение CUDA Runtime API
#include <iostream>                                      // Подключение потоков ввода/вывода
#include <vector>                                        // Подключение контейнера std::vector
#include <iomanip>                                       // Подключение форматированного вывода (setw, setprecision)
#include <limits>                                        // Подключение numeric_limits (infinity и т.п.)
#include <cmath>                                         // Подключение math-функций (на всякий случай)

// Макрос для проверки ошибок CUDA (сделано так, чтобы компилировалось: комментарии не после '\')
#define CUDA_CHECK(call) do {                             /* Начало макроса-обёртки */ \
    cudaError_t err = (call);                             /* Выполнение CUDA-вызова и сохранение кода ошибки */ \
    if (err != cudaSuccess) {                             /* Проверка: произошла ли ошибка */ \
        std::cerr << "CUDA error: "                       /* Начало вывода сообщения об ошибке */ \
                  << cudaGetErrorString(err)              /* Текстовое описание ошибки CUDA */ \
                  << " at " << __FILE__                   /* Имя файла, где произошла ошибка */ \
                  << ":" << __LINE__                      /* Номер строки, где произошла ошибка */ \
                  << std::endl;                           /* Перевод строки */ \
        std::exit(1);                                     /* Завершение программы с кодом ошибки */ \
    }                                                     /* Конец блока if */ \
} while(0)                                                /* Макрос оформлен как безопасный блок */

// CUDA-ядро: поэлементное сложение двух векторов
__global__ void vec_add(const float* __restrict__ a,      // Входной массив a (только чтение)
                        const float* __restrict__ b,      // Входной массив b (только чтение)
                        float* __restrict__ c,            // Выходной массив c (запись результата)
                        int n)                            // Количество элементов
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;        // Глобальный индекс элемента для текущего потока
    if (i < n) c[i] = a[i] + b[i];                        // Если индекс в пределах — записываем сумму
}

// Заполнение вектора тестовыми данными
static void fill_data(std::vector<float>& v, float seed) { // v — вектор для заполнения, seed — базовое значение
    for (size_t i = 0; i < v.size(); ++i) v[i] = seed + 0.001f * float(i % 1000); // Заполняем по шаблону
}

// Бенчмарк: измеряет среднее время выполнения ядра vec_add для заданного block (grid вычисляется внутри)
static float bench_kernel(const float* d_a, const float* d_b, float* d_c, // Указатели на GPU-данные
                          int n, int block, int warmup, int iters)        // n, размер блока, прогрев, итерации
{
    int grid = (n + block - 1) / block;                    // Расчёт grid: сколько блоков нужно для n элементов

    // прогрев
    for (int i = 0; i < warmup; ++i)                        // Цикл прогревочных запусков
        vec_add<<<grid, block>>>(d_a, d_b, d_c, n);         // Запуск ядра без замера времени
    CUDA_CHECK(cudaDeviceSynchronize());                    // Дожидаемся завершения прогрева

    cudaEvent_t s, e;                                       // CUDA-события для измерения времени
    CUDA_CHECK(cudaEventCreate(&s));                        // Создание события старта
    CUDA_CHECK(cudaEventCreate(&e));                        // Создание события конца

    CUDA_CHECK(cudaEventRecord(s));                         // Запись события старта
    for (int i = 0; i < iters; ++i)                         // Цикл измеряемых запусков
        vec_add<<<grid, block>>>(d_a, d_b, d_c, n);         // Запуск ядра
    CUDA_CHECK(cudaEventRecord(e));                         // Запись события конца
    CUDA_CHECK(cudaEventSynchronize(e));                    // Ожидание завершения всех запусков до события e

    float ms = 0.f;                                         // Переменная для времени в миллисекундах
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));             // Вычисление времени между событиями
    CUDA_CHECK(cudaEventDestroy(s));                         // Удаление события старта
    CUDA_CHECK(cudaEventDestroy(e));                         // Удаление события конца

    return ms / iters;                                      // Возврат среднего времени одного запуска ядра
}

int main() {                                                // Главная функция программы
    // Можно поставить N = 1<<24 как в твоём Assignment 2
    const int N = 1 << 24;                                   // Размер массива: 2^24 элементов
    const size_t bytes = (size_t)N * sizeof(float);          // Размер одного массива в байтах

    // Кандидаты размеров блока (можно расширять)
    std::vector<int> candidates = {64, 128, 256, 512};       // Набор вариантов block для теста

    const int warmup = 5;                                    // Количество прогревочных итераций
    const int iters  = 50;                                   // Количество итераций для усреднения

    // “Неоптимальная” конфигурация (фиксируем для сравнения)
    const int bad_block = 64;                                // Выбранный “плохой” block для сравнения

    // Хост
    std::vector<float> h_a(N), h_b(N);                       // Векторы на CPU для a и b
    fill_data(h_a, 1.0f);                                    // Заполнение a тестовыми значениями
    fill_data(h_b, 2.0f);                                    // Заполнение b тестовыми значениями

    // Девайс
    float *d_a=nullptr, *d_b=nullptr, *d_c=nullptr;          // Указатели на память GPU
    CUDA_CHECK(cudaMalloc(&d_a, bytes));                     // Выделение памяти под a на GPU
    CUDA_CHECK(cudaMalloc(&d_b, bytes));                     // Выделение памяти под b на GPU
    CUDA_CHECK(cudaMalloc(&d_c, bytes));                     // Выделение памяти под c на GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice)); // Копирование a: CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice)); // Копирование b: CPU -> GPU

    std::cout << "=== Автоподбор конфигурации (vec_add) ===\n"; // Заголовок вывода
    std::cout << "N = " << N << "\n";                             // Вывод N
    std::cout << "warmup = " << warmup << ", iters = " << iters << "\n\n"; // Вывод параметров замера

    std::cout << std::left                                     // Вывод таблицы в левом выравнивании
              << std::setw(10) << "Block"                       // Заголовок колонки Block
              << std::setw(12) << "Grid"                        // Заголовок колонки Grid
              << std::setw(14) << "Avg(ms)"                     // Заголовок колонки среднего времени
              << "\n";                                          // Переход на новую строку
    std::cout << std::string(38, '-') << "\n";                  // Разделительная линия

    float best_t = std::numeric_limits<float>::infinity();      // Лучшее (минимальное) время, стартуем с +inf
    int best_block = -1;                                        // Лучший block (пока не определён)

    for (int block : candidates) {                              // Перебор кандидатов block
        CUDA_CHECK(cudaMemset(d_c, 0, bytes));                  // Обнуление выходного массива на GPU
        float t = bench_kernel(d_a, d_b, d_c, N, block, warmup, iters); // Замер среднего времени для block
        int grid = (N + block - 1) / block;                     // Расчёт grid для текущего block

        std::cout << std::left                                  // Печать строки таблицы
                  << std::setw(10) << block                     // Вывод block
                  << std::setw(12) << grid                      // Вывод grid
                  << std::setw(14) << std::fixed << std::setprecision(4) << t // Вывод времени с 4 знаками
                  << "\n";                                      // Переход на новую строку

        if (t < best_t) {                                       // Если время лучше текущего лучшего
            best_t = t;                                         // Обновляем лучшее время
            best_block = block;                                 // Запоминаем лучший block
        }                                                       // Конец условия обновления лучшего варианта
    }                                                           // Конец перебора кандидатов

    int best_grid = (N + best_block - 1) / best_block;          // Вычисление grid для лучшего block
    int bad_grid  = (N + bad_block  - 1) / bad_block;           // Вычисление grid для “плохого” block

    // Замер “плохой” отдельно (честно ещё раз)
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));                       // Обнуление выходного массива
    float t_bad = bench_kernel(d_a, d_b, d_c, N, bad_block, warmup, iters); // Повторный замер для bad_block

    // Замер “лучшей” отдельно
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));                       // Обнуление выходного массива
    float t_best = bench_kernel(d_a, d_b, d_c, N, best_block, warmup, iters); // Повторный замер для best_block

    double speedup = t_bad / t_best;                             // Расчёт ускорения: во сколько раз стало быстрее
    double improvement_pct = (t_bad - t_best) / t_bad * 100.0;   // Процентное снижение времени

    std::cout << "\n=== Итог сравнения ===\n";                   // Заголовок итогового сравнения
    std::cout << "Неоптимальная: block = " << bad_block          // Вывод информации о плохой конфигурации
              << ", grid = " << bad_grid                         // Вывод grid для bad_block
              << ", time = " << std::fixed << std::setprecision(4) << t_bad << " ms\n"; // Вывод времени
    std::cout << "Оптимальная:   block = " << best_block         // Вывод информации о лучшей конфигурации
              << ", grid = " << best_grid                        // Вывод grid для best_block
              << ", time = " << std::fixed << std::setprecision(4) << t_best << " ms\n"; // Вывод времени
    std::cout << "Ускорение: " << std::setprecision(3) << speedup // Вывод ускорения
              << "x (снижение времени на " << std::setprecision(2) << improvement_pct << "%)\n"; // Вывод процента

    CUDA_CHECK(cudaFree(d_a));                                   // Освобождение памяти d_a на GPU
    CUDA_CHECK(cudaFree(d_b));                                   // Освобождение памяти d_b на GPU
    CUDA_CHECK(cudaFree(d_c));                                   // Освобождение памяти d_c на GPU
    return 0;                                                    // Завершение программы без ошибок
}                                                                // Конец main
