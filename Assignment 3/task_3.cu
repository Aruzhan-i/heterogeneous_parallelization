#include <cuda_runtime.h>                                      // Подключение CUDA Runtime API
#include <iostream>                                           // Потоки ввода/вывода
#include <vector>                                             // Контейнер std::vector
#include <iomanip>                                            // Форматированный вывод

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) do {                                 /* Начало макроса-обёртки */ \
    cudaError_t err = (call);                                 /* Выполнение CUDA-вызова и сохранение кода ошибки */ \
    if (err != cudaSuccess) {                                 /* Проверка: произошла ли ошибка */ \
        std::cerr << "CUDA error: "                           /* Начало вывода сообщения об ошибке */ \
                  << cudaGetErrorString(err)                  /* Текстовое описание ошибки CUDA */ \
                  << " at " << __FILE__                       /* Имя файла, где возникла ошибка */ \
                  << ":" << __LINE__                          /* Номер строки, где возникла ошибка */ \
                  << std::endl;                               /* Перевод строки */ \
        std::exit(1);                                         /* Завершение программы с кодом ошибки */ \
    }                                                         /* Конец блока if */ \
} while(0)                                                    /* Макрос оформлен как безопасный блок */

// CUDA-ядро с коалесцированным доступом к памяти
__global__ void kernel_coalesced(const float* __restrict__ in,// Указатель на входной массив
                                 float* __restrict__ out,    // Указатель на выходной массив
                                 int n)                      // Размер массива
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;            // Вычисление глобального индекса потока
    if (i < n) out[i] = in[i] * 1.1f + 2.0f;                  // Последовательный доступ: умножение и прибавление
}

// CUDA-ядро с некоалесцированным доступом к памяти
__global__ void kernel_noncoalesced(const float* __restrict__ in,// Указатель на входной массив
                                    float* __restrict__ out,    // Указатель на выходной массив
                                    int n, int stride)          // Размер массива и шаг доступа
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;            // Вычисление глобального индекса потока
    if (i < n) {                                              // Проверка выхода за границы
        int j = (int)((1LL * i * stride) % n);                // Вычисление псевдослучайного индекса
        out[j] = in[j] * 1.1f + 2.0f;                          // Запись по неупорядоченному адресу
    }
}

// Функция заполнения вектора тестовыми данными
static void fill_data(std::vector<float>& v) {                // Принимает вектор по ссылке
    for (size_t i = 0; i < v.size(); ++i)                     // Проход по всем элементам
        v[i] = 0.001f * (float)(i % 1000);                    // Заполнение повторяющимся шаблоном
}

// Функция измерения времени для коалесцированного ядра
static float time_coalesced(const float* d_in, float* d_out,  // Указатели на вход и выход на GPU
                            int n, int block, int iters)      // Размер, блок и число итераций
{
    int grid = (n + block - 1) / block;                       // Расчёт количества блоков
    cudaEvent_t s, e;                                         // CUDA-события для таймера
    CUDA_CHECK(cudaEventCreate(&s));                          // Создание события начала
    CUDA_CHECK(cudaEventCreate(&e));                          // Создание события конца

    for (int i = 0; i < 5; ++i)                                // Прогревочные запуски
        kernel_coalesced<<<grid, block>>>(d_in, d_out, n);    // Запуск ядра без замера
    CUDA_CHECK(cudaDeviceSynchronize());                      // Ожидание завершения прогрева

    CUDA_CHECK(cudaEventRecord(s));                           // Запись события начала
    for (int i = 0; i < iters; ++i)                            // Основной цикл измерений
        kernel_coalesced<<<grid, block>>>(d_in, d_out, n);    // Запуск ядра
    CUDA_CHECK(cudaEventRecord(e));                           // Запись события конца
    CUDA_CHECK(cudaEventSynchronize(e));                      // Ожидание завершения

    float ms = 0.f;                                           // Переменная для времени в мс
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));              // Получение времени между событиями
    CUDA_CHECK(cudaEventDestroy(s));                          // Удаление события начала
    CUDA_CHECK(cudaEventDestroy(e));                          // Удаление события конца
    return ms / iters;                                        // Возврат среднего времени одного запуска
}

// Функция измерения времени для некоалесцированного ядра
static float time_noncoalesced(const float* d_in, float* d_out,// Указатели на вход и выход на GPU
                               int n, int block, int iters,   // Размер, блок и число итераций
                               int stride)                   // Шаг доступа к памяти
{
    int grid = (n + block - 1) / block;                       // Расчёт количества блоков
    cudaEvent_t s, e;                                         // CUDA-события для таймера
    CUDA_CHECK(cudaEventCreate(&s));                          // Создание события начала
    CUDA_CHECK(cudaEventCreate(&e));                          // Создание события конца

    for (int i = 0; i < 5; ++i)                                // Прогревочные запуски
        kernel_noncoalesced<<<grid, block>>>(d_in, d_out, n, stride); // Запуск ядра без замера
    CUDA_CHECK(cudaDeviceSynchronize());                      // Ожидание завершения прогрева

    CUDA_CHECK(cudaEventRecord(s));                           // Запись события начала
    for (int i = 0; i < iters; ++i)                            // Основной цикл измерений
        kernel_noncoalesced<<<grid, block>>>(d_in, d_out, n, stride); // Запуск ядра
    CUDA_CHECK(cudaEventRecord(e));                           // Запись события конца
    CUDA_CHECK(cudaEventSynchronize(e));                      // Ожидание завершения

    float ms = 0.f;                                           // Переменная для времени в мс
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));              // Получение времени между событиями
    CUDA_CHECK(cudaEventDestroy(s));                          // Удаление события начала
    CUDA_CHECK(cudaEventDestroy(e));                          // Удаление события конца
    return ms / iters;                                        // Возврат среднего времени одного запуска
}

int main() {                                                  // Точка входа в программу
    const int N = 1'000'000;                                  // Количество элементов массива
    const size_t bytes = (size_t)N * sizeof(float);          // Размер массива в байтах

    const int block = 256;                                    // Размер блока потоков
    const int grid  = (N + block - 1) / block;                // Количество блоков в сетке
    const int iters = 300;                                    // Количество итераций для замеров

    // Для N=1,000,000 берём нечётный stride, не кратный 2 или 5
    const int stride = 999'983;                               // Шаг доступа для некоалесцированного ядра

    std::vector<float> h_in(N), h_out(N);                     // Векторы на стороне CPU
    fill_data(h_in);                                          // Заполнение входного массива

    float *d_in=nullptr, *d_out=nullptr;                      // Указатели на память GPU
    CUDA_CHECK(cudaMalloc(&d_in, bytes));                     // Выделение памяти под входной массив
    CUDA_CHECK(cudaMalloc(&d_out, bytes));                    // Выделение памяти под выходной массив
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice)); // Копирование данных на GPU

    std::cout << "=== Параметры ===\n";                       // Заголовок блока параметров
    std::cout << "N = " << N << "\n";                          // Вывод размера массива
    std::cout << "block = " << block << ", grid = " << grid << "\n"; // Вывод конфигурации сетки
    std::cout << "iters = " << iters << "\n";                  // Вывод числа итераций
    std::cout << "stride (не коалесц.) = " << stride << "\n\n";// Вывод шага доступа

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));                   // Обнуление выходного массива
    float t_coal = time_coalesced(d_in, d_out, N, block, iters); // Замер коалесцированного ядра

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));                   // Повторное обнуление выходного массива
    float t_non  = time_noncoalesced(d_in, d_out, N, block, iters, stride); // Замер некоалесцированного ядра

    double gb_moved = (2.0 * (double)bytes) / 1e9;             // Объём переданных данных (чтение + запись)
    double bw_coal = gb_moved / (t_coal / 1e3);                // Пропускная способность для coalesced
    double bw_non  = gb_moved / (t_non  / 1e3);                // Пропускная способность для non-coalesced

    std::cout << "=== Результаты (среднее время ядра) ===\n";  // Заголовок результатов
    std::cout << std::fixed << std::setprecision(6);           // Формат вывода: фиксированный, 6 знаков
    std::cout << "Coalesced:     " << t_coal << " ms"          // Вывод времени для coalesced
              << " | ~" << std::setprecision(2) << bw_coal << " GB/s\n"; // Вывод пропускной способности
    std::cout << std::setprecision(6);                         // Возврат точности к 6 знакам
    std::cout << "Non-coalesced: " << t_non  << " ms"          // Вывод времени для non-coalesced
              << " | ~" << std::setprecision(2) << bw_non  << " GB/s\n"; // Вывод пропускной способности
    std::cout << std::setprecision(3)                           // Установка точности для коэффициента
              << "Замедление: " << (t_non / t_coal) << "x\n";  // Вывод замедления

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost)); // Копирование результата на CPU
    std::cout << "\nПример out[0..4] после ПОСЛЕДНЕГО ядра:\n";  // Заголовок примера вывода
    for (int i = 0; i < 5; ++i)                                 // Вывод первых пяти элементов
        std::cout << "out[" << i << "] = " << h_out[i] << "\n";// Печать значения элемента

    CUDA_CHECK(cudaFree(d_in));                                 // Освобождение памяти входного массива на GPU
    CUDA_CHECK(cudaFree(d_out));                                // Освобождение памяти выходного массива на GPU
    return 0;                                                   // Завершение программы
}
